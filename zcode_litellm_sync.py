#!/usr/bin/env python3
"""Прописывает в конфиг ZCode модели, которые реально отдаёт LiteLLM.

ZCode держит провайдеров в ~/.zcode/cli/config.json (ключ `provider`, формат opencode)
и сам список моделей не запрашивает: его проверка соединения бьёт только в /chat/completions.
Скрипт читает у LiteLLM `GET /model_group/info` и дописывает недостающие модели.

По умолчанию ничего не меняет — печатает план. Запись только с --write.
"""

import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request

# Десктопный ZCode держит рядом второй файл: он решает, что показать в интерфейсе.
RULES_FILE = "provider_config.json"
# Встроенный каталог моделей ZCode (macOS). На других ОС путь передаётся через --zcode-builtin.
BUILTIN_CATALOG = "/Applications/ZCode.app/Contents/Resources/config/provider/zcode-builtin.json"
# Флаги, которые ручное правило ZCode обязано задать; берём их из каталога, чтобы ничего не выключить.
FLAGS = ("supportsJsonSchemaOutput", "supportsNativeWebSearch", "supportsMidConversationSystem")
INPUT_FLAGS = ("supportsVideo", "supportsPdf")


def load_builtin_rules(path):
    """modelRules встроенного каталога ZCode; нет файла — пустой список (флаги станут false)."""
    try:
        with open(path, encoding="utf-8") as fh:
            return ((json.load(fh).get("config") or {}).get("modelConfigRules") or {}).get("modelRules") or []
    except (OSError, ValueError):
        return []


def builtin_properties(builtin_rules, model_id):
    """Свойства модели по каталогу ZCode: все совпавшие по modelMatch правила, поздние поверх ранних."""
    props, fmt = {}, {}
    for rule in builtin_rules:
        try:
            if not re.fullmatch(rule.get("modelMatch") or "", model_id):
                continue
        except re.error:
            continue
        p = (rule.get("config") or {}).get("properties") or {}
        props.update({k: p[k] for k in FLAGS if isinstance(p.get(k), bool)})
        f = p.get("inputFormat") or {}
        fmt.update({k: f[k] for k in INPUT_FLAGS if isinstance(f.get(k), bool)})
    return props, fmt


def load_api_map(path, api_type):
    """`reasoningLevel.map` общего правила каталога для типа API провайдера: схема ZCode требует
    map в ручном правиле, а этот — тот же, что ZCode сам применил бы к модели. Нет — None."""
    try:
        with open(path, encoding="utf-8") as fh:
            rules = ((json.load(fh).get("config") or {}).get("modelConfigRules") or {}).get("modelApiRules") or []
    except (OSError, ValueError):
        return None
    for rule in rules:
        if rule.get("modelMatch") == ".*" and rule.get("apiTypeMatch") == api_type:
            m = (((rule.get("config") or {}).get("optionSpecs") or {}).get("reasoningLevel") or {}).get("map")
            if isinstance(m, str):
                return m
    return None


def builtin_levels(builtin_rules, model_id):
    """Уровни reasoning, которые ZCode сам даст модели по имени: последнее совпавшее правило с values."""
    levels = None
    for rule in builtin_rules:
        try:
            if not re.fullmatch(rule.get("modelMatch") or "", model_id):
                continue
        except re.error:
            continue
        values = (((rule.get("config") or {}).get("optionSpecs") or {}).get("reasoningLevel") or {}).get("values")
        if isinstance(values, list):
            levels = values
    return levels


def default_config_path():
    """Так же, как считает сам ZCode: ZCODE_DATA_BASE_DIR или домашний каталог, дальше .zcode/v2.

    Схема одинакова на macOS, Linux и Windows (там это %USERPROFILE%\\.zcode\\v2).
    """
    # HOME проверяем отдельно: на Windows os.path.expanduser его игнорирует и берёт USERPROFILE,
    # а ZCode читает именно HOME — в Git Bash и MSYS это разные каталоги.
    base = (
        os.environ.get("ZCODE_DATA_BASE_DIR", "").strip()
        or os.environ.get("HOME", "").strip()
        or os.path.expanduser("~")
    )
    return os.path.join(base, ".zcode", "v2", "config.json")
DEFAULT_CONTEXT = 128_000
DEFAULT_OUTPUT = 8_192

# Режимы, которые в списке моделей чата не нужны.
SKIP_MODES = {
    "embedding",
    "rerank",
    "moderation",
    "moderations",
    "image_generation",
    "audio_transcription",
    "audio_speech",
}


def proxy_root(base_url):
    """`https://host/v1` → `https://host`: model_group/info живёт в корне прокси."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def fetch_groups(base_url, api_key, timeout=30):
    req = urllib.request.Request(proxy_root(base_url) + "/model_group/info")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("data") or []


def reasoning_pinned_off(deployment):
    """Reasoning выключен на самом деплойменте — уровни от клиента он всё равно не примет.

    Из litellm_params читаем только эти поля: остальное там — ссылки на ключи провайдеров.
    """
    params = deployment.get("litellm_params") or {}
    thinking = params.get("thinking")
    return (
        params.get("reasoning_effort") == "none"
        or params.get("enable_thinking") is False
        or (isinstance(thinking, dict) and thinking.get("type") == "disabled")
    )


def key_models(deployments):
    """Модели, доступные ключу: имя → reasoning выключен на всех его деплойментах."""
    out = {}
    for d in deployments:
        name = d.get("model_name")
        if name:
            out[name] = out.get(name, True) and reasoning_pinned_off(d)
    return out


def fetch_key_models(base_url, api_key, timeout=30):
    """`/model_group/info` отдаёт и чужие для ключа группы (запрос к ним — 403), `/model/info` —
    только доступные ключу, но без уровней reasoning. Если `/model/info` не ответил или пуст —
    None: список не сужаем, как раньше.
    """
    req = urllib.request.Request(proxy_root(base_url) + "/model/info")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            models = key_models(json.loads(resp.read().decode("utf-8")).get("data") or [])
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        print("model/info: %s — показываю все группы, часть может ответить 403" % exc, file=sys.stderr)
        return None
    if not models:
        print("model/info: пустой список — показываю все группы", file=sys.stderr)
        return None
    return models


def is_chat_group(group):
    # mode у части рабочих моделей null — это не повод их прятать.
    return str(group.get("mode") or "").lower() not in SKIP_MODES


def _limit(value, fallback):
    return int(value) if isinstance(value, (int, float)) and value > 0 else fallback


def to_model(group, default_context=DEFAULT_CONTEXT, default_output=DEFAULT_OUTPUT, no_reasoning=False):
    """Модель в формате `provider.<id>.models.<id>` конфига ZCode."""
    vision = group.get("supports_vision") is True
    # У `*-no-reasoning` LiteLLM объявляет supports_reasoning, но деплоймент его выключает.
    reasoning = group.get("supports_reasoning") is True and not no_reasoning
    efforts = (group.get("supported_reasoning_efforts") or []) if reasoning else []
    model = {
        "limit": {
            "context": _limit(group.get("max_input_tokens"), default_context),
            "output": _limit(group.get("max_output_tokens"), default_output),
        },
        "modalities": {
            "input": ["text", "image"] if vision else ["text"],
            "output": ["text"],
        },
    }
    if efforts:
        # Только объявленные уровни: непринятый уровень = 400 у провайдера,
        # деплоймент уходит в cooldown, остальные получают 429.
        model["reasoning"] = {"enabled": True, "variants": list(efforts)}
    else:
        model["reasoning"] = reasoning
    return model


def build_models(groups, deleted=(), allowed=None, **kwargs):
    """allowed — результат key_models(); None значит «список ключа неизвестен, не сужаем»."""
    skip = {str(x).strip().lower() for x in deleted}
    return {
        g["model_group"]: to_model(g, no_reasoning=bool(allowed and allowed.get(g["model_group"])), **kwargs)
        for g in groups
        if is_chat_group(g)
        and g.get("model_group")
        and g["model_group"].lower() not in skip
        and (allowed is None or g["model_group"] in allowed)
    }


def merge_models(existing, discovered):
    """Ничего не затираем: добавляем новые модели и дозаполняем пустые поля у старых.

    Возвращает (модели, добавленные, дополненные).
    """
    merged = dict(existing)
    added, filled = [], []
    for model_id, fresh in discovered.items():
        current = merged.get(model_id)
        if current is None:
            merged[model_id] = fresh
            added.append(model_id)
            continue
        if not isinstance(current, dict):
            continue
        missing = {k: v for k, v in fresh.items() if k not in current}
        if missing:
            merged[model_id] = {**current, **missing}
            filled.append(model_id)
    return merged, added, filled


def merge_ids(existing, discovered, deleted=()):
    """Дописывает недостающие id в конец: порядок, выбранный в интерфейсе, не трогаем."""
    skip = {str(x).strip().lower() for x in deleted}
    have = {str(x).strip().lower() for x in existing}
    added = [m for m in discovered if m.lower() not in have and m.lower() not in skip]
    return list(existing) + added, added


def find_provider_rule(rules, provider_id):
    """Правило провайдера в provider_config.json — по нему интерфейс строит список моделей."""
    items = (((rules.get("config") or {}).get("providerConfigRules") or {}).get("providerRules")) or []
    for item in items:
        if isinstance(item, dict) and item.get("providerId") == provider_id:
            return item
    return None


def new_provider_rule(provider_id, provider):
    """Правило личного провайдера — в том виде, в каком его пишет сам ZCode (3.12).

    Удаляя провайдера в интерфейсе, ZCode стирает только это правило, а запись в config.json
    оставляет; без правила провайдер не виден, сколько моделей ему ни дописывай.
    Ключ и адрес берутся из той же записи config.json, где ZCode их уже хранит.
    """
    options = provider.get("options") or {}
    return {
        "providerId": provider_id,
        "providerName": provider.get("name") or provider_id,
        "config": {
            "group": "standard-personal",
            "access": {"type": "api-key", "apiKey": options.get("apiKey") or ""},
            "api": {"type": "openai-chat-completions", "baseUrl": options.get("baseURL")},
            "personalModelIds": [],
            "modelOrder": [],
        },
    }


def add_provider_rule(rules, rule):
    """Дописывает правило и ставит провайдера в конец providerOrder."""
    cfg = rules.setdefault("config", {})
    cfg.setdefault("providerConfigRules", {}).setdefault("providerRules", []).append(rule)
    order = cfg.setdefault("providerOrder", [])
    if rule["providerId"] not in order:
        order.append(rule["providerId"])


def zcode_levels(variants):
    """Уровни LiteLLM в словаре ZCode: «выключено» у него `disabled`, у LiteLLM — `none`.
    Правило API ZCode для `disabled` само шлёт провайдеру `reasoning_effort: "none"`."""
    return ["disabled" if v == "none" else v for v in variants]


def manual_rule(provider_id, model_id, model, builtin_rules=(), api_map=None):
    """Ручное правило ZCode с уровнями reasoning модели.

    ZCode берёт уровни не из config.json, а из встроенного каталога по регулярке на имя:
    `deepseek-v4-flash-no-reasoning` для него — deepseek-v4-flash с уровнями, а `coding-fast`
    (имя без семейства) попадает только в общее правило `.*` с on/off. Перекрыть это можно
    только ручным правилом, а в нём схема ZCode требует все флаги `properties` — их берём из
    того же каталога, чтобы не выключить модели JSON-вывод или видео.
    Без reasoning: `values: ["disabled"]`, `map: "{}"` — в запрос ничего не добавляем.
    С уровнями: `map` общего правила API из каталога (ZCode требует map в ручном правиле).
    """
    limit = model.get("limit") or {}
    image = "image" in ((model.get("modalities") or {}).get("input") or [])
    props, fmt = builtin_properties(builtin_rules, model_id)
    variants = (model.get("reasoning") or {}).get("variants") if isinstance(model.get("reasoning"), dict) else None
    level = {"values": zcode_levels(variants), "map": api_map} if variants else {"values": ["disabled"], "map": "{}"}
    return {
        "providerId": provider_id,
        "modelId": model_id,
        "config": {
            "properties": {
                "contextWindow": _limit(limit.get("context"), DEFAULT_CONTEXT),
                "inputFormat": {"supportsImage": image, **{k: fmt.get(k, False) for k in INPUT_FLAGS}},
                **{k: props.get(k, False) for k in FLAGS},
            },
            "optionSpecs": {
                "reasoningLevel": level,
                "maxOutputTokens": {"max": _limit(limit.get("output"), DEFAULT_OUTPUT)},
            },
        },
    }


def needs_manual_rule(model_id, model, builtin_rules, api_map=None):
    """Нужно ли ручное правило: reasoning выключен или каталог ZCode даст не те уровни."""
    reasoning = model.get("reasoning")
    if not reasoning:
        return True
    variants = reasoning.get("variants") if isinstance(reasoning, dict) else None
    # Без каталога не знаем, что ZCode покажет сам, и нечем заполнить map — уровни не трогаем.
    return bool(variants and builtin_rules and api_map) and builtin_levels(builtin_rules, model_id) != zcode_levels(variants)


def add_manual_rules(rules, provider_id, models, builtin_rules=(), api_map=None):
    """Ручные правила для моделей без reasoning и для тех, чьи уровни каталог ZCode
    по имени не угадывает. Существующие правила не трогаем:
    ни ручные (их мог поправить пользователь), ни «умные» — ZCode запрещает оба сразу.

    Возвращает id моделей, которым правило добавлено.
    """
    mcr = rules.setdefault("config", {}).setdefault("modelConfigRules", {})
    manual = mcr.setdefault("manualProviderModelRules", [])
    taken = {
        (r.get("providerId"), r.get("modelId"))
        for r in manual + (mcr.get("providerModelRules") or [])
        if isinstance(r, dict)
    }
    added = []
    for model_id in sorted(models):
        model = models[model_id]
        if (not isinstance(model, dict) or (provider_id, model_id) in taken
                or not needs_manual_rule(model_id, model, builtin_rules, api_map)):
            continue
        manual.append(manual_rule(provider_id, model_id, model, builtin_rules, api_map))
        added.append(model_id)
    return added


def pick_provider(config, provider_id):
    providers = config.get("provider") or {}
    if not providers:
        raise SystemExit("в конфиге нет ни одного провайдера (ключ `provider`)")
    if provider_id:
        if provider_id not in providers:
            raise SystemExit(
                "провайдер %r не найден, есть: %s" % (provider_id, ", ".join(sorted(providers)))
            )
        return provider_id, providers[provider_id]
    if len(providers) == 1:
        return next(iter(providers.items()))
    raise SystemExit(
        "провайдеров несколько — укажите --provider: " + ", ".join(sorted(providers))
    )


def write_config(path, config):
    """Пишем рядом и переименовываем: ZCode читает этот файл на старте и при смене модели."""
    tmp = path + ".tmp"
    try:
        shutil.copy2(path, path + ".bak")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except PermissionError as exc:
        # На Windows запущенный ZCode держит файл открытым и переименование падает.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise SystemExit("%s занят (%s). Закройте ZCode и повторите." % (path, exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=default_config_path(), help="путь к config.json ZCode")
    parser.add_argument("--provider", help="id провайдера в конфиге (если он там не один)")
    parser.add_argument("--write", action="store_true", help="применить изменения")
    parser.add_argument("--default-context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--default-output", type=int, default=DEFAULT_OUTPUT)
    parser.add_argument("--zcode-builtin", default=BUILTIN_CATALOG,
                        help="zcode-builtin.json из приложения ZCode (флаги моделей для ручных правил)")
    args = parser.parse_args(argv)

    try:
        with open(args.config, encoding="utf-8") as fh:
            config = json.load(fh)
    except OSError as exc:
        raise SystemExit("не читается %s: %s" % (args.config, exc))
    except ValueError as exc:
        raise SystemExit("%s — битый JSON: %s" % (args.config, exc))

    provider_id, provider = pick_provider(config, args.provider)
    options = provider.get("options") or {}
    base_url = options.get("baseURL")
    if not base_url:
        raise SystemExit("у провайдера %r не задан options.baseURL" % provider_id)
    api_key = os.environ.get("LITELLM_API_KEY") or options.get("apiKey")

    try:
        groups = fetch_groups(base_url, api_key)
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        raise SystemExit("LiteLLM не ответил (%s): %s" % (proxy_root(base_url), exc))

    allowed = fetch_key_models(base_url, api_key)
    deleted = ((provider.get("zcode") or {}).get("deletedModels")) or []
    discovered = build_models(
        groups,
        deleted=deleted,
        allowed=allowed,
        default_context=args.default_context,
        default_output=args.default_output,
    )
    existing = provider.get("models") or {}
    merged, added, filled = merge_models(existing, discovered)
    stale = sorted(set(existing) - set(discovered))

    rules_path = os.path.join(os.path.dirname(os.path.abspath(args.config)), RULES_FILE)
    rules = rule = None
    if os.path.exists(rules_path):
        try:
            with open(rules_path, encoding="utf-8") as fh:
                rules = json.load(fh)
        except ValueError as exc:
            raise SystemExit("%s — битый JSON: %s" % (rules_path, exc))
        rule = find_provider_rule(rules, provider_id)
    # Встроенные провайдеры (`builtin:*`) интерфейс показывает без правил — им не создаём.
    created_rule = rule is None and rules is not None and not provider_id.startswith("builtin:")
    if created_rule:
        rule = new_provider_rule(provider_id, provider)
        add_provider_rule(rules, rule)

    print("провайдер: %s (%s)" % (provider_id, base_url))
    print("у LiteLLM: %d моделей, в конфиге было: %d" % (len(discovered), len(existing)))
    for model_id in added:
        print("  + %s" % model_id)
    for model_id in filled:
        print("  ~ %s (дозаполнены поля)" % model_id)
    for model_id in stale:
        print("  ? %s — у LiteLLM нет или ключу недоступна, оставляю" % model_id)
    if deleted:
        print("  пропущены скрытые в ZCode: %s" % ", ".join(deleted))

    shown = list((rule.get("config") or {}).get("personalModelIds") or []) if rule else []
    order = list((rule.get("config") or {}).get("modelOrder") or []) if rule else []
    shown_new, shown_added = merge_ids(shown, sorted(discovered), deleted)
    order_new, _ = merge_ids(order, sorted(discovered), deleted)
    if created_rule:
        print("  + правило провайдера в %s: его не было, без него провайдер в интерфейсе не виден" % RULES_FILE)
    if rule is None and rules is not None:
        print("  в %s нет правила для этого провайдера — список в интерфейсе не трогаю" % RULES_FILE)
    elif rule is not None:
        print("в интерфейсе показано: %d, добавится: %d" % (len(shown), len(shown_added)))
        for model_id in shown_added:
            print("  + %s (в список интерфейса)" % model_id)

    manual_added = []
    if rule is not None:
        api_type = ((rule.get("config") or {}).get("api") or {}).get("type") or "openai-chat-completions"
        manual_added = add_manual_rules(rules, provider_id, merged, load_builtin_rules(args.zcode_builtin),
                                        load_api_map(args.zcode_builtin, api_type))
        by_id = {r.get("modelId"): r for r in rules["config"]["modelConfigRules"]["manualProviderModelRules"]
                 if r.get("providerId") == provider_id}
        for model_id in manual_added:
            levels = by_id[model_id]["config"]["optionSpecs"]["reasoningLevel"]["values"]
            print("  + %s: reasoning в селекторе — %s (ручное правило ZCode)" % (model_id, ", ".join(levels)))

    if not added and not filled and not shown_added and not created_rule and not manual_added:
        print("менять нечего")
        return 0
    if not args.write:
        print("\nэто предпросмотр, запись — с --write")
        return 0

    provider["models"] = merged
    write_config(args.config, config)
    if rule is not None and (shown_added or created_rule or manual_added):
        rule.setdefault("config", {})["personalModelIds"] = shown_new
        rule["config"]["modelOrder"] = order_new
        write_config(rules_path, rules)
        print("список интерфейса обновлён в %s (копия — %s.bak)" % (rules_path, rules_path))
    print("\nзаписано в %s (копия прежнего — %s.bak)" % (args.config, args.config))
    print("ZCode перечитывает конфиг при старте — перезапустите его")
    return 0


if __name__ == "__main__":
    sys.exit(main())
