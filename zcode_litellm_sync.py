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
import shutil
import sys
import urllib.error
import urllib.request

DEFAULT_CONFIG = os.path.expanduser("~/.zcode/cli/config.json")
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


def is_chat_group(group):
    # mode у части рабочих моделей null — это не повод их прятать.
    return str(group.get("mode") or "").lower() not in SKIP_MODES


def _limit(value, fallback):
    return int(value) if isinstance(value, (int, float)) and value > 0 else fallback


def to_model(group, default_context=DEFAULT_CONTEXT, default_output=DEFAULT_OUTPUT):
    """Модель в формате `provider.<id>.models.<id>` конфига ZCode."""
    vision = group.get("supports_vision") is True
    efforts = group.get("supported_reasoning_efforts") or []
    reasoning = group.get("supports_reasoning") is True
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


def build_models(groups, deleted=(), **kwargs):
    skip = {str(x).strip().lower() for x in deleted}
    return {
        g["model_group"]: to_model(g, **kwargs)
        for g in groups
        if is_chat_group(g) and g.get("model_group") and g["model_group"].lower() not in skip
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
    shutil.copy2(path, path + ".bak")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="путь к config.json ZCode")
    parser.add_argument("--provider", help="id провайдера в конфиге (если он там не один)")
    parser.add_argument("--write", action="store_true", help="применить изменения")
    parser.add_argument("--default-context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--default-output", type=int, default=DEFAULT_OUTPUT)
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

    deleted = ((provider.get("zcode") or {}).get("deletedModels")) or []
    discovered = build_models(
        groups,
        deleted=deleted,
        default_context=args.default_context,
        default_output=args.default_output,
    )
    existing = provider.get("models") or {}
    merged, added, filled = merge_models(existing, discovered)
    stale = sorted(set(existing) - set(discovered))

    print("провайдер: %s (%s)" % (provider_id, base_url))
    print("у LiteLLM: %d моделей, в конфиге было: %d" % (len(discovered), len(existing)))
    for model_id in added:
        print("  + %s" % model_id)
    for model_id in filled:
        print("  ~ %s (дозаполнены поля)" % model_id)
    for model_id in stale:
        print("  ? %s — у LiteLLM больше нет, оставляю" % model_id)
    if deleted:
        print("  пропущены скрытые в ZCode: %s" % ", ".join(deleted))

    if not added and not filled:
        print("менять нечего")
        return 0
    if not args.write:
        print("\nэто предпросмотр, запись — с --write")
        return 0

    provider["models"] = merged
    write_config(args.config, config)
    print("\nзаписано в %s (копия прежнего — %s.bak)" % (args.config, args.config))
    print("ZCode перечитывает конфиг при старте — перезапустите его")
    return 0


if __name__ == "__main__":
    sys.exit(main())
