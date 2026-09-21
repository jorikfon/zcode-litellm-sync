#!/usr/bin/env python3
"""Проверки маппинга и записи. Запуск: python3 test_sync.py"""

import json
import os
import tempfile

from zcode_litellm_sync import (
    build_models,
    find_provider_rule,
    merge_ids,
    merge_models,
    pick_provider,
    proxy_root,
    to_model,
    write_config,
)

# Срез живого ответа LiteLLM /model_group/info вместе с грязью:
# mode=null у рабочей чат-модели, лимиты null, эмбеддинги в общем списке.
GROUPS = [
    {
        "model_group": "deepseek-v4-flash",
        "mode": "chat",
        "max_input_tokens": 1000000.0,
        "max_output_tokens": 393216.0,
        "supports_vision": True,
        "supports_reasoning": True,
        "supported_reasoning_efforts": ["none", "low", "high", "max"],
    },
    {"model_group": "qwen3.8-max", "mode": None, "max_input_tokens": None},
    {"model_group": "ollama/bge-m3", "mode": "embedding"},
    {"model_group": "zai/glm-5.3", "mode": "chat", "supports_reasoning": True,
     "supported_reasoning_efforts": ["low", "high", "max"]},
]


def test_build():
    models = build_models(GROUPS)
    assert sorted(models) == ["deepseek-v4-flash", "qwen3.8-max", "zai/glm-5.3"], models
    assert models["deepseek-v4-flash"]["limit"] == {"context": 1000000, "output": 393216}
    assert models["deepseek-v4-flash"]["modalities"]["input"] == ["text", "image"]
    # Уровни строго объявленные: непринятый = 400 у провайдера и cooldown деплоймента.
    assert models["zai/glm-5.3"]["reasoning"] == {"enabled": True, "variants": ["low", "high", "max"]}
    # Ничего не объявлено — просто флаг, набор не выдумываем.
    assert models["qwen3.8-max"]["reasoning"] is False
    assert models["qwen3.8-max"]["limit"] == {"context": 128000, "output": 8192}


def test_deleted_skipped():
    models = build_models(GROUPS, deleted=["DeepSeek-V4-Flash"])
    assert "deepseek-v4-flash" not in models, "скрытая в ZCode модель не должна возвращаться"


def test_merge_preserves():
    existing = {
        "deepseek-v4-flash": {"name": "мой ярлык", "limit": {"context": 42}},
        "ручная-модель": {"name": "не трогать"},
    }
    merged, added, filled = merge_models(existing, build_models(GROUPS))
    assert merged["deepseek-v4-flash"]["limit"] == {"context": 42}, "правка руками важнее"
    assert merged["deepseek-v4-flash"]["modalities"]["output"] == ["text"], "пустое поле дозаполнено"
    assert merged["ручная-модель"] == {"name": "не трогать"}
    assert added == ["qwen3.8-max", "zai/glm-5.3"], added
    assert filled == ["deepseek-v4-flash"], filled


def test_merge_ids_for_ui_list():
    # Порядок в интерфейсе выбран человеком — новые id дописываем в конец, ничего не переставляя.
    shown = ["moonshot/kimi-k3", "zai/glm-5.3"]
    new, added = merge_ids(shown, ["deepseek-v4-flash", "Zai/GLM-5.3", "moonshot/kimi-k3"])
    assert new == ["moonshot/kimi-k3", "zai/glm-5.3", "deepseek-v4-flash"], new
    assert added == ["deepseek-v4-flash"], added
    # Скрытая в ZCode модель в список интерфейса не возвращается.
    _, added = merge_ids(shown, ["qwen3.8-max"], deleted=["QWEN3.8-MAX"])
    assert added == [], added


def test_find_provider_rule():
    rules = {"config": {"providerConfigRules": {"providerRules": [
        {"providerId": "other", "config": {}},
        {"providerId": "ours", "config": {"personalModelIds": ["a"]}},
    ]}}}
    assert find_provider_rule(rules, "ours")["config"]["personalModelIds"] == ["a"]
    assert find_provider_rule(rules, "нет такого") is None
    assert find_provider_rule({}, "ours") is None


def test_write_keeps_everything_else():
    config = {"mcp": {"servers": {"cognee": {"type": "http"}}},
              "provider": {"litellm": {"options": {"apiKey": "секрет", "baseURL": "x"}, "models": {}}}}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(config, fh)
        config["provider"]["litellm"]["models"] = {"m": {"limit": {"context": 1}}}
        write_config(path, config)
        written = json.load(open(path, encoding="utf-8"))
        assert written["mcp"]["servers"]["cognee"]["type"] == "http", "чужие ключи должны уцелеть"
        assert written["provider"]["litellm"]["options"]["apiKey"] == "секрет", "ключ не теряем"
        assert written["provider"]["litellm"]["models"] == {"m": {"limit": {"context": 1}}}
        assert json.load(open(path + ".bak", encoding="utf-8"))["provider"]["litellm"]["models"] == {}


def test_pick_provider_and_root():
    assert proxy_root("https://litellm.example/v1") == "https://litellm.example"
    assert proxy_root("https://litellm.example/") == "https://litellm.example"
    name, _ = pick_provider({"provider": {"only": {}}}, None)
    assert name == "only"
    try:
        pick_provider({"provider": {"a": {}, "b": {}}}, None)
    except SystemExit as exc:
        assert "--provider" in str(exc)
    else:
        raise AssertionError("при нескольких провайдерах нужен явный выбор")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("всё прошло")
