# zcode-litellm-sync

Writes the models of a LiteLLM proxy into [ZCode](https://z.ai)'s config, so you don't add them
by hand one at a time.

ZCode keeps its providers in `config.json` (key `provider`, opencode-shaped) and never asks a
provider for its model list — its connectivity check only calls `/chat/completions`, and its
plugin marketplace has no hook for providers or models. So discovery has to happen outside the
app: this script reads LiteLLM's `GET /model_group/info` and fills in what's missing.

Two files are involved, and both matter:

- `config.json` (`~/.zcode/v2/config.json` for the desktop app, `~/.zcode/cli/config.json` for the
  CLI) — the model definitions: limits, modalities, reasoning levels.
- `provider_config.json`, next to it — what the app actually *shows*. A personal provider lists
  its visible models in `providerConfigRules.providerRules[].config.personalModelIds`, with
  `modelOrder` beside it. A model defined only in `config.json` does not appear in the UI.

The script updates both: definitions in the first, missing ids appended to the end of the second.
Order you arranged in the UI is never rearranged, and nothing is ever removed from either list.

`/model_group/info` is used on purpose: it lists what your key's team actually has, while
`/v1/model/info` returns different things to different keys (sometimes nothing at all).

## Use

```sh
# посмотреть план — ничего не меняется
./zcode_litellm_sync.py

# применить
./zcode_litellm_sync.py --write
```

Options: `--config PATH` (default `~/.zcode/v2/config.json`; pass `~/.zcode/cli/config.json` for
the CLI layout), `--provider ID` (needed when the config has more than one), `--default-context` / `--default-output` for models LiteLLM reports no
limits for (defaults 128000 / 8192).

The proxy address and key are taken from the provider itself (`options.baseURL`,
`options.apiKey`); `$LITELLM_API_KEY` overrides the key. Nothing is written back into the key.

## What it will and won't do

- **Adds** models that LiteLLM serves and the config lacks.
- **Fills** only the fields an existing model entry doesn't have — your own `name`, limits or
  anything you edited in the UI stay as they are.
- **Never deletes.** A model that disappeared from LiteLLM is reported and left alone; so is a
  model you added by hand.
- **Respects the UI.** Models you hid in ZCode (`provider.<id>.zcode.deletedModels`) are skipped.
- **Touches only `provider.<id>.models`** and, in `provider_config.json`, only
  `personalModelIds` / `modelOrder` of that one provider. The rest of `config.json` — `mcp`, other providers,
  everything — keeps its content and key order; the file itself is re-serialized with
  `json.dump(indent=2)`, so whitespace may differ. The previous file is kept as `config.json.bak`.
- Embeddings, rerank and other non-chat modes are filtered out; a model with `mode: null` is kept
  (LiteLLM leaves it empty for plenty of working chat models).

Reasoning levels: when LiteLLM announces `supported_reasoning_efforts` for a model, they are
written as `reasoning: {enabled: true, variants: [...]}` — exactly the levels the model takes.
A level the provider does not take is a 400, which puts the deployment into cooldown and turns
into 429s for everyone on that proxy, so nothing is invented when LiteLLM announces nothing.

Run it with ZCode closed (or restart ZCode afterwards): the app rewrites this file from memory
and would overwrite the sync.

Discovered ≠ callable: `/model_group/info` lists the team's models, and a key may still be denied
a particular one (403 at request time).

## Development

```sh
python3 test_sync.py   # stdlib only
```

MIT
