# zcode-litellm-sync

Writes the models of a LiteLLM proxy into [ZCode](https://z.ai)'s config, so you don't add them
by hand one at a time.

ZCode keeps its providers in `config.json` (key `provider`, opencode-shaped) and never asks a
provider for its model list — its connectivity check only calls `/chat/completions`, and its
plugin marketplace has no hook for providers or models. So discovery has to happen outside the
app: this script reads LiteLLM's `GET /model_group/info` and fills in what's missing.

Two files are involved, and both matter:

- `config.json` — the model definitions: limits, modalities, reasoning levels.
- `provider_config.json`, next to it — what the app actually *shows*. A personal provider lists
  its visible models in `providerConfigRules.providerRules[].config.personalModelIds`, with
  `modelOrder` beside it. A model defined only in `config.json` does not appear in the UI.

The script updates both: definitions in the first, missing ids appended to the end of the second.

Deleting a personal provider in ZCode removes only its rule from `provider_config.json` and leaves
the entry in `config.json`; models synced into such an entry stay invisible. So when the provider
has no rule, the script writes one the way ZCode does (`group: standard-personal`, the API key and
`baseURL` from the provider's own `config.json` entry, `openai-chat-completions`) and appends the
provider to `providerOrder`. Built-in providers (`builtin:*`) are left alone. If the entry has no
`apiKey`, the rule gets an empty one — enter the key in ZCode's provider settings.
Order you arranged in the UI is never rearranged, and nothing is ever removed from either list.

Limits and reasoning levels come from `/model_group/info`; the list is narrowed to the models
`GET /model/info` shows for your key, so a model the key may not call is not added only to answer
403. If `/model/info` fails or comes back empty, every model group is used, as before, and a
`model/info: …` line on stderr says why.

## Use

```sh
# preview — changes nothing
python3 zcode_litellm_sync.py

# apply
python3 zcode_litellm_sync.py --write
```

Options: `--config PATH`, `--provider ID` (needed when the config holds more than one, and ZCode
names custom providers by UUID), `--default-context` / `--default-output` for models LiteLLM
reports no limits for (defaults 128000 / 8192).

The proxy address and credentials are taken from the provider entry itself; `$LITELLM_API_KEY`
overrides the latter. Nothing is ever written back into that field, and it is never printed.

## Where the config lives

The script resolves the same way ZCode itself does: `$ZCODE_DATA_BASE_DIR`, else the home
directory, then `.zcode/v2/config.json`.

| | desktop app | CLI |
|---|---|---|
| macOS / Linux | `~/.zcode/v2/config.json` | `~/.zcode/cli/config.json` |
| Windows | `%USERPROFILE%\.zcode\v2\config.json` | `%USERPROFILE%\.zcode\cli\config.json` |

For the CLI layout pass the path explicitly with `--config`. On Windows run it as
`python zcode_litellm_sync.py` (or `py -3 …`); the script is stdlib-only, so nothing needs
installing. It is developed and tested on macOS — the Windows paths above come from ZCode's own
resolution logic, not from a test run.

## What it will and won't do

- **Adds** models that LiteLLM serves and the config lacks.
- **Fills** only the fields an existing model entry doesn't have — your own `name`, limits or
  anything you edited in the UI stay as they are.
- **Never deletes.** A model that disappeared from LiteLLM is reported and left alone; so is a
  model you added by hand.
- **Respects the UI.** Models you hid in ZCode (`provider.<id>.zcode.deletedModels`) are skipped.
- **Touches only `provider.<id>.models`** and, in `provider_config.json`, only
  `personalModelIds` / `modelOrder` of that one provider (or its whole rule, when it is missing,
  plus its place in `providerOrder`). The rest of `config.json` — `mcp`, other
  providers, everything — keeps its content and key order; the file itself is re-serialized with
  `json.dump(indent=2)`, so whitespace may differ. The previous file is kept as `config.json.bak`.
- Embeddings, rerank and other non-chat modes are filtered out; a model with `mode: null` is kept
  (LiteLLM leaves it empty for plenty of working chat models).

Reasoning levels: when LiteLLM announces `supported_reasoning_efforts` for a model, they are
written as `reasoning: {enabled: true, variants: [...]}` — exactly the levels the model takes.
A level the provider does not take is a 400, which puts the deployment into cooldown and turns
into 429s for everyone on that proxy, so nothing is invented when LiteLLM announces nothing. A model with reasoning switched off on the deployment itself (`reasoning_effort: "none"`, `enable_thinking: false` or `thinking.type: "disabled"` in `litellm_params` of every deployment, as `/model/info` shows them) is written
with `reasoning: false` — that is how `*-no-reasoning` groups look, even though LiteLLM reports
`supports_reasoning: true` for them. Only those fields are read from `litellm_params`; the rest of
it (provider credentials) is neither kept nor printed.

The script never rewrites a field that is already there, so an entry synced by an older version
keeps its old `reasoning` value and a model the key may not call stays in the config (reported as
`?`). Remove such entries by hand, or from ZCode, and run the sync again.

Run it with ZCode closed, then start ZCode: the app holds both files in memory and rewrites them,
so a sync applied underneath a running app is lost. On Windows a running ZCode also locks the
files — the script then stops with a message instead of leaving a half-written config.


## Development

```sh
python3 test_sync.py   # stdlib only
```

MIT
