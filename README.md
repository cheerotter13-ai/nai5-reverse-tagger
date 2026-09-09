# NAI5 reverse tagger

Local web app that turns one anime image into a NovelAI Diffusion V5 prompt: a base box, optional natural-language sentence, and per-character Danbooru tags with `source#` / `target#`.

It does **not** fold clothes, expressions, or actions into wildcards. Tags stay as recognized.

## What you get

```
Prompt:
2girls, yuri, bondage, nsfw, night, alley, cowboy shot

Two girls standing close together in a narrow brick alley, bodies pressed side by side.

Character 1:
girl, pink hair, long hair, pointy ears, white shorts, white thighhighs, gag, target#hug person

Character 2:
girl, pink hair, long hair, white dress, white gloves, hat, finger to mouth, source#hug person
```

- Base box: count, theme, short scene/camera tags, `nsfw`.
- Second paragraph: **one English sentence** for spatial layout and interactions that short tags cannot express.
- Character boxes: Danbooru-style short tags only. `source#` / `target#` / `mutual#` share the same verb.
- Artist names, quality tags, and glossy/oiled material tags are dropped.

## Requirements

- Python 3.11+
- A vision-capable OpenAI-compatible endpoint (for example a local Grok gateway)
- Optional: [WD14 EVA02](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) ONNX for extra Danbooru tags

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Copy `.env.example` values into your environment:

| Variable | Meaning |
|---|---|
| `NAI5_TAGGER_VLM_BASE` | Chat completions base, default `http://127.0.0.1:8000/v1` |
| `NAI5_TAGGER_VLM_MODEL` | Vision model id, default `grok-4.6` |
| `NAI5_TAGGER_VLM_KEY` | Bearer token if the gateway requires one |
| `NAI5_TAGGER_WD14_DIR` | Folder with `model.onnx` and `selected_tags.csv`. Leave empty to skip WD14 |

Do not commit API keys.

## Run

```bash
python -m nai5_tagger.server
```

Open `http://127.0.0.1:18770/`. Drop an image. Copy each box into NovelAI V5.

CLI:

```bash
python -m nai5_tagger path/to/image.png
python -m nai5_tagger path/to/image.png --json
python -m nai5_tagger path/to/image.png --no-nl
```

## Tests

```bash
python -m pytest tests/ -q
```

## License

MIT. See [LICENSE](LICENSE).
