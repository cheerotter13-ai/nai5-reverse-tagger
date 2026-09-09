# NAI5 提示词反推

把**一张二次元图**反推成 NovelAI Diffusion V5 能直接粘贴的多角色提示词。

普通 WD14 / Tagger 给你的是一条扁平 Danbooru 列表，没有「谁在左边、谁捆着谁、谁贴在谁脸上」。本工具按 **NAI5 主框 + 角色框** 输出，并写出官方施受语法 `source#` / `target#` / `mutual#`。

## 它能做什么

1. **按人分框**  
   主框只放人数、玩法、场景、镜头、`nsfw`。每个人的外观、衣服、表情、姿势进各自角色框，不串框。

2. **Danbooru 短 tag 骨架**  
   角色框用 NAI 吃得惯的短词：`pink hair, white thighhighs, bit gag, finger to mouth`，不是一整段英文描写。

3. **一句自然语言补空间和复杂互动**  
   短 tag 说不清的东西写在主框第二段：**谁贴着墙、谁从后面搂、谁坐在谁脸上、手伸进哪里**。只留一句，不把 tag 再复述一遍。

4. **NAI `source#` / `target#` / `mutual#`**  
   同一互动两边动词必须相同，例如：
   - `source#hug person` + `target#hug person`
   - `source#sitting on person` + `target#sitting on person`  
   这是 NAI5 多角色官方施受写法，扁平 Tagger 不会给你配好。

5. **可粘贴进 NAI5**  
   本地网页分框复制。不反推画师串、质量词、油光材质。

## 和常见 Tagger 差在哪

| | WD14 / 多数反推 | 本工具 |
|---|---|---|
| 输出形态 | 一条扁平 tag | NAI5 **主框 + Character 1…N** |
| 两人衣服 | 混在一起 | 分到各自角色框 |
| 施受关系 | 没有，或只剩 `hug` | `source#动词` / `target#动词`，动词对齐 |
| 空间关系 | 几乎没有 | 主框一句 NL 写站位、贴靠、复杂接触 |
| 画师 / 质量 | 经常带上 | 默认丢掉 |

![示例：双人巷内站立构图](docs/example.jpg)

> 示例图含 NSFW。请自行判断浏览环境。

上图反推后大致是这样（可直接贴进 NAI V5）：

```text
Prompt:
2girls, yuri, bondage, shibari, gag, cowboy shot, nsfw, night, full moon, outdoors, brick wall, alley

Two girls standing close together in a narrow brick alley at night, bodies pressed side by side.

Character 1:
girl, pink hair, long hair, pointy ears, elf, blue eyes, large breasts, shibari over clothes, red rope, corset, white shorts, white thighhighs, gag, bit gag, arms behind back, source#hug person

Character 2:
girl, pink hair, long hair, pointy ears, elf, blue eyes, white dress, white gloves, hat, beret, finger to mouth, shushing, smile, target#hug person
```

短 tag 负责「长什么样、穿什么」；那句英文负责「两个人在巷子里贴在一起」这种构图关系。

## 环境

- Python 3.11+
- 带视觉的 OpenAI 兼容接口（例如本机 Grok 网关）
- 可选：[WD14 EVA02](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) ONNX，用来补 Danbooru 短词

## 安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -r requirements.txt
```

## 配置

把 `.env.example` 里的变量写进环境，**不要把 Key 提交进仓库**：

| 变量 | 含义 |
|---|---|
| `NAI5_TAGGER_VLM_BASE` | Chat Completions 地址，默认 `http://127.0.0.1:8000/v1` |
| `NAI5_TAGGER_VLM_MODEL` | 视觉模型 id，默认 `grok-4.6` |
| `NAI5_TAGGER_VLM_KEY` | 网关需要鉴权时的 Bearer |
| `NAI5_TAGGER_WD14_DIR` | 含 `model.onnx` 与 `selected_tags.csv` 的目录；留空则跳过 WD14 |

## 使用

```bash
python -m nai5_tagger.server
```

打开 `http://127.0.0.1:18770/`，拖入图片，把主框和角色框分别复制进 NovelAI V5。

命令行：

```bash
python -m nai5_tagger 图片.png
python -m nai5_tagger 图片.png --json
python -m nai5_tagger 图片.png --no-nl
```

双人及以上默认带那句 NL；`--no-nl` 可关掉。

## 测试

```bash
python -m pytest tests/ -q
```

## 许可

MIT，见 [LICENSE](LICENSE)。
