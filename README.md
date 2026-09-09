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
   这是 NAI5 多角色官方施受写法，不是「Character 1 默认就是 source」。
   - `source#动词` = **施动者**（正在做这件事的人）
   - `target#动词` = **受动者**（被做的人）
   - 两边动词必须相同，例如 `source#hug person` + `target#hug person`
   - 口塞、站立、脸红这类**状态**写普通 tag（`bamboo gag`），不要单独冒出一个没有另一半的 `target#gagged`

5. **可粘贴进 NAI5**  
   网页分框复制。不反推画师串、质量词、油光材质。

## 和常见 Tagger 差在哪

| | WD14 / 多数反推 | 本工具 |
|---|---|---|
| 输出形态 | 一条扁平 tag | NAI5 **主框 + Character 1…N** |
| 两人衣服 | 混在一起 | 分到各自角色框 |
| 施受关系 | 没有，或只剩 `hug` | `source#动词` / `target#动词`，动词对齐、成对出现 |
| 空间关系 | 几乎没有 | 主框一句 NL 写站位、贴靠、复杂接触 |
| 画师 / 质量 | 经常带上 | 默认丢掉 |

![示例：双人巷内站立构图](docs/example.jpg)

> 示例图含 NSFW。请自行判断浏览环境。

上图反推后大致是这样（可直接贴进 NAI V5）：

```text
Prompt:
2girls, yuri, bondage, shibari, gag, cowboy shot, nsfw, night, full moon, outdoors, brick wall, alley

Two girls standing close together in a narrow brick alley at night. The girl in the hat stands slightly behind and to the right, one arm around the bound girl's shoulder.

Character 1:
girl, pink hair, long hair, pointy ears, elf, blue eyes, large breasts, shibari over clothes, red rope, corset, white shorts, white thighhighs, gag, bit gag, arms behind back, target#hug person

Character 2:
girl, pink hair, long hair, pointy ears, elf, blue eyes, white dress, white gloves, hat, beret, finger to mouth, shushing, smile, source#hug person
```

戴帽子、手臂搂着对方的是 **source**（施动）；双手反绑、被搂着的是 **target**（受动）。口塞只写 `gag` / `bit gag`，不会再单独写 `target#gagged`。

短 tag 负责「长什么样、穿什么」；那句英文负责「谁站在谁侧后方、手臂怎么搂」这种构图关系。

## 环境

- Python 3.11+
- 一个 **OpenAI 兼容**、**能看图** 的 Chat Completions 接口
- 可选：[WD14 EVA02](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) ONNX，用来补 Danbooru 短词

## 安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -r requirements.txt
```

## 配置视觉接口

本工具要调用 `POST {BASE}/chat/completions`，图片走 `image_url`。填三样即可：

| 你要填的 | 环境变量 | 怎么写 |
|---|---|---|
| 接口根地址 | `NAI5_TAGGER_VLM_BASE` | 写到 `/v1` 为止，**不要**再加 `/chat/completions` |
| 视觉模型名 | `NAI5_TAGGER_VLM_MODEL` | 该接口上**能看图**的模型名 |
| 密钥 | `NAI5_TAGGER_VLM_KEY` | 需要 Bearer 就填 API Key；接口免鉴权就留空 |

两种填法，选一种：

1. **网页里填（推荐）**  
   启动后打开页面顶部的「视觉接口」，填 BASE / MODEL / KEY，点「测试连接」。这三项存在你的浏览器里，不会写进仓库。

2. **`.env` 文件**  
   复制 `.env.example` 为 `.env`，改成你的值，**不要把密钥提交进仓库**。

填写示例（把密钥换成你自己的）：

| 服务 | BASE | MODEL |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| OpenRouter | `https://openrouter.ai/api/v1` | 该站的视觉模型名，例如 `openai/gpt-4o` |
| 其它兼容网关 | `https://你的主机/v1` | 网关列出的视觉模型名 |

可选 WD14：`NAI5_TAGGER_WD14_DIR` 指向含 `model.onnx` 和 `selected_tags.csv` 的目录；不配就跳过 WD14。

## 使用

```bash
python -m nai5_tagger.server
```

浏览器打开终端打印的地址（默认 `http://127.0.0.1:18770/`），先测通视觉接口，再拖入图片，把主框和角色框分别复制进 NovelAI V5。

命令行（同样读 `.env`）：

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

## 没有模型接口？

填上你的模型接口，拖入图片即可反推。什么？没有 agent 或者模型 API 接口怎么办？往下看~

常用 AI 产品低价代充： https://aizhanghao.com?from=8530  
ChatGPT、Gemini、Codex、Grok 会员全网最低价开通。用这个邀请链接注册，优惠多多。

## 许可

MIT，见 [LICENSE](LICENSE)。
