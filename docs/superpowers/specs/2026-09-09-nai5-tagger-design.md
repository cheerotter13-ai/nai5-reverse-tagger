# NAI5 多角色提示词反推工具设计

日期：2026-09-09  
状态：用户已批准；实施计划见 `docs/superpowers/plans/2026-09-09-nai5-tagger.md`  
工作区：`D:\nai5提示词反推工具`

## 1. 背景

现有 WD14 打标器面向 SDXL / Illustrious 一类扁平 tag 列表，没有 NovelAI Diffusion V5 的「主提示词框 + 角色框」结构。现有 NovelAI 工具包（`C:\Users\ROG\Downloads\NAI45_提示词工具包`）负责正向编译与出图填框，不负责从图片反推。自动化打标工具（`C:\Users\ROG\Documents\自动化ai作品打标签`）负责 WD14 + Stitch / Story Control，也不产出 NAI 角色框。

NAI5 官方多角色规则（[Multiple Characters](https://docs.novelai.net/image/multiplecharacters.html)）：

- 一张图 = 一个 base prompt + 最多 22 个 character prompts。
- 存在角色框时，`|` 语法禁用。
- 角色框内可用 `source#动作` / `target#动作` / `mutual#动作` 标明施受。
- 人数 tag 写在主框；角色框用 `girl` / `boy`，不带数字。

本工具的目标：对二次元 NSFW 图片，产出可直接粘贴进 NAI5 各框的文本。口径以用户既有写法为准（动作和施受前缀写在角色框），不以社区 skill「动作一律进主框」为准。

## 2. 已确认决策

1. 独立新仓库，路径 `D:\nai5提示词反推工具`。不修改 NovelAI 工具包，不修改自动化打标工具。
2. 识图后端：本机已通的 Grok 网关 `http://127.0.0.1:8000/v1`，模型 `grok-4.6`，OpenAI 兼容 `chat/completions` + 图像。CLIProxyAPI `127.0.0.1:8317` 仅作可选备用，默认不依赖。
3. Tag 补全：本机 WD14 EVA02 ONNX，`E:\SD\lora-scripts-v1.12.0\wd14_tagger_model\SmilingWolf_wd-eva02-large-tagger-v3`，阈值 0.35，优先 CUDA。
4. 编译器是纯函数：结构化 JSON + WD14 tag 列表 → NAI5 文本框。不识别、不反推、不输出画风 / 画师串 / 质量词 / 油光类全局材质。
5. 第一版交付本地网页 + CLI。自动填 NAI 网页留第二版。
6. 若输入图带完整 NAI 隐写/Comment 元数据，直接还原字段，不跑反推。
7. 凭据只从本机网关配置或环境变量读取，不写入仓库、笔记或结果 JSON。

## 3. 目标与非目标

### 3.1 第一版目标

- 一张 png/jpg/webp → 主框、角色框 1…N、UC。
- 支持 `source#` / `target#` / `mutual#`，按角色分框。
- 自然语言作为主框可选第二段，辅助空间关系和互动。
- 规范化输出：JSON + 分框复制文本，字段与 NAI 界面对应。
- 黄金样例锁住编译行为。

### 3.2 非目标（第一版不做）

- 画风、画师串、质量词、油光/亮皮等全局材质的识别或填写。
- 自动导入 / 自动填 NAI 网页。
- Custom Position / 5×5 网格。
- 负向词从画面反推（UC 用默认模板）。
- 批量目录、Stitch 入库、出图。
- 本地 JoyCaption / Qwen 等备用 VLM（Grok 失败即失败，不静默降级）。

## 4. 总体架构

```text
输入图
  │
  ├─ metadata_reader：NAI PNG/WebP 隐写或 Comment
  │     完整主框+角色框 → 直接还原，source=metadata
  │
  └─ 否则并行
        ├─ wd14_tagger（本机 ONNX）→ TagHit[]
        └─ grok_vision（127.0.0.1:8000）→ SceneDraft JSON
              │
              ▼
        compiler（纯函数）
              │
              ▼
        Nai5Prompt { base, characters[], uc, warnings, unassigned, ignored }
              │
              ├─ CLI 文本 / --json
              └─ 本地网页分框复制（127.0.0.1:18770）
```

编排器（`pipeline`）负责读图、短路元数据、并行调用、把结果交给编译器。编译器不访问网络和 GPU。

## 5. 组件与接口

每个单元只做一件事，通过下面的类型通信。实现时函数名与字段名必须与本节一致。

### 5.1 `metadata_reader`

- 输入：图像路径或字节。
- 输出：`MetadataResult | None`。
- 识别 NovelAI PNG text/iTXt、alpha 隐写 `stealth_pngcomp`、WebP EXIF UserComment。
- 仅当能解析出非空 `base_caption` 且 `char_captions` 为列表（可为空，单人图允许 0 个角色框）时视为命中。
- 命中后 `pipeline` 不再调用 WD14 / Grok。

### 5.2 `wd14_tagger`

- 输入：RGB 图像。
- 输出：`list[TagHit]`，`TagHit = {name: str, score: float, category: int}`。
- `category` 沿用 WD14 csv：`0` general，`1` artist，`3` copyright，`4` character，`5` meta。
- 阈值默认 0.35；低于阈值的 tag 丢弃。
- 加载失败或推理失败：返回空列表，由 pipeline 记 `wd14_skipped`，不中断。

### 5.3 `grok_vision`

- 输入：图像字节（jpeg/png，最长边缩到 1536 以内）、固定 system/user 提示（见第 8 节）。
- 输出：`SceneDraft`（第 6.1 节）。解析失败则整次 pipeline 失败。
- HTTP：`POST {base}/chat/completions`，`model=grok-4.6`，`response_format` 若网关支持 json object 则开启，否则从回复中抽取第一个 JSON 对象。
- 超时 120 秒。429 重试一次，间隔 5 秒。其他 4xx/5xx 不重试。

### 5.4 `compiler`

- 输入：`SceneDraft`、`list[TagHit]`、`CompileOptions`。
- 输出：`Nai5Prompt`。
- 无 I/O。规则见第 7 节。

### 5.5 `pipeline`

- 输入：图像路径、`CompileOptions`。
- 输出：`Nai5Prompt`（含 `meta.source`：`metadata` | `reverse`）。
- 负责短路、并行、错误包装。

### 5.6 `app` / `cli`

- 网页与 CLI 只渲染 `Nai5Prompt`，不包含编译规则。

## 6. 数据合同

### 6.1 `SceneDraft`（Grok 必须产出）

```json
{
  "count_tag": "2girls",
  "themes": ["yuri"],
  "scene": ["indoors", "wooden interior"],
  "camera": ["sunlight"],
  "nsfw": true,
  "nl": "The silver-haired girl sits on the other girl's back and holds a leash.",
  "characters": [
    {
      "gender": "girl",
      "identity": "",
      "appearance": ["long silver hair", "ponytail"],
      "clothing": ["nude"],
      "pose": ["sitting", "crossed legs"],
      "expression": ["blush"],
      "actions": [
        {"role": "source", "verb": "sitting on person"}
      ]
    }
  ]
}
```

约束：

- `gender` 只能是 `girl` | `boy` | `other`。
- `actions[].role` 只能是 `source` | `target` | `mutual`。
- `verb` 小写英文短语，不含 `source#` 前缀。
- `identity` 仅在有把握的版权角色时填写 ASCII / 罗马音（如 `bronya zaychik (honkai: star rail)`）。没把握则空字符串。
- 不得包含画师名、质量词、油光类材质词（见 7.3）。

### 6.2 `Nai5Prompt`（编译产物）

```json
{
  "base": { "tags": ["2girls", "yuri", "indoors", "nsfw"], "nl": "..." },
  "characters": [
    {
      "gender": "girl",
      "identity": "",
      "appearance": [],
      "clothing": [],
      "pose": [],
      "expression": [],
      "actions": [{"role": "source", "verb": "sitting on person"}]
    }
  ],
  "uc": { "preset": "default", "extra": [] },
  "unassigned": [],
  "ignored": [],
  "warnings": [],
  "meta": {
    "source": "reverse",
    "count_tag": "2girls",
    "wd14_skipped": false,
    "character_count": 2
  }
}
```

### 6.3 渲染文本

```text
Prompt:
<base.tags 逗号连接>
<若开启 NL 且 nl 非空：空一行后跟 nl，不加引号>

Character 1:
<gender>, <identity 若有>, <appearance>, <clothing>, <expression>, <pose>, <source#verb / target#verb / mutual#verb>

Character 2:
…

UC:
<默认预设文本>
```

字段名用半角冒号。不得出现 `char1`、`char2`、`Character 1:` 作为 tag 内容（框标题除外）。

### 6.4 `CompileOptions`

- `include_nl: bool` — 默认：角色数 ≥ 2 为 true，否则 false。
- `uc_preset: str` — 默认 `"default"`。
- 没有画风、质量、材质相关选项。

## 7. 编译规则

### 7.1 分框

| 进主框 | 进角色框 | 丢弃（`ignored`） | `unassigned` |
|---|---|---|---|
| 人数、玩法/关系（yuri / hetero / ntr / cuckolding…）、场景、镜头、`nsfw` | 性别（无数字）、身份、外观、服装、该人动作与表情、`source#`/`target#`/`mutual#` | 画师、质量、油光/亮皮类材质（7.3） | WD14 有、VLM 未归属、且不属于丢弃列表的内容 tag |

`pov` 双重身份：主框写 `first person view`（若 VLM/WD14 给出），对应角色框再写 `pov`。

WD14 的 `explicit`、`rating:explicit`、`nsfw` 一律归一成主框里的一个 `nsfw`，不保留 `rating:` 前缀。

### 7.2 硬约束

1. 人数 tag 只在主框出现一次。角色框禁止 `1girl` / `2boys` 等带数字人数词。
2. 禁止输出 `char1:` / `char2:` / `character 1` 这类假标签。
3. 同一 `verb` 若出现 `source#`，应对应至少一个 `target#`（或全体 `mutual#`）。不成对则仍输出，并加 warning `unpaired_action:<verb>`。
4. 不同 verb 可以分属不同角色，不必强行配成同一动词。
5. 同一角色对同一 verb 不得同时 `source` 与 `target`；若草稿如此，改写为一条 `mutual`，并 warning `collapsed_to_mutual:<verb>`。
6. 同一角色可以对不同 verb 同时挂 source 和 target。
7. 前缀仅 `source#` / `target#` / `mutual#`，小写，`#` 后为 verb 原文。
8. 某外观/服装 tag 一旦进入某角色框，从主框删除，禁止双写。
9. 版权名：仅当 `SceneDraft.identity` 非空，且（WD14 category 4 命中同一角色或 VLM 明确给出）时写入角色框。冲突则不写名字，warning `identity_unconfirmed`。
10. 默认不加 `{tag}`、`n::tag::`。草稿里若自带权重语法，剥掉后当普通 tag。
11. 自然语言只进主框第二段，不加引号。
12. 角色框数量以 `SceneDraft.characters` 为准。与 `count_tag` 推导出的人数不一致时，仍用角色框数量渲染，并 warning `count_mismatch`，同时把 `count_tag` 改成与角色框一致的值（例如 1 girl + 2 boy → `1girl, 2boys`）。
13. WD14 无法归属的非丢弃 tag 进 `unassigned`，不塞进任意角色。网页可拖入某框后只重跑 compiler，不重跑模型。

### 7.3 明确不关心（丢弃）

编译器和 VLM 提示都必须排除下列内容，不得进入主框或角色框：

- WD14 `category == 1`（artist）。
- 质量/完成度：`masterpiece`、`best quality`、`amazing quality`、`newest`、`absurdres`、`highres`、`very aesthetic`、`aesthetic`、`year 2023`、`year 2024`、`year 2025`。
- 油光/亮皮类：`shiny skin`、`oiled skin`、`oiled`、`shiny clothes`、`glistening skin`。

匹配时忽略下划线与空格差异（`shiny_skin` 等同 `shiny skin`）。未列但明显是画师名的 category 1 仍丢弃。不要把「湿身因为落水」这类场景状态扩进这个列表；只丢弃上列材质/质量/画师。

### 7.4 WD14 归属

顺序：

1. 先按 7.3 丢进 `ignored`。
2. 人数、玩法、场景、镜头、`nsfw`/`explicit` → 主框候选。
3. category 4 → 尝试匹配某角色 `identity`（大小写不敏感、下划线/空格归一）。匹配失败则 `unassigned`，不自动当身份。
4. 外观/服装/表情/姿势：若该 tag 出现在某角色草稿的对应列表中，或 VLM `nl`/appearance 字符串包含该 tag，则归该角色。
5. 其余 general tag → `unassigned`。

不得仅因「这是服装 tag」就塞给角色 1。

### 7.5 UC

`preset=default` 使用仓库内固定文本（实现时放 `nai5_tagger/data/uc_default.txt`）。内容采用 NAI 网页常见英文负面预设的一份静态副本，不从画面生成。`extra` 第一版恒为空。

## 8. Grok 视觉提示合同

System 提示固定，禁止运行时拼接用户自由文本（避免提示注入）。要求模型：

- 只输出一个 JSON，符合 `SceneDraft`。
- 二次元 NSFW 按画面如实描述，不回避、不改写为 SFW。
- 按人拆外观、服装、动作；互动用 `actions` 的 source/target/mutual。
- 主视角角色标 `pov` 放在该角色，并在 `camera` 放 `first person view`。
- 不要输出画师、质量词、油光类材质。
- 不要输出 `char1` 标签。
- 版权角色没把握则 `identity` 留空，改写外观。

User 提示只含「分析这张图并填 JSON」及 schema 摘要。图像以 `image_url` data URL 发送。

## 9. 网关与凭据

| 项 | 值 |
|---|---|
| 默认 `NAI5_TAGGER_VLM_BASE` | `http://127.0.0.1:8000/v1` |
| 默认 `NAI5_TAGGER_VLM_MODEL` | `grok-4.6` |
| API Key | 只读环境变量 `NAI5_TAGGER_VLM_KEY`。网关免密则留空。不把密钥写入仓库、日志、结果 JSON |
| 可选备用 | `NAI5_TAGGER_VLM_BASE=http://127.0.0.1:8317/v1` |

启动时对 `GET {base}/models` 做一次可达检查；失败则网页显示「Grok 网关不可用」，禁止空跑成功。

WD14 模型目录可用 `NAI5_TAGGER_WD14_DIR` 覆盖，默认见第 2 节。

## 10. 界面与 CLI

### 10.1 网页

- 启动：`启动反推.bat` → 若 `127.0.0.1:18770` 已在听则只打开浏览器，否则起进程再打开。控制台窗口保持打开。
- 拖入或选择一张图。
- 状态行：`metadata` / `wd14` / `grok` / `compile` / 错误原文。
- 结果：主框、每个角色框、UC，各带复制按钮，外加复制全部。
- `warnings` 列表。`unassigned` 可拖到某框后本地重编译。
- `ignored` 默认折叠，不提供「加回主框」的画风开关。
- NL 开关：默认按 `CompileOptions.include_nl`。
- 不提供画师串、质量、油光相关控件。

### 10.2 CLI

```text
python -m nai5_tagger path\to\image.png
python -m nai5_tagger path\to\image.png --json
python -m nai5_tagger path\to\image.png --no-nl
```

默认 stdout 为第 6.3 节文本，编码 UTF-8。`--json` 打印 `Nai5Prompt`。退出码：成功 0，Grok 失败 2，读图失败 1。WD14 跳过仍为 0，JSON 里 `wd14_skipped=true`。

## 11. 错误处理

| 情况 | 行为 |
|---|---|
| Grok 不可达、超时、非 JSON、拒识图 | pipeline 失败，返回错误原文；不改走本地 VLM |
| HTTP 429 | 重试一次，再失败则失败 |
| WD14 缺失或 CUDA/CPU 均失败 | 继续，`wd14_skipped=true` |
| 完整 NAI 元数据 | `source=metadata`，跳过反推 |
| 不成对 action | 输出 + warning |
| 人数与角色数不一致 | 校正 count_tag + warning |
| 网关返回 401 | 失败并提示检查 `NAI5_TAGGER_VLM_KEY`，不把密钥写入日志 |

## 12. 测试与黄金样例

CI / 本地单测不调用真网关、不加载真 ONNX。`grok_vision` 与 `wd14_tagger` 用假适配器。

### 12.1 编译器单测必须覆盖

- **样例 A（三人）**：输入与用户案例等价的 `SceneDraft`。主框含 `1girl`、`2boys`、`cuckolding theme`、`first person view`、`nsfw`；三个角色框分别为女、施动男、pov 男；对应 `target#` / `source#` 落在正确框；全文不含 `char1`；不含 `shiny skin` / `oiled` / 画师 / 质量词。
- **样例 B（双人骑跨）**：主框 `2girls, yuri` + 室内；角色 1 `source#sitting on person`；角色 2 `target#sitting on person`；发色与姿势不串框。
- **样例 C**：WD14 有 `silver hair`，草稿只把该外观写在角色 2 → 仅角色 2 含 `silver hair`，主框与角色 1 不含。
- 人数不进角色框。
- 7.3 丢弃列表中的 tag 无论来自草稿还是 WD14 都不进主框/角色框。
- 同角色同 verb 的 source+target 折叠为 mutual。
- 拖动 `unassigned` 后再次 `compile` 结果稳定。

### 12.2 编排单测

- 元数据命中则 mock 的 Grok/WD14 调用次数为 0。
- Grok 抛错则 pipeline 失败。
- WD14 抛错则仍返回 `Nai5Prompt` 且 `wd14_skipped=true`。

### 12.3 现场验收（实现后，不在 CI）

用设计讨论中的双人截图走网页：能分出两个角色框、施受前缀正确、复制按钮可用。Grok 网关与 WD14 以当时现场为准。

## 13. 建议目录

```text
D:\nai5提示词反推工具\
  docs\superpowers\specs\2026-09-09-nai5-tagger-design.md
  nai5_tagger\
    __init__.py
    __main__.py          # CLI
    types.py             # SceneDraft, Nai5Prompt, TagHit
    metadata_reader.py
    wd14_tagger.py
    grok_vision.py
    compiler.py
    pipeline.py
    render.py            # Nai5Prompt → 文本
    server.py            # 18770
    data\uc_default.txt
    data\ignore_tags.txt
  tests\
    test_compiler.py
    test_pipeline.py
    test_render.py
    fixtures\
  启动反推.bat
  TASK_CONTEXT.md
```

不把自动化打标仓库的 Python 包加进本项目依赖。WD14 推理在本仓库内用 onnxruntime 直接加载同一权重目录。

## 14. 第二版（本文件不实施）

- 浏览器填入 NAI 角色槽（可复用工具包已有 CDP 填框经验）。
- 批量目录。
- 用户把 `unassigned` 的归属规则存成偏好。
