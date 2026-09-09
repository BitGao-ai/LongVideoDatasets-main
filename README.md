# LongVideo-Dataset · 长视频理解测评集 · LLM 自动标注 v2

用 **Qwen(DashScope)/ Kimi(Moonshot)** 多模态大模型 API,对 **30min–2h 长视频**做自动数据标注,
输出严格符合 `schema/annotation.schema.json` 的中英双语标注结果。

> ⚠️ 机器标注结果 `annotation_meta.review_status = "draft"`,是**人工复核前的初标**。
> 按数据集方案,LLM 只做辅助扩量,**必须经人工验证 + 抗捷径过滤 + 发布门禁**后才能进入正式基准。

---

## 目录结构

```
LongVideo-datasets/
├── requirements.txt
├── preprocess.py                 # ① 预处理:镜头分割 + WhisperX + PaddleOCR + 帧索引 + 音频事件轨 + manifest
├── run_annotate.py               # ② 标注 CLI 入口(S1~S7 全阶段,详述默认必产)
├── run_describe.py               # ③ 给已标注文件补全片详述 CLI
├── run_rescreen.py               # ④ 换模型复筛 CLI
├── run_qc.py                     # ⑤ 黄金题 + 一致性质检 + 发布门禁 CLI(v2)
├── schema/annotation.schema.json # 输出格式(双语 schema v2)
├── REVIEW.txt / Res.txt          # 审查报告 与 v2 解决方案详设(实现依据)
├── examples/
│   ├── meta_template.json        # 每个视频需人工提供的元信息模板
│   └── responses_template.json   # 验证者独立作答记录模板 v2(mcq/temporal/open 三种作答)
└── annotator/
    ├── config.py                 # 供应商 + 全部运行参数(RunConfig)
    ├── llm_client.py             # 统一多模态客户端(OpenAI 兼容 + 重试 + JSON 抽取 + trace 审计)
    ├── local_client.py           # 本地小模型(vLLM):1fps 帧级短描述 / 音频事件类型判别(两级策略)
    ├── frames_store.py           # 帧索引(ffmpeg 1fps,文件名即时间戳):抽帧/单帧/8×8 马赛克/verify_index
    ├── manifest.py               # 阶段状态机 + 输入哈希幂等 + 窗口级 StageCache + gaps/human_todo
    ├── orchestrator.py           # 端到端编排(S1~S7 状态机 + 断点续跑 + 落盘 + schema 校验)
    ├── pipeline.py               # 兼容入口(委托 orchestrator)
    ├── media.py                  # 抽帧/编码、字幕&OCR 切片、镜头合并成窗口
    ├── prompts.py                # 双语提示词(场景/详述/事件/出题/核验/过滤)
    ├── annotate.py               # S1:逐窗口场景+事件标注(失败重试 + gaps,不再静默丢窗)
    ├── events.py                 # S3 事件引擎:挖掘 → 8×8 边界复核 → 去重 → 关键性分级 → span 校验
    ├── aggregate.py              # S4 全局聚合:梗概/人物表/关系/时间线/因果核验/主题/转折/伏笔
    ├── dense_caption.py          # S2 密集详述引擎:shot→segment→全片 三级金字塔 + 质量门自愈
    ├── describe.py               # 兼容入口(委托 dense_caption)
    ├── native_video.py           # native_video 模式:ffmpeg 切片 + DashScope 原生视频(仅 Qwen)
    ├── audio_events.py           # 音频事件轨:ASR 空白 + 能量突增 → {start,end,type,desc}
    ├── qa_engine.py              # S5 出题引擎:证据锚定 + 题干指代校验 + 能力矩阵硬约束 + 按时长配额
    ├── qa_generate.py            # 兼容入口(委托 qa_engine)
    ├── qa_verify.py              # S6 双模型(VIDEO)答案核验 + 梗概泄漏检测
    ├── anti_shortcut.py          # S7 五通道抗捷径(盲答/单帧/字幕/梗概/语言先验)+ 非 MCQ 通道
    ├── filter.py                 # 兼容入口(委托 anti_shortcut)
    ├── difficulty_tagger.py      # 难度自动标定(easy/medium/hard,后过滤重算)
    ├── rescreen.py               # 换模型复筛(另一家供应商重跑抗捷径,判定取或)
    ├── qc_v2.py                  # 质检 v2:MCQ α/黄金题/打回 + temporal mIoU + open 一致率 + 发布门禁
    └── qc.py                     # Krippendorff α / Cohen κ / 黄金题(纯标准库)
tests/                        # 纯逻辑回归测试(pytest,无需视频/API key)
```

## 流水线(v2)

```
视频 + 人工元信息(meta)
  → preprocess.py          : shots / subtitles / ocr / 音频事件轨 / 帧索引(frames/) / manifest
  → S1 structure           : 逐 ~180s 窗口送【采样帧+字幕+OCR】→ scenes / events(失败重试→gaps,不静默)
  → S2 describe            : shot.desc → segment 详述 → 全片两级汇总(质量门:字数/要素/时间锚点,自愈重试)
  → S3 events              : 事件挖掘(滑窗帧描述轮廓)→ 8×8 马赛克边界复核 → 去重 → key/minor 分级
  → S4 aggregate           : 梗概 / 人物表(确定性统计)/ 关系 / 时间线 / 因果边核验 / 主题 / 转折 / 伏笔
  → S5 qa                  : 锚定出题(MCQ+temporal+open+summary,题干指代上下文,能力矩阵硬约束)
  → S6 verify              : 另一家供应商看真实证据帧核验答案(不一致打回重写)+ 梗概泄漏剔除
  → S7 filter              : 五通道抗捷径(盲答/单帧/字幕/梗概/语言先验),命中默认剔除
  → 难度标定 → 组装为 schema v2 JSON,校验后落盘(review_status=draft)
```

- **长视频如何塞进 API**:不整段上传;按镜头合并成 ~180s 窗口,每窗采样 10 帧(长边≤768px)
  送多模态模型;全局聚合与出题只用文本,把 token/成本压到可控。
- **帧索引(1fps)预构建一次**:标注/详述/边界复核/过滤全部从帧索引取帧,不再二次 seek 视频流
  (替换旧版 `CAP_PROP_POS_MSEC`),文件名即时间戳,`verify_index` 校验单调/首帧≈0/末帧≈时长。
- **断点续跑**:manifest 记录每阶段输入内容哈希,哈希不变即复用;窗口级结果缓存到磁盘,
  中断重跑不重复烧钱。
- **两级模型策略**:本地小模型(`--local-base-url/--local-model`,如 Qwen2.5-VL-8B)做
  1fps 帧级短描述与音频事件分类等重活;云端大模型做镜头/片段详述、边界复核、出题等精活。

## 安装

```bash
pip install -r requirements.txt
# 预处理依赖较重,建议单独虚拟环境:
# pip install "scenedetect[opencv]" whisperx paddleocr paddlepaddle-gpu
```

## 配置 API Key

```bash
# Qwen(阿里云百炼 DashScope)
export DASHSCOPE_API_KEY=sk-xxx
# 或 Kimi(Moonshot)
export MOONSHOT_API_KEY=sk-xxx
# 答案核验/复筛用另一家时,两家 key 都要有
```

> 模型名在 `annotator/config.py` 里改。请按你开通的版本填,例如
> `qwen-vl-max` / `qwen2.5-vl-72b-instruct`(视觉)、`qwen-plus`(文本);
> Kimi 用 `moonshot-v1-32k-vision-preview`(视觉)、`moonshot-v1-32k`(文本)。

## 运行

所有入口都支持两种方式:**批量**(处理文件夹下所有视频/标注文件,推荐)与**单条**(单个视频/单个文件)。
各入口参数完全相同,`--in` / `--videos-dir` / `--annotations` 传目录即批量,传具体文件即单条。

### 批量运行(推荐,一次处理文件夹下所有视频)

```bash
# 0) 说话人分离 token(可选;不设则跳过说话人分离)
export HF_TOKEN=xxx

# ---- ① 预处理:videos/ 下所有视频 ----
# 产物:structure/<id>.json(结构)+ structure/frames/<id>/(帧索引)+ structure/<id>.manifest.json
# 已存在自动跳过(--force 重跑);输出目录用 --structure-dir 改(默认 structure/)
python preprocess.py --videos-dir videos --lang zh

# ---- ② LLM 标注:structure/ 下所有结果 ----
# 产物:annotations/<id>.json(标注结果);meta 缺失时自动生成到 meta/<id>.json
# 输出目录用 --out-dir 改(默认 annotations/);meta 目录用 --meta-dir 改(默认 meta/)
python run_annotate.py --in structure/ --provider qwen \
    --meta-dir meta --out-dir annotations --videos-dir videos

# ---- ③ (可选)补全片详述:annotations/ 下所有文件 ----
python run_describe.py --in annotations/ --out-dir annotations_desc/ --provider qwen

# ---- ④ (可选)换模型复筛:annotations/ 下所有文件 ----
export MOONSHOT_API_KEY=sk-xxx
python run_rescreen.py --in annotations/ --out-dir annotations_rescreened/ \
    --provider kimi --report reports/rescreen.json

# ---- ⑤ (可选)质检:整个 annotations/ 批次 ----
python run_qc.py --annotations annotations/ --responses reviews/responses.json \
    --report reports/qc.json --release-report reports/release_report.json
```

产物目录速览(视频 ID = 文件名去扩展名):

```
structure/          -- 预处理产物(--structure-dir 可改)
├── <id>.json             结构 JSON(shots/subtitles/ocr/audio_events)
├── <id>.manifest.json    断点续跑 manifest
└── frames/<id>/          1fps 帧索引(文件名即时间戳)
meta/<id>.json      -- 元信息(缺失自动生成,请人工补 genre/title/license)
annotations/        -- 标注产物(--out-dir 可改)
├── <id>.json             标注结果(schema v2)
└── .cache/<id>/          窗口级缓存(断点续跑)
reports/            -- 复筛/质检报告 与 llm_trace_<id>.jsonl(LLM 调用审计)
```

### 单条运行(单个视频 / 单个标注文件)

```bash
# 0) 说话人分离 token(可选)
export HF_TOKEN=xxx

# ---- ① 预处理:单个视频 ----
# 产物:--out 指定的 JSON(structure/doc_00001.json)+ 同目录 frames/doc_00001/(帧索引)
# + doc_00001.manifest.json;--out 缺省写入 structure/<id>.json
python preprocess.py --video videos/doc_00001.mp4 --video-id doc_00001 \
    --lang en --out structure/doc_00001.json

# ---- ② 准备 meta(拷贝模板改字段:license/source/release_date 等须人工填) ----
cp examples/meta_template.json meta/doc_00001.json

# ---- ③ LLM 标注:单个视频 ----
# (v2:全片详述默认必产;--qa 留空=按时长自动出题)
python run_annotate.py \
    --meta      meta/doc_00001.json \
    --structure structure/doc_00001.json \
    --out       annotations/doc_00001.json \
    --provider  qwen

# ---- ④ (可选)补全片详述:单个标注文件 ----
python run_describe.py --in annotations/doc_00001.json \
    --out-dir annotations_desc/ --provider qwen

# ---- ⑤ (可选)换模型复筛:单个标注文件 ----
export MOONSHOT_API_KEY=sk-xxx
python run_rescreen.py --in annotations/doc_00001.json \
    --out-dir annotations_rescreened/ --provider kimi

# ---- ⑥ (可选)质检:单个标注文件 ----
python run_qc.py --annotations annotations/doc_00001.json \
    --responses reviews/responses.json --report reports/qc.json
```

单条产物位置:标注结果由 `--out` 指定;预处理时 `--out` 指定结构 JSON,
`frames/` 与 manifest 自动放 `--out` 同目录;`--out` 缺省写入 `structure/<id>.json`。

> **双模型发布前**:标注时配 `--verify-provider kimi`(答案核验用另一家)。
> **批次特性**:单文件失败不阻断其余文件,失败清单在结束时报出并以非零码退出;
> 重跑时标注/详述/复筛均有 manifest 或产物复用机制,不重复烧钱。
> **Shell 提示(zsh)**:环境变量请用 `export` 设置,不要把未设置的 `$VAR` 直接放在命令行参数里。
> zsh 会把未设置的 `$VAR` 展开为"零个参数"(bash 则为空串),导致 `--xxx $VAR` 变成裸 `--xxx`
> 并报 `argument --xxx: expected one argument`。`--hf-token` 已兼容此场景(留空读环境变量)。

常用参数:

| 参数 | 说明 |
|---|---|
| `--window-sec` / `--frames` | 窗口时长 / 每窗采样帧数(默认 180s / 10) |
| `--qa` | 出题数;0 = 按时长自动(`ceil(hours×24)`,下限 10) |
| `--no-describe` | 关闭全片详述(默认必产) |
| `--describe-mode native_video` | 原生视频模式(仅 Qwen,需 ffmpeg+dashscope) |
| `--frame-rate` | 帧索引抽帧率(默认 1fps) |
| `--local-base-url` / `--local-model` | 启用本地小模型做帧描述/音频判别 |
| `--max-windows N` | 调试:只跑前 N 个窗口 |
| `--keep-shortcut` | 命中捷径只标记不剔除 |
| `--keep-single-frame` | opt-in:保留单帧可答的视觉题(默认剔除) |
| `--no-verify` | 关闭双模型答案核验 |
| `--workers` | 单视频内的阶段并发线程数(默认 4):镜头描述 / 片段详述 / 事件挖掘 / 边界复核 / 答案核验 / 抗捷径过滤 |
| `--video-workers` | 批量模式下同时标注几个视频(默认 1 串行);总在途请求 ≈ `video-workers × workers`,按供应商 QPS 上限调 |

## 事件引擎(关键事件定位)

S3 保证"关键事件 100% 定位"这一硬性需求:

1. **挖掘**:帧级描述流(本地模型 1fps 全密度 / 无本地模型时云端降频 1/20s)按滑窗
   (60s、50% 步进叠加)用相邻帧主题一致性合并出候选事件,再用大模型汇总修正边界与双语描述;
   无帧描述时退化为字幕句组候选。
2. **边界复核**:每个候选 span ±4s 内取【真实存在的不重复帧】排成方阵马赛克送视觉模型,
   四选一判定 `verified / refined / split / rejected`;Refined/Split 二次复核 ≤2 轮。
   帧数不足以铺满 8×8 时自动缩小方阵(如 11 帧 → 3×3),不用重复帧凑数;提示词附带
   **逐格时间对照表**,模型给出的新 span 会被夹到马赛克实际覆盖范围内。定位可信度核心。
   > 边界精度上限 = 帧索引抽帧率的倒数。1fps 下最细只能定位到 ±1s;需要更细就提高
   > `--frame-rate`(磁盘与预处理成本同比上升)。
3. **去重**:时间重叠 >50% 且 embedding 相似度 ≥0.85 合并;≥3 事件的组交大模型三要素裁决。
4. **关键性分级**:`key / minor` + 一句依据;key 事件强制 span∈[0,duration] 且宽度 ≥3s,违例降级。
5. **span 合法性**:越界 clamp、倒置修复、重叠 >80% 报警留痕;非法事件与被判 `rejected`
   的误报**不进数据集**,一并写入 `annotation_meta.human_todo` 供人工确认。

统计(refined_ratio / rejected_ratio / event_density / key 数)写入 manifest 与 annotation_meta。

## 全片详细描述(detailed description)

v2 起**默认随标注必产**(`--no-describe` 关闭),三级描述金字塔:

- **镜头级 `shot.desc`**:首/中/尾 3 关键帧 + 字幕 + OCR → 中英密集描述,要素 ≥6/8、
  中文 ≥80 字/英文 ≥60 词,不达标带反馈重试;
- **片段级 `structure.segments[]`**:以镜头级描述(带起止标注)汇总 ~180s 段,中文 ≥400 字/
  英文 ≥350 词、要素 ≥6/8、≥2 处"约 MM:SS"锚点;仍不过进 `quiet_segments` 人工清单(不静默丢弃);
- **全片级 `global.detailed_description`**:zh+en 双轨两级汇总,中文 ≥1000 字门;缺侧语言由
  纯文本模型翻译补齐并做长度比/回译粗查(`--strict-lang-check` 更严)。

**两种模式,用 `--describe-mode` 切换:**

| 模式 | 送模型方式 | 适用 | 依赖 |
|---|---|---|---|
| `frames`(默认) | 每窗抽帧 + 字幕 + OCR | Qwen / Kimi 通用,成本低 | opencv |
| `native_video` | ffmpeg 切片,把**真实视频片段**直接喂 Qwen-VL | **仅 Qwen**,更"直接"、成本高 | ffmpeg + `pip install dashscope` |

> 30min–2h 无论哪种模式都**无法一次喂整段**(Qwen-VL 原生视频也有时长/帧数上限),
> 故都是按窗口分块;`native_video` 是把每个窗口切成短视频片段直接送模型,而非抽帧。
> Kimi 无原生视频输入,选 `native_video` + `--provider kimi` 会直接报错并提示改回 `frames`。

```bash
# 给已标注文件补详述(同样支持切换模式)
python run_describe.py --in annotations/ --out-dir annotations_desc/ --provider qwen \
    --describe-mode native_video --native-fps 2
```

调参:`--native-fps` 控制 DashScope 对片段的抽帧率(越高越细、越贵);切片临时文件默认
用完即删,设 `RunConfig.native_clip_dir` 可保留以便排查。

## QA 引擎(锚定出题 + 能力矩阵)

- **证据锚定**:锚点池 = 关键事件(优先)∪ 普通事件 ∪ 显著镜头 ∪ 音频事件;每题强制绑定
  `evidence_spans`,题干必须含**指代上下文**(时间描述 / 序数消歧词 / 命名锁定短语),
  不满足自动重写 ≤2 次,仍不过剔除;
- **能力矩阵硬约束**(代码门禁,非 prompt 口头要求):L1~L5 每层 ≥2 题、cross_scene ≥50%、
  task_type 分布(mcq 50–70% / temporal 15–25%)、时长 4 桶每桶 ≥15%;不达标时出题提示词
  注入缺口,最终 `qa_coverage` 报告未达标即发布拦截;
- **数量按时长**:`qa_quota = ceil(duration_hour × 24)`(下限 10),超长视频分批出题;
- **防梗概泄漏**:出题输入不含 synopsis/全片详述;另设"题干+梗概可答 → 剔除"检测通道;
- **非 MCQ 规范**:open/summary 参考答案中文 ≥80 字/英文 ≥220 词且含 ≥2 个具体细节;
  temporal_target 宽度 ≥5s、落在锚点 span 内。

## 换模型复筛 + 答案核验(cross-model)

初标与其抗捷径过滤用的是同一家模型,存在**同源偏置**。发布前用【另一家】模型独立核验:

- `run_annotate.py --verify-provider kimi`:标注时即让另一家模型看**真实证据帧**独立作答,
  与生成答案不一致 → 打回重写(≤2 次),结果写 `q.verification {provider, agreement, pass}`;
  梗概泄漏命中直接剔除并记 `quality_flag`;`verify_agreement` 作为 KPI 进 manifest。
- `run_rescreen.py`:对每题按 task_type 重跑对应捷径通道(盲答/单帧/字幕/梗概/语言先验、
  temporal 字幕定位、open 盲答相似度),两家判定**取或**,追加 `cross_check` 留痕。

```bash
# 初标用 qwen -> 复筛用 kimi(供应商须不同,否则跳过;--force 可强制)
export MOONSHOT_API_KEY=sk-xxx
python run_rescreen.py --in annotations/ --out-dir annotations_rescreened/ \
    --provider kimi --report reports/rescreen.json
```

复筛结果仍是 `draft`,须人工终审。

## 抗捷径过滤(五通道 + 非 MCQ)

任一通道命中即剔除(默认;`--keep-shortcut` 只标记不剔除):

| 通道 | 判定 | 处置 |
|---|---|---|
| 盲答 | 题干+选项(纯文本)答对 | 一律剔除 |
| 单帧 | 证据区 1 帧答对(视觉题) | 剔除(`--keep-single-frame` 可保留) |
| 字幕 | 字幕文本答对(视觉题;音频/字幕原生题豁免) | 剔除 |
| 梗概 | 题干+梗概答对(**全部 MCQ**,不豁免音频题) | 剔除 |
| 语言先验 | 选项模板("以上都不对"等)/ 选项原样出现在题干(**全部 MCQ**) | 剔除 |
| temporal 字幕定位 | 只给字幕定位区间 IoU ≥0.6 | 打回 |
| open 盲答相似度 | 盲答文本与参考答案相似度 ≥0.85 | 剔除 |

判定结果写 `q.anti_shortcut` 留痕;过滤后按保留顺序重排 `qid`。各题的通道判定
彼此独立、按 `--workers` 并发;单题异常或取帧失败只让该通道失效,不影响同题其它通道。
梗概通道在 S6 已执行过时 S7 自动跳过,避免同口径重复调用。

## 黄金题 + 一致性质检 + 发布门禁(QC v2)

| 指标 | 公式 | 门槛 |
|---|---|---|
| MCQ 一致性 | Krippendorff α(≥2 验证者) | α ≥ 0.8 |
| 黄金题 | 标注员在黄金题上正确率 | ≥ 0.9,低者标记复训 |
| temporal mIoU | 验证者区间 vs 参考答案 IoU | 合格率 ≥ 90%(单题 IoU ≥ 0.5) |
| open/summary | ROUGE-L / embedding 相似度一致 | 一致率 ≥ 0.6 |
| 覆盖率 | annotation_meta.gaps 为空 | 0 gaps |
| 事件密度 | events / 时长 | ≥ 45/h;key 事件 span 100% 有效 |
| 描述门 | shot.desc 全非空;quiet_segments 为空 | 100% |
| 双语 | question 中英齐备 | ≥ 95% |

```bash
# 标准答案与黄金题(is_gold)自动从 --annotations 读取;
# 验证者独立作答见 examples/responses_template.json(三种题型作答字段)
python run_qc.py --annotations annotations/ --responses reviews/responses.json \
    --alpha-threshold 0.8 --gold-threshold 0.9 --report reports/qc.json \
    --release-report reports/release_report.json
```

输出:α 及是否达标、每位标注员黄金题正确率与复训标记、需打回的题清单、
**视频级发布门禁**(全部指标达标 → `review_status=verified`,否则 draft + 拒绝原因 +
人工 TODO 清单)。门禁判定会**回写标注文件**(`--no-write-back` 可只出报告)。
α / mIoU / open 一致率按**每个视频自己的作答样本**判定,不会因为同批另一个视频的
数据不足而整批被拒;缺样本的指标记为「未评估」而非「不达标」。
`annotator/qc_v2.py` 也可作为库调用;`--embedding` 可用供应商 embedding
替代 ROUGE-L 计算 open 一致率(ROUGE-L 已按 CJK 逐字分词,中文可用)。

## 输出

严格符合 `schema/annotation.schema.json` v2,含 `meta / media / structure(shots,scenes,events,
segments,subtitles,ocr,audio_events,subtitles_meta,frames_index_url) / global(synopsis,
detailed_description,characters,relations,timeline,themes,turning_points,motifs) / qa[] /
annotation_meta(gaps,stats,human_todo,stages,frames_index)`。每道 QA 带 `capability /
evidence_spans / referring_ctx / min_watch / verification / anti_shortcut / quality_flags /
difficulty`;每个事件带 `importance / importance_reason / boundary_state / dedup_id / causes /
scene_id / scene_alignment`。`segments` 与 `detailed_description` 默认必产(`--no-describe` 关闭)。

## 断点续跑(manifest + 缓存)

- 每阶段记录**输入内容哈希**(非路径),哈希不变 → 直接复用该阶段产物(天然缓存,复跑零成本);
- 窗口级 LLM 结果缓存到 `<out>/.cache/<video_id>/`,中断后续跑不重复烧钱;
- 窗口失败指数退避重试 ≤3 次后写入 `annotation_meta.gaps[](start,end,reason)`,
  发布门禁要求 **0 gaps**;
- 每次 LLM 调用写 `reports/llm_trace_<video_id>.jsonl`(时间戳/模型/输入哈希/重试/输出摘要),
  可复现、可审计。

## 成本与限制

- 主要成本:场景标注(每窗一次多模态调用)+ 详述(默认必产)+ 事件边界复核(每事件 1 次
  马赛克调用)+ 出题 + 每题捷径通道调用。**1 小时视频实测约 800~1500 次调用**(事件候选
  数与镜头数决定上限),可用 `--frames` / `--window-sec` / `--frame-rate` / `--qa` /
  `--no-describe` / `--workers` 调节,或配置本地小模型(`--local-base-url/--local-model`)
  把帧级描述降为免费。
- 磁盘:帧索引 1fps × 2h ≈ **0.4~0.9 GB/视频**(768px JPEG),批量处理前请预留空间;
  `--frame-rate` 提高会同比放大。
- 显存:预处理在同一进程里串行跑 WhisperX + 说话人分离 + PaddleOCR,每个重模型用完会
  显式 `empty_cache()` 释放。16GB 卡建议 `--whisper-batch 8`;无 GPU 用
  `--device cpu`(会自动切到 `int8`)。
- `scenes/events` 由 LLM 生成,时间戳经马赛克边界复核修正,仍建议人工抽测;
  **因果边、人物合并需人工校验**(违例边自动丢弃并进人工清单;场景临时称谓未能并入
  人物表的也会进清单)。
- 抗捷径过滤用的是同一家模型,建议正式发布前**换一家模型复筛**,避免同源偏置。
- 结果是 `draft`,进入基准前务必经人工复核(一致性 α≥0.8、黄金题监控、发布门禁全绿)。

## 测试

```bash
pip install pytest
python -m pytest tests/ -q     # 185 项;不需要视频/API key/GPU(cv2 自动 stub,LLM 走假件)
```

覆盖:帧索引文件名↔时间戳往返(各 fps)、窗口覆盖校验、MCQ 结构校验、中文 ROUGE-L、
音频事件边界、详述质量门与重试、马赛克去重复帧与状态机、抗捷径通道与并发确定性、
出题跨轮去重、落盘装配、自动生成 meta 的 schema 合法性、预处理与批量标注 CLI 开关;
另有 S1~S7 端到端回归(`test_pipeline_e2e.py`,假 LLM + 假帧索引跑完整条链路,
校验 schema 合法性、断点续跑零成本、并发不改变产物)。
