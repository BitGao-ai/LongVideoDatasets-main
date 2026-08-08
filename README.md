# LongVideo-Dataset · 长视频理解测评集 · LLM 自动标注

用 **Qwen(DashScope)/ Kimi(Moonshot)** 多模态大模型 API,对 **30min–2h 长视频**做自动数据标注,
输出严格符合 `schema/annotation.schema.json` 的中英双语标注结果。

> ⚠️ 机器标注结果 `annotation_meta.review_status = "draft"`,是**人工复核前的初标**。
> 按数据集方案,LLM 只做辅助扩量,**必须经人工验证 + 抗捷径过滤**后才能进入正式基准。

---

## 目录结构

```
LongVideo-daatset/
├── requirements.txt
├── preprocess.py                 # ① 预处理:PySceneDetect + WhisperX + PaddleOCR
├── run_annotate.py               # ② 标注 CLI 入口
├── run_describe.py               # ③ 给已标注文件补全片详述 CLI
├── run_rescreen.py               # ④ 换模型复筛 CLI
├── run_qc.py                     # ⑤ 黄金题 + 一致性质检 CLI
├── schema/annotation.schema.json # 输出格式(双语 schema)
├── examples/
│   ├── meta_template.json        # 每个视频需人工提供的元信息模板
│   └── responses_template.json   # 验证者独立作答记录模板(质检输入)
└── annotator/
    ├── config.py                 # Qwen/Kimi 供应商 + 运行参数
    ├── llm_client.py             # 统一多模态客户端(OpenAI 兼容 + 重试 + JSON 抽取)
    ├── media.py                  # 抽帧/编码、字幕&OCR 切片、镜头合并成窗口
    ├── prompts.py                # 双语提示词(场景/全局/出题/过滤)
    ├── annotate.py               # 场景+事件 逐窗标注、全局聚合(梗概/人物/关系/因果)
    ├── qa_generate.py            # 出题(能力标签/证据时间戳/干扰项)
    ├── filter.py                 # 抗捷径过滤(盲答/单帧/字幕)+ check_one/should_drop
    ├── describe.py               # 全片详细描述(分段密集详述 + 两级汇总;frames/native_video 双模式)
    ├── native_video.py           # native_video 模式:ffmpeg 切片 + DashScope 原生视频调用(仅 Qwen)
    ├── rescreen.py               # 换模型复筛(不同供应商重跑抗捷径,判定取或)
    ├── qc.py                     # Krippendorff α / Cohen κ / 黄金题 / 打回(纯标准库)
    └── pipeline.py               # 端到端编排 + schema 校验 + 落盘
```

## 流水线

```
视频 + 人工元信息(meta)
  → preprocess.py         : shots / subtitles / ocr
  → annotate_structure    : 逐 180s 窗口送【采样帧+字幕+OCR】给多模态模型 → scenes / events
  → annotate_global       : 汇总场景摘要 → 梗概 / 人物表 / 关系 / 故事时间线 / 事件因果图
  → generate_qa           : 按能力体系出题(MCQ 为主,带证据时间戳+干扰项陷阱)
  → run_filters           : 盲答/单帧/字幕三重抗捷径,命中即标记(默认剔除)
  → 组装为 schema JSON,校验后落盘(review_status=draft)
```

**长视频如何塞进 API**:不整段上传;按镜头合并成 ~180s 窗口,每窗仅采样 10 帧(长边≤768px)
送多模态模型;全局聚合与出题只用**文本**(场景摘要),把 token/成本压到可控。

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
```

> 模型名在 `annotator/config.py` 里改。用户提到的 “Qwen3.7-plus” 不是真实模型 id,
> 请按你开通的版本填,例如 `qwen-vl-max` / `qwen2.5-vl-72b-instruct`(视觉)、`qwen-plus`(文本);
> Kimi 用 `moonshot-v1-32k-vision-preview`(视觉)、`moonshot-v1-32k`(文本)。

## 运行

```bash
# 1) 预处理(产出 structure)
python preprocess.py --video videos/doc_00001.mp4 --video-id doc_00001 \
    --lang en --out structure/doc_00001.json --hf-token $HF_TOKEN

# 2) 准备 meta(拷贝模板改字段:license/source/release_date 等须人工填)
cp examples/meta_template.json meta/doc_00001.json

# 3) LLM 标注
python run_annotate.py \
    --meta      meta/doc_00001.json \
    --structure structure/doc_00001.json \
    --out       annotations/doc_00001.json \
    --provider  qwen --qa 12

# 调试:只跑前 2 个窗口、保留(不剔除)捷径题
python run_annotate.py ... --max-windows 2 --keep-shortcut
```

批量:外层对视频列表循环调用 `annotator.pipeline.annotate_video(...)`,失败重试即可。

## 全片详细描述(detailed description)

在标注概要(scene summary / synopsis)之外,可另出**详细描述**:逐 ~180s 窗口生成
中英**密集详述**(画面/动作/镜头切换/字幕对白要点,非概要),写入带时间戳的
`structure.segments[]`;再两级汇总成连贯的 `global.detailed_description`(中英)。

**两种模式,用 `--describe-mode` 手动切换:**

| 模式 | 送模型方式 | 适用 | 依赖 |
|---|---|---|---|
| `frames`(默认) | 每窗抽帧 + 字幕 + OCR | Qwen / Kimi 通用,成本低 | opencv |
| `native_video` | ffmpeg 切片,把**真实视频片段**直接喂 Qwen-VL | **仅 Qwen**,更"直接"、成本高 | ffmpeg + `pip install dashscope` |

> 30min–2h 无论哪种模式都**无法一次喂整段**(Qwen-VL 原生视频也有时长/帧数上限),
> 故都是按窗口分块;`native_video` 是把每个窗口切成短视频片段直接送模型,而非抽帧。
> Kimi 无原生视频输入,选 `native_video` + `--provider kimi` 会直接报错并提示改回 `frames`。

```bash
# 抽帧模式(默认):随标注一起产出
python run_annotate.py --meta meta/doc_00001.json --structure structure/doc_00001.json \
    --out annotations/doc_00001.json --provider qwen --describe --frames 12

# 原生视频模式:仅 Qwen,需先 `brew install ffmpeg` 且 `pip install dashscope`
python run_annotate.py --meta meta/doc_00001.json --structure structure/doc_00001.json \
    --out annotations/doc_00001.json --provider qwen \
    --describe --describe-mode native_video --native-fps 2

# 给已标注文件补详述(同样支持切换模式)
python run_describe.py --in annotations/ --out-dir annotations_desc/ --provider qwen \
    --describe-mode native_video --native-fps 2
```

调参:`--native-fps` 控制 DashScope 对片段的抽帧率(越高越细、越贵);切片临时文件默认
用完即删,设 `RunConfig.native_clip_dir` 可保留以便排查。

## 换模型复筛(cross-model re-screening)

初标与其抗捷径过滤用的是同一家模型,存在**同源偏置**。发布前用【另一家】模型独立复筛:
对每题重跑盲答/单帧/字幕检查,两家判定**取或**(任一家能走捷径就算捷径),更保守可信。

```bash
# 初标用 qwen -> 复筛用 kimi(供应商须不同,否则跳过;--force 可强制)
export MOONSHOT_API_KEY=sk-xxx
python run_rescreen.py --in annotations/ --out-dir annotations_rescreened/ \
    --provider kimi --report reports/rescreen.json
```

每题追加 `cross_check`(保留两家原始判定),`anti_shortcut` 更新为合并判定,`checked_by`
记为 `auto:qwen+auto:kimi`。结果仍是 `draft`,须人工终审。

## 黄金题 + 一致性质检(QC)

按标注手册第 6–7 节:黄金题监控标注员(正确率 <0.9 触发复训)、验证者一致性
Krippendorff α(入库门槛 ≥0.8)、验证者与出题者不一致的题打回。纯标准库,无第三方依赖。

```bash
# 标准答案与黄金题(is_gold)自动从 --annotations 读取;
# 验证者独立作答见 examples/responses_template.json
python run_qc.py --annotations annotations/ --responses reviews/responses.json \
    --alpha-threshold 0.8 --gold-threshold 0.9 --report reports/qc.json
```

输出:α 及是否达标、每位标注员黄金题正确率与复训标记、需打回的题清单。
`annotator/qc.py` 也可作为库调用:`krippendorff_alpha / cohen_kappa / grade_gold / run_qc`。

## 输出

严格符合 `schema/annotation.schema.json`,含 `meta / media / structure(shots,scenes,events,
segments,subtitles,ocr) / global(synopsis,detailed_description,characters,relations,timeline) /
qa[] / annotation_meta`。每道 QA 带 `capability / lang_mode / evidence_spans / min_watch /
anti_shortcut / difficulty`。`segments` 与 `detailed_description` 仅在启用 `--describe` 时产出。

## 成本与限制

- 主要成本在**场景标注**(每窗一次多模态调用)。1 小时视频约 20 窗 → ~20 次视觉调用 +
  1 次全局 + 1 次出题 + 每题 3 次过滤;可用 `--frames` / `--window-sec` / `--qa` 调节。
- `scenes/events` 由 LLM 生成,时间戳可能有偏差;**因果边、人物合并需人工校验**。
- 抗捷径过滤用的是同一家模型,建议正式发布前**换一家模型复筛**,避免同源偏置。
- 结果是 `draft`,进入基准前务必经人工复核(见标注手册:一致性 α≥0.8、黄金题监控)。
