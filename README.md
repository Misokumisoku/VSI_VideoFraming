# VSI 视频关键帧抽取复现

这是对 `Reference/2025_VSI_Visual_Subtitle_Integration.pdf` 中 VSI 算法的独立复现。核心算法不绑定模型，便于单测、替换检测器以及做消融；`vsi/adapters.py` 提供论文所需文本编码器与 YOLO-World 的可选适配器。

## 对应论文

- 式 (4)：检测器适配器返回 `max(confidence * object_weight)`。
- 式 (8)：`soft_threshold`。
- 式 (9)-(10)：字幕分数的高斯传播与 max 聚合。
- 式 (11)：视觉/文本 Z-score 融合。
- 式 (13)-(15)：已访问帧的自然三次样条插值、概率下界、Sigmoid 归一化。
- Algorithm 1：首轮均匀抽样，后续按更新后的概率无放回抽样，预算耗尽后返回 Top-K。

论文没有指定随机种子、样条边界条件、相同分数的排序规则。本实现分别采用固定 seed、natural cubic spline、稳定排序。为避免仅凭字幕返回未经视觉确认的帧，最终 Top-K 限定在检测器已访问帧内。

## 安装与测试

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

## 对白蛇视频直接抽帧

默认输入为工作区中的 `test_data/白蛇：浮生.mp4`，默认以 YOLO-World 搜索
`person,horse,road`。该 MP4 没有独立字幕轨，字幕是烧录在画面中的硬字幕；程序会先对画面下部做中文 OCR，合并为带时间戳的字幕段，然后运行 VSI 的视觉+字幕双分支：

```bash
cd "/Users/libu/Documents/THU Courses/Huawei/VSI_reproduce"
python examples/run_vsi.py
```

结果写入 `output/白蛇浮生/`：8 张关键帧 JPG 和 `result.json`。也可以覆盖参数：

```bash
python examples/run_vsi.py \
  --video "../test_data/白蛇：浮生.mp4" \
  --objects "person,horse,road" \
  --question "Find scenes where a person is riding a horse on a road." \
  --top-k 8 \
  --budget 64 \
  --device cpu
```

OCR 结果缓存为 `output/白蛇浮生/ocr_subtitles.json`，后续运行会直接复用。若提供质量更高的 SRT 或 JSON，则优先使用外部字幕：

```bash
python examples/run_vsi.py --subtitles "../test_data/白蛇：浮生.srt" \
  --question "什么时候有人骑马经过道路？"
```

中文字幕默认由 `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` 编码。若传入英文字幕，可用 `--text-model sentence-transformers/all-mpnet-base-v2` 严格切回论文模型。OCR 只负责恢复 VSI 所需的“文本+起止时间”；之后仍执行论文的余弦相似度、软阈值增强、高斯时间传播和跨模态融合。传 `--no-ocr` 可关闭语义分支。

核心 API：

```python
from vsi import VSIConfig, select_keyframes

result = select_keyframes(
    n_frames=1000,
    fps=25,
    object_scorer=lambda indices: detector_scores(indices),
    text_scores=per_frame_subtitle_scores,
    config=VSIConfig(top_k=8, samples_per_round=16, detection_budget=64),
)
print(result.frame_indices, result.timestamps)
```

端到端模式使用 `opencv-python`、`ultralytics`、`easyocr` 和 `sentence-transformers`。使用 `SentenceTransformerMatcher` 计算问题/字幕余弦相似度，依次调用 `soft_threshold` 和 `subtitle_frame_scores` 得到逐帧文本分数；使用 `UltralyticsYOLOWorldScorer` 作为 `object_scorer`。模型权重首次使用时会下载。

## 与作者公开代码的差异

作者当前公开的 `VSI_keyframe_search.py` 在融合处使用 min-max normalization，而论文式 (11) 明确写 Z-score；本复现遵循论文。作者脚本的 soft-threshold 实际写法也与论文式 (8) 不同；本实现同样以论文公式为准。这两处都适合后续做 paper/code 两套配置的消融对比。
