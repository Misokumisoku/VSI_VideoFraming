from __future__ import annotations

import json
import re
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

from .core import Subtitle


def normalize_ocr_text(text: str) -> str:
    text = re.sub(r"\s+", "", text).strip()
    return re.sub(r"[^\w\u3400-\u9fff，。！？、：；,.!?-]", "", text)


def merge_ocr_observations(
    observations: Sequence[tuple[float, str]], *, sample_period: float,
    similarity_threshold: float = 0.72,
) -> list[Subtitle]:
    """Merge repeated OCR observations into timestamped subtitle segments."""
    segments: list[Subtitle] = []
    current_text = ""
    start = last = 0.0
    for timestamp, raw_text in observations:
        text = normalize_ocr_text(raw_text)
        if not text:
            if current_text and timestamp - last > sample_period * 1.5:
                segments.append(Subtitle(start, last + sample_period, current_text))
                current_text = ""
            continue
        similarity = SequenceMatcher(None, current_text, text).ratio() if current_text else 0.0
        if current_text and similarity >= similarity_threshold and timestamp - last <= sample_period * 1.5:
            if len(text) > len(current_text):
                current_text = text
            last = timestamp
        else:
            if current_text:
                segments.append(Subtitle(start, last + sample_period, current_text))
            current_text, start, last = text, timestamp, timestamp
    if current_text:
        segments.append(Subtitle(start, last + sample_period, current_text))
    return segments


def extract_burned_subtitles(
    video_path: str | Path, *, sample_fps: float = 2.0, crop_top_ratio: float = 0.62,
    languages: Sequence[str] = ("ch_sim", "en"), confidence_threshold: float = 0.30,
    cache_path: str | Path | None = None, device: str = "cpu",
    model_storage_directory: str | Path | None = None,
) -> list[Subtitle]:
    """OCR burned-in subtitles into the timed segments expected by VSI."""
    if not 0 < sample_fps <= 10:
        raise ValueError("sample_fps must be in (0, 10]")
    if not 0 <= crop_top_ratio < 1:
        raise ValueError("crop_top_ratio must be in [0, 1)")
    cache = Path(cache_path) if cache_path else None
    if cache and cache.is_file():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return [Subtitle(float(x["start"]), float(x["end"]), x["text"]) for x in data]
    try:
        import cv2
        import easyocr
    except ImportError as exc:
        raise RuntimeError("hard-subtitle OCR requires: pip install easyocr") from exc

    path = str(Path(video_path).expanduser().resolve())
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0 or n_frames <= 0:
        cap.release()
        raise RuntimeError(f"cannot open video for OCR: {path}")
    model_dir = Path(model_storage_directory) if model_storage_directory else (
        Path(__file__).resolve().parents[1] / "output" / "easyocr_models"
    )
    required_models = (model_dir / "craft_mlt_25k.pth", model_dir / "zh_sim_g2.pth")
    missing = [str(path) for path in required_models if not path.is_file()]
    if missing:
        raise RuntimeError(f"EasyOCR model files are missing: {', '.join(missing)}")
    reader = easyocr.Reader(
        list(languages),
        gpu=device not in {"cpu", "mps"},
        model_storage_directory=str(model_dir),
        download_enabled=False,
    )
    step = max(1, round(fps / sample_fps))
    observations: list[tuple[float, str]] = []
    try:
        for frame_index in range(0, n_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            crop = frame[int(frame.shape[0] * crop_top_ratio):, :]
            detections = reader.readtext(crop, detail=1, paragraph=False)
            lines = [(box, normalize_ocr_text(text)) for box, text, confidence in detections
                     if confidence >= confidence_threshold and normalize_ocr_text(text)]
            lines.sort(key=lambda item: min(point[1] for point in item[0]))
            observations.append((frame_index / fps, "".join(text for _, text in lines)))
    finally:
        cap.release()
    segments = merge_ocr_observations(observations, sample_period=1.0 / sample_fps)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([asdict(x) for x in segments], ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return segments
