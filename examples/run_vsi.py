"""Run the reproduced VSI algorithm on one video and save selected frames."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / "output" / "model_cache" / "ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "output" / "model_cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "output" / "model_cache" / "matplotlib"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vsi.adapters import DEFAULT_TEXT_MODEL, SentenceTransformerMatcher, UltralyticsYOLOWorldScorer
from vsi.core import VSIConfig, select_keyframes, soft_threshold, subtitle_frame_scores
from vsi.io import load_subtitles
from vsi.ocr import extract_burned_subtitles


DEFAULT_VIDEO = WORKSPACE_ROOT / "test_data" / "白蛇：浮生.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "白蛇浮生"


def _parse_objects(value: str) -> list[str]:
    objects = [item.strip() for item in value.split(",") if item.strip()]
    if not objects:
        raise argparse.ArgumentTypeError("at least one object is required")
    return objects


def _save_frames(video: Path, indices: list[int], output_dir: Path) -> list[str]:
    import cv2

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    saved: list[str] = []
    try:
        for rank, index in enumerate(indices, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to decode selected frame {index}")
            path = frames_dir / f"keyframe_{rank:02d}_frame_{index}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"failed to save {path}")
            saved.append(str(path))
    finally:
        cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-faithful VSI keyframe selection")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--objects", type=_parse_objects, default=_parse_objects("person,horse,road"),
                        help="comma-separated target and cue objects")
    parser.add_argument("--question", default="寻找有人骑马或出现在道路上的场景。")
    parser.add_argument("--subtitles", type=Path, default=None,
                        help="optional .srt/.json subtitles; takes priority over OCR")
    parser.add_argument("--no-ocr", action="store_true",
                        help="disable hard-subtitle OCR when no subtitle file is supplied")
    parser.add_argument("--ocr-fps", type=float, default=2.0)
    parser.add_argument("--ocr-crop-top", type=float, default=0.62)
    parser.add_argument("--ocr-cache", type=Path, default=None)
    parser.add_argument("--text-model", default=str(DEFAULT_TEXT_MODEL))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=64,
                        help="total number of frames evaluated by YOLO-World")
    parser.add_argument("--samples-per-round", type=int, default=16)
    parser.add_argument("--text-weight", type=float, default=0.3)
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Missing OpenCV. Install: pip install opencv-python ultralytics") from exc

    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"Video does not exist: {video}")
    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if n_frames <= 0 or fps <= 0:
        raise SystemExit(f"Could not read video metadata: {video}")

    text_scores = None
    effective_text_weight = 0.0
    subtitles = []
    subtitle_source = None
    if args.subtitles is not None:
        subtitle_path = args.subtitles.expanduser().resolve()
        if not subtitle_path.is_file():
            raise SystemExit(f"Subtitle file does not exist: {subtitle_path}")
        subtitles = load_subtitles(subtitle_path)
        subtitle_source = str(subtitle_path)
    elif not args.no_ocr:
        cache_path = args.ocr_cache or (args.output_dir / "ocr_subtitles.json")
        print(f"Extracting/reusing burned-in subtitles: {cache_path}")
        subtitles = extract_burned_subtitles(
            video, sample_fps=args.ocr_fps, crop_top_ratio=args.ocr_crop_top,
            cache_path=cache_path, device=args.device,
        )
        subtitle_source = f"OCR:{Path(cache_path).expanduser().resolve()}"

    if subtitles:
        matcher = SentenceTransformerMatcher(model=args.text_model, device=args.device)
        similarities = soft_threshold(matcher(args.question, [s.text for s in subtitles]))
        text_scores = subtitle_frame_scores(subtitles, similarities, n_frames, fps)
        effective_text_weight = args.text_weight
        mode = "visual+subtitle"
        print(f"Loaded {len(subtitles)} timed subtitle segments from {subtitle_source}.")
    else:
        mode = "visual-only"
        print("No subtitle text found: running visual-only with text_weight=0.")

    print(f"Video: {video}")
    print(f"Frames: {n_frames}, FPS: {fps:.3f}, duration: {n_frames / fps:.1f}s")
    print(f"Mode: {mode}; objects: {', '.join(args.objects)}")
    detector = UltralyticsYOLOWorldScorer(
        str(video), args.objects, model=args.model, device=args.device
    )
    result = select_keyframes(
        n_frames,
        fps,
        detector,
        text_scores=text_scores,
        config=VSIConfig(
            top_k=args.top_k,
            samples_per_round=args.samples_per_round,
            detection_budget=args.budget,
            text_weight=effective_text_weight,
            seed=args.seed,
        ),
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_frames = _save_frames(video, result.frame_indices, output_dir)
    records = []
    for index, timestamp, path in zip(result.frame_indices, result.timestamps, saved_frames):
        record = {
            "frame_index": index,
            "timestamp_seconds": timestamp,
            "score": float(result.fused_scores[index]),
            "image": path,
        }
        records.append(record)
        print(f"frame={index}\ttime={timestamp:.3f}s\tscore={record['score']:.6f}\t{path}")

    summary = {
        "video": str(video),
        "mode": mode,
        "subtitle_source": subtitle_source,
        "subtitle_segments": len(subtitles),
        "text_model": args.text_model if subtitles else None,
        "question": args.question,
        "objects": args.objects,
        "fps": fps,
        "total_frames": n_frames,
        "rounds": result.rounds,
        "visited_frames": result.visited_indices,
        "keyframes": records,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} keyframes and metadata to: {output_dir}")


if __name__ == "__main__":
    main()
