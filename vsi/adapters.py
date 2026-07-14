from __future__ import annotations

from typing import Sequence

import numpy as np


class SentenceTransformerMatcher:
    """Lazy all-mpnet-base-v2 subtitle matcher used by the VSI paper."""

    def __init__(self, model: str = "sentence-transformers/all-mpnet-base-v2", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("install the 'models' extra to use subtitle encoding") from exc
        self.model = SentenceTransformer(model, device=device)

    def __call__(self, question: str, subtitles: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode([question, *subtitles], normalize_embeddings=True)
        return np.asarray(embeddings[1:] @ embeddings[0], dtype=float)


class UltralyticsYOLOWorldScorer:
    """Batched Equation (4) scorer backed by Ultralytics YOLO-World."""

    def __init__(self, video_path: str, objects: Sequence[str], weights: dict[str, float] | None = None,
                 model: str = "yolov8s-worldv2.pt", device: str | None = None):
        try:
            import cv2
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError("install the 'models' extra to use video detection") from exc
        self.cv2, self.video_path, self.objects = cv2, video_path, list(objects)
        self.weights, self.device = weights or {}, device
        self.model = YOLOWorld(model)
        self.model.set_classes(self.objects)

    def __call__(self, indices: Sequence[int]) -> np.ndarray:
        cap = self.cv2.VideoCapture(self.video_path)
        frames = []
        for index in indices:
            cap.set(self.cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"failed to decode frame {index}")
            frames.append(frame)
        cap.release()
        results = self.model.predict(frames, verbose=False, device=self.device)
        output = []
        for result in results:
            best = 0.0
            for cls, conf in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                name = self.objects[int(cls)]
                best = max(best, float(conf) * self.weights.get(name, 1.0))
            output.append(best)
        return np.asarray(output)
