from __future__ import annotations

import json
import re
from pathlib import Path

from .core import Subtitle


def _seconds(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = value.replace(",", ".").split(":")
    if len(parts) == 1:
        return float(parts[0])
    h, m, s = map(float, parts)
    return h * 3600 + m * 60 + s


def load_subtitles(path: str | Path) -> list[Subtitle]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Subtitle(_seconds(x["start"]), _seconds(x["end"]), x.get("text", x.get("line", ""))) for x in data]
    if path.suffix.lower() == ".srt":
        result = []
        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
            lines = block.splitlines()
            timeline = next((line for line in lines if "-->" in line), None)
            if timeline:
                start, end = (part.strip() for part in timeline.split("-->", 1))
                idx = lines.index(timeline)
                text = re.sub(r"<[^>]+>", "", " ".join(lines[idx + 1 :])).strip()
                result.append(Subtitle(_seconds(start), _seconds(end), text))
        return result
    raise ValueError("subtitle format must be .json or .srt")
