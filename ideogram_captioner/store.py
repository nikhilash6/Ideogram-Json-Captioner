from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exif_caption import try_import_caption_from_exif
from .schema import IMAGE_EXTENSIONS, caption_from_plain_text, default_caption, parse_caption_text, serialize_caption


class CaptionStore:
    def __init__(self, folder: str | Path, extension: str) -> None:
        self.folder = Path(folder)
        self.extension = extension

    def images(self) -> list[Path]:
        if self.folder.name.lower() == "edit":
            return []
        return sorted(
            [path for path in self.folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda path: path.name.lower(),
        )

    def caption_path(self, image_path: Path) -> Path:
        return image_path.with_suffix(self.extension)

    def failure_path(self, image_path: Path) -> Path:
        return image_path.with_suffix(".caption_failed.json")

    def load_failure_marker(self, image_path: Path) -> dict[str, Any] | None:
        path = self.failure_path(image_path)
        if not path.exists():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def has_failure_marker(self, image_path: Path) -> bool:
        return self.failure_path(image_path).exists()

    def save_failure_marker(self, image_path: Path, marker: dict[str, Any]) -> Path:
        path = self.failure_path(image_path)
        path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def clear_failure_marker(self, image_path: Path) -> bool:
        path = self.failure_path(image_path)
        if not path.exists():
            return False
        path.unlink()
        return True

    def load_caption(self, image_path: Path) -> tuple[dict[str, Any], str | None]:
        caption_path = self.caption_path(image_path)
        if not caption_path.exists():
            caption, message = try_import_caption_from_exif(image_path)
            if caption is not None:
                return caption, message
            return default_caption(), f"No {self.extension} JSON caption yet; edit fields or click Save to create it."

        raw = caption_path.read_text(encoding="utf-8-sig")
        if not raw.strip():
            return default_caption(), f"{caption_path.name} is empty."

        try:
            return parse_caption_text(raw), None
        except (json.JSONDecodeError, ValueError) as exc:
            if self.extension in {".txt", ".caption"}:
                return caption_from_plain_text(raw), f"Imported plain text from {caption_path.name}; save will convert it to Ideogram JSON."
            return default_caption(), f"Could not parse {caption_path.name}: {exc}"

    def save_caption(self, image_path: Path, caption: dict[str, Any]) -> Path:
        caption_path = self.caption_path(image_path)
        caption_path.write_text(serialize_caption(caption), encoding="utf-8")
        return caption_path
