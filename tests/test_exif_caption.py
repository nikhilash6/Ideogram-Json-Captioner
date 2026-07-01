import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from ideogram_captioner.exif_caption import try_import_caption_from_exif, workflow_text_candidates
from ideogram_captioner.schema import normalize_caption
from ideogram_captioner.store import CaptionStore


CAPTION_WITH_OBJ_BBOX = {
    "high_level_description": "A red box on white",
    "style_description": {
        "aesthetics": "minimal",
        "lighting": "flat",
        "photo": "studio",
        "medium": "photograph",
    },
    "compositional_deconstruction": {
        "background": "white",
        "elements": [{"type": "obj", "bbox": [100, 200, 300, 400], "desc": "red box"}],
    },
}

CAPTION_TEXT_ONLY_BBOX = {
    "high_level_description": "Sign",
    "compositional_deconstruction": {
        "background": "wall",
        "elements": [{"type": "text", "bbox": [10, 20, 30, 40], "text": "SALE", "desc": "letters"}],
    },
}


def _save_png_with_comfy_prompt(path: Path, node_texts: list[str], *, class_type: str = "CLIPTextEncode") -> None:
    prompt: dict[str, dict] = {}
    for index, text in enumerate(node_texts, start=1):
        prompt[str(index)] = {
            "class_type": class_type,
            "inputs": {"text": text},
            "_meta": {"title": f"Prompt {index}"},
        }

    image = Image.new("RGB", (8, 8), color="white")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", json.dumps(prompt))
    image.save(path, pnginfo=metadata)


def _save_png_with_caption_metadata(path: Path, caption: dict) -> None:
    image = Image.new("RGB", (8, 8), color="white")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("caption", json.dumps(caption))
    image.save(path, pnginfo=metadata)


def _save_png_with_comfy_workflow(path: Path, node_texts: list[str]) -> None:
    workflow = {
        "nodes": [
            {
                "id": index,
                "type": "CLIPTextEncode",
                "title": f"Prompt {index}",
                "widgets_values": [text],
            }
            for index, text in enumerate(node_texts, start=1)
        ]
    }
    image = Image.new("RGB", (8, 8), color="white")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("workflow", json.dumps(workflow))
    image.save(path, pnginfo=metadata)


class ExifCaptionTests(unittest.TestCase):
    def test_workflow_text_candidates_sort_by_length_desc(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            _save_png_with_comfy_prompt(image_path, ["short", "x" * 30, "medium length text here"])

            candidates = workflow_text_candidates(image_path, limit=5)

            self.assertEqual(candidates[0], "x" * 30)
            self.assertEqual(len(candidates), 3)

    def test_imports_valid_obj_bbox_json_from_workflow_text(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            valid_json = json.dumps(CAPTION_WITH_OBJ_BBOX, separators=(",", ":"))
            invalid_json = '{"high_level_description":"no bbox"}'
            _save_png_with_comfy_prompt(image_path, [invalid_json, valid_json])

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")
            self.assertEqual(
                normalize_caption(caption)["compositional_deconstruction"]["elements"][0]["bbox"],
                [100, 200, 300, 400],
            )

    def test_imports_valid_obj_bbox_json_from_ui_workflow_text(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            _save_png_with_comfy_workflow(image_path, [json.dumps(CAPTION_WITH_OBJ_BBOX)])

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")

    def test_imports_json_wrapped_in_markdown_fence(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            fenced = "```json\n" + json.dumps(CAPTION_WITH_OBJ_BBOX, indent=2) + "\n```"
            _save_png_with_comfy_prompt(image_path, [fenced])

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")

    def test_imports_direct_caption_json_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            _save_png_with_caption_metadata(image_path, CAPTION_WITH_OBJ_BBOX)

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")

    def test_imports_jpeg_exif_image_description(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.jpg"
            image = Image.new("RGB", (8, 8), color="white")
            exif = Image.Exif()
            exif[270] = json.dumps(CAPTION_TEXT_ONLY_BBOX)
            image.save(image_path, exif=exif)

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")
            self.assertEqual(caption["high_level_description"], "Sign")

    def test_imports_caption_without_obj_bbox(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            _save_png_with_comfy_prompt(image_path, [json.dumps(CAPTION_TEXT_ONLY_BBOX)])

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNotNone(caption)
            self.assertIn("Imported caption JSON", message or "")
            self.assertEqual(caption["high_level_description"], "Sign")

    def test_ignores_workflow_without_caption_json(self):
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.png"
            _save_png_with_comfy_prompt(image_path, [json.dumps({"not_a_caption": {"x": 1}})])

            caption, message = try_import_caption_from_exif(image_path)

            self.assertIsNone(caption)
            self.assertIsNone(message)


class StoreExifImportTests(unittest.TestCase):
    def test_missing_caption_file_imports_from_image_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image_path = folder / "sample.png"
            _save_png_with_comfy_prompt(image_path, [json.dumps(CAPTION_WITH_OBJ_BBOX)])
            store = CaptionStore(folder, ".json")

            caption, message = store.load_caption(image_path)

            self.assertIn("Imported caption JSON", message or "")
            self.assertEqual(caption["high_level_description"], "A red box on white")
            self.assertFalse(store.caption_path(image_path).exists())

    def test_existing_caption_file_takes_priority_over_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image_path = folder / "sample.png"
            _save_png_with_comfy_prompt(image_path, [json.dumps(CAPTION_WITH_OBJ_BBOX)])
            sidecar = {
                "high_level_description": "From sidecar",
                "compositional_deconstruction": {"background": "", "elements": []},
            }
            store = CaptionStore(folder, ".json")
            store.save_caption(image_path, sidecar)

            caption, message = store.load_caption(image_path)

            self.assertIsNone(message)
            self.assertEqual(caption["high_level_description"], "From sidecar")


if __name__ == "__main__":
    unittest.main()
