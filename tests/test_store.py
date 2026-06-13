import json
import tempfile
import unittest
from pathlib import Path

from ideogram_captioner.store import CaptionStore


class StoreTests(unittest.TestCase):
    def test_lists_images_and_saves_matching_caption_stem(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.PNG"
            image.write_bytes(b"not actually loaded by the store")
            (folder / "sample.txt").write_text("plain caption", encoding="utf-8")

            store = CaptionStore(folder, ".caption")
            self.assertEqual(store.images(), [image])

            saved_path = store.save_caption(
                image,
                {
                    "high_level_description": "A sign",
                    "compositional_deconstruction": {"background": "wall", "elements": []},
                },
            )

            self.assertEqual(saved_path, folder / "sample.caption")
            raw = saved_path.read_text(encoding="utf-8")
            self.assertNotIn(": ", raw)
            self.assertEqual(json.loads(raw)["high_level_description"], "A sign")

    def test_imports_plain_text_caption_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.jpg"
            image.write_bytes(b"x")
            (folder / "sample.txt").write_text("a concise plain caption", encoding="utf-8")

            caption, message = CaptionStore(folder, ".txt").load_caption(image)

            self.assertIn("Imported plain text", message)
            self.assertEqual(caption["high_level_description"], "a concise plain caption")

    def test_missing_caption_loads_blank_and_saves_new_json(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.jpg"
            image.write_bytes(b"x")
            store = CaptionStore(folder, ".json")

            caption, message = store.load_caption(image)
            self.assertIn("click Save to create it", message)
            self.assertEqual(caption["high_level_description"], "")

            caption["high_level_description"] = "A new caption"
            saved_path = store.save_caption(image, caption)

            self.assertEqual(saved_path, folder / "sample.json")
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["high_level_description"], "A new caption")
            self.assertIn("style_description", saved)
            self.assertIn("compositional_deconstruction", saved)

    def test_failure_marker_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.jpg"
            image.write_bytes(b"x")
            store = CaptionStore(folder, ".json")

            path = store.save_failure_marker(
                image,
                {
                    "operation": "json_image",
                    "reason": "caption_json_parse_failed",
                    "error": "Could not parse model JSON",
                },
            )

            self.assertEqual(path, folder / "sample.caption_failed.json")
            self.assertTrue(store.has_failure_marker(image))
            marker = store.load_failure_marker(image)
            self.assertIsNotNone(marker)
            self.assertEqual(marker["operation"], "json_image")
            self.assertTrue(store.clear_failure_marker(image))
            self.assertFalse(store.has_failure_marker(image))
            self.assertFalse(store.clear_failure_marker(image))

    def test_edit_folder_is_not_listed_as_source_images(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            edit_folder = folder / "edit"
            edit_folder.mkdir()
            (edit_folder / "sample.jpg").write_bytes(b"x")

            self.assertEqual(CaptionStore(edit_folder, ".json").images(), [])


if __name__ == "__main__":
    unittest.main()
