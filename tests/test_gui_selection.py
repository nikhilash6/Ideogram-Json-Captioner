import tempfile
import unittest
from pathlib import Path

from ideogram_captioner.gui import CAPTION_FILTER_BOTH, CAPTION_FILTER_JSON, CAPTION_FILTER_ORIGINAL, CaptionEditorApp


class FakeListbox:
    def __init__(self, selection: tuple[int, ...] = ()) -> None:
        self.selection = set(selection)
        self.activated: int | None = None
        self.seen: int | None = None
        self.anchor: int | None = None
        self.xview: float | None = None
        self.yview_top = 0.0

    def curselection(self) -> tuple[int, ...]:
        return tuple(sorted(self.selection))

    def nearest(self, y: int) -> int:
        return int(y)

    def bbox(self, index: int) -> tuple[int, int, int, int]:
        return (0, int(index), 10, 1)

    def selection_clear(self, _first: int, _last: str) -> None:
        self.selection.clear()

    def selection_set(self, first: int, last: int | str | None = None) -> None:
        if last is None:
            self.selection.add(int(first))
            return
        if last == "end":
            raise AssertionError("FakeListbox needs an explicit item count for selection_set(..., 'end').")
        for index in range(int(first), int(last) + 1):
            self.selection.add(index)

    def activate(self, index: int) -> None:
        self.activated = int(index)

    def see(self, index: int) -> None:
        self.seen = int(index)

    def xview_moveto(self, fraction: float) -> None:
        self.xview = float(fraction)

    def yview(self) -> tuple[float, float]:
        return (self.yview_top, min(1.0, self.yview_top + 0.25))

    def yview_moveto(self, fraction: float) -> None:
        self.yview_top = float(fraction)

    def selection_anchor(self, index: int) -> None:
        self.anchor = int(index)


class FakeEvent:
    def __init__(self, y: int, state: int = 0) -> None:
        self.y = y
        self.state = state


class GuiSelectionTests(unittest.TestCase):
    def make_app(self, images: list[Path], selection: tuple[int, ...], current_index: int) -> CaptionEditorApp:
        app = object.__new__(CaptionEditorApp)
        app.images = images
        app.image_list = FakeListbox(selection)
        app.current_index = current_index
        app.loading_list = False
        app.pending_image_select_job = None
        app.last_image_click_index = None
        app.image_selection_anchor_path = None
        app.after_idle = lambda callback: "job"
        app.after_cancel = lambda _job: None
        return app

    def test_current_image_selection_paths_deduplicates_selected_indices(self) -> None:
        images = [Path("a.png"), Path("b.png"), Path("c.png")]
        app = self.make_app(images, (0, 2), 0)

        self.assertEqual(app.current_image_selection_paths(), [Path("a.png"), Path("c.png")])

    def test_populate_image_selection_restores_paths_after_reorder(self) -> None:
        images = [Path("c.png"), Path("a.png"), Path("b.png")]
        app = self.make_app(images, (), 2)

        app.populate_image_selection(selected_paths=[Path("a.png"), Path("c.png")])

        self.assertEqual(app.image_list.curselection(), (0, 1))
        self.assertEqual(app.image_list.activated, 2)
        self.assertEqual(app.image_list.seen, 2)

    def test_populate_image_selection_falls_back_to_current_when_paths_disappear(self) -> None:
        images = [Path("a.png"), Path("b.png"), Path("c.png")]
        app = self.make_app(images, (), 1)

        app.populate_image_selection(selected_paths=[Path("missing.png")])

        self.assertEqual(app.image_list.curselection(), (1,))
        self.assertEqual(app.image_list.activated, 1)
        self.assertEqual(app.image_list.seen, 1)

    def test_populate_image_selection_can_preserve_list_viewport(self) -> None:
        images = [Path("a.png"), Path("b.png"), Path("c.png")]
        app = self.make_app(images, (), 1)

        app.populate_image_selection(reveal_current=False)

        self.assertEqual(app.image_list.curselection(), (1,))
        self.assertEqual(app.image_list.activated, 1)
        self.assertIsNone(app.image_list.seen)
        self.assertEqual(app.image_list.xview, 0.0)

    def test_horizontal_image_list_scroll_is_blocked(self) -> None:
        app = self.make_app([Path("a.png")], (), 0)

        self.assertEqual(app.block_image_list_horizontal_scroll(), "break")
        self.assertEqual(app.image_list.xview, 0.0)

    def test_image_list_vertical_scroll_can_be_restored(self) -> None:
        app = self.make_app([Path("a.png")], (), 0)
        app.image_list.yview_top = 0.6
        scroll = app.image_list_vertical_scroll()

        app.image_list.yview_top = 0.0
        app.restore_image_list_vertical_scroll(scroll)

        self.assertEqual(app.image_list.yview_top, 0.6)

    def test_shift_click_selects_range_from_saved_anchor(self) -> None:
        images = [Path("a.png"), Path("b.png"), Path("c.png"), Path("d.png"), Path("e.png")]
        app = self.make_app(images, (1,), 1)
        app.image_selection_anchor_path = Path("b.png")

        result = app.remember_image_list_click(FakeEvent(y=4, state=0x0001))

        self.assertEqual(result, "break")
        self.assertEqual(app.image_list.curselection(), (1, 2, 3, 4))
        self.assertEqual(app.image_list.activated, 4)
        self.assertEqual(app.image_list.seen, 4)
        self.assertEqual(app.image_list.anchor, 1)

    def test_caption_filter_paths_deduplicates_same_caption_source(self) -> None:
        image = Path("sample.png")

        self.assertEqual(
            CaptionEditorApp.caption_filter_paths_for_image(image, ".txt", ".txt", CAPTION_FILTER_BOTH),
            [Path("sample.txt")],
        )

    def test_caption_filter_matches_json_or_original_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.png"
            image.write_bytes(b"x")
            image.with_suffix(".json").write_text('{"high_level_description":"bright red sign"}', encoding="utf-8")
            image.with_suffix(".txt").write_text("plain caption with blue paint", encoding="utf-8")

            self.assertTrue(
                CaptionEditorApp.image_matches_caption_filter(
                    image, "red sign", ".json", ".txt", CAPTION_FILTER_JSON
                )
            )
            self.assertFalse(
                CaptionEditorApp.image_matches_caption_filter(
                    image, "blue paint", ".json", ".txt", CAPTION_FILTER_JSON
                )
            )
            self.assertTrue(
                CaptionEditorApp.image_matches_caption_filter(
                    image, "blue paint", ".json", ".txt", CAPTION_FILTER_ORIGINAL
                )
            )
            self.assertTrue(
                CaptionEditorApp.image_matches_caption_filter(
                    image, "bright red", ".json", ".txt", CAPTION_FILTER_BOTH
                )
            )

    def test_json_caption_style_fields_infers_mode_and_medium(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            caption = folder / "sample.json"
            caption.write_text(
                '{"style_description":{"_mode":"art_style","medium":"painting","photo":"ignored","art_style":"oil"}}',
                encoding="utf-8",
            )

            self.assertEqual(CaptionEditorApp.json_caption_style_fields(caption), ("art_style", "painting"))

    def test_image_matches_style_filter_uses_json_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.png"
            image.write_bytes(b"x")
            image.with_suffix(".json").write_text(
                '{"style_description":{"photo":"85mm lens","medium":"Photograph"}}',
                encoding="utf-8",
            )

            self.assertTrue(CaptionEditorApp.image_matches_style_filter(image, ".json", "photo", "photograph"))
            self.assertTrue(CaptionEditorApp.image_matches_style_filter(image, ".json", "", "photograph"))
            self.assertFalse(CaptionEditorApp.image_matches_style_filter(image, ".json", "art_style", "photograph"))
            self.assertFalse(CaptionEditorApp.image_matches_style_filter(image, ".json", "photo", "painting"))


if __name__ == "__main__":
    unittest.main()
