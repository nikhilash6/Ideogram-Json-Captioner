import unittest
from pathlib import Path

from ideogram_captioner.gui import CaptionEditorApp


class FakeListbox:
    def __init__(self, selection: tuple[int, ...] = ()) -> None:
        self.selection = set(selection)
        self.activated: int | None = None
        self.seen: int | None = None
        self.anchor: int | None = None

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


if __name__ == "__main__":
    unittest.main()
