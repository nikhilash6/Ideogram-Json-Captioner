# Ideogram-JSON-Captioner

A local desktop editor for image-caption pairs in Ideogram 4 structured JSON
format. It is designed for reviewing image datasets, fixing generated captions,
and drawing or adjusting object/text bounding boxes without leaving the folder
you are working in.

The app runs locally. It does not upload images or captions anywhere.  It does not yet support automatically captioning images.

## Features

- Open a folder of images and step through them with keyboard shortcuts.
- Edit Ideogram JSON fields for high-level description, style, background,
  elements, rendered text, bounding boxes, and color palettes.
- Draw, move, resize, remove, and numerically edit bounding boxes.
- Color-coded object boxes on the canvas and in the element list.
- Maintain a separate editable original caption file, such as `.txt` or
  `.original`, while saving structured JSON separately.
- Copy the current image into an `edit` subfolder for later Photoshop work.
- Sort the image list by name, modified date, missing structured captions, or
  missing original captions.
- Autosave edits and save manually with the Save button.
- It does not yet support running a model to caption or recaption automatically.  This may be added later.

## Install

The easiest way to install and use it Ideogram-JSON-Captioner in Windows is to grab the .exe from the releases section.  Alternatively,

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python -m ideogram_captioner
```

You can also use the launcher script:

```powershell
python run_captioner.py
```

## Caption Files

Caption files are matched by image stem. For `photo_001.png`, the app reads and
writes `photo_001.json`, `photo_001.txt`, or `photo_001.caption` depending on
the selected caption-file extension.

The separate `Original files` dropdown lets you load and edit plain source
captions such as `photo_001.txt` or `photo_001.original` while keeping the main
structured caption output separate. If the original extension would point at the
same file as the active structured caption extension, the original editor is
disabled to prevent accidental overwrite.

## Keyboard Shortcuts

- `Enter`: save and move to the next image.
- `Shift+Enter`: insert a newline in a text field.
- Arrow keys: navigate images when focus is not inside an input.
- `Ctrl+Up` / `Ctrl+Down`: navigate images even when an input has focus.
- `Ctrl+S`: save.
- `Esc`: cancel eyedropper mode or exit fullscreen.
- `F11`: toggle fullscreen.

## Ideogram 4 Format Notes

The app writes compact UTF-8 JSON using the Ideogram 4 caption structure:

- Top-level keys: `high_level_description`, `style_description`,
  `compositional_deconstruction`.
- `style_description` uses either `photo` or `art_style`, not both.
- `compositional_deconstruction.background` comes before `elements`.
- Element types are `obj` and `text`.
- Bounding boxes are `[y_min, x_min, y_max, x_max]` in normalized `0-1000`
  coordinates.
- Global palettes allow up to 16 uppercase `#RRGGBB` values.
- Element palettes allow up to 5 uppercase `#RRGGBB` values.

Reference: <https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md>


## License

MIT License. See [LICENSE](LICENSE).
