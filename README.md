# Ideogram-JSON-Captioner

A local desktop editor for image-caption pairs in Ideogram 4 structured JSON
format. It is designed for reviewing image datasets, fixing generated captions,
and drawing or adjusting object/text bounding boxes without leaving the folder
you are working in.

The app runs locally by default. Manual editing never uploads images or captions
anywhere. Auto-captioning talks to the OpenAI-compatible endpoint you configure
in Preferences, which can be a local llama.cpp/LM Studio/vLLM server or another
compatible service.

## Features

- Open a folder of images and step through them with keyboard shortcuts.
- Edit Ideogram JSON fields for high-level description, style, background,
  elements, rendered text, bounding boxes, and color palettes.
- Draw, move, resize, remove, and numerically edit bounding boxes.
- Color-coded object boxes on the canvas and in the element list.
- Maintain a separate editable original caption file, such as `.txt` or
  `.original`, while saving structured JSON separately.
- Generate or redo plain text captions, structured JSON captions from text,
  structured JSON captions directly from the image, and bbox coordinates.
- Run auto-captioning jobs against one image or a shift/ctrl-selected batch, with
  confirmation before multi-image jobs.
- Undo the last auto-captioning job by restoring the previous sidecar files.
- Configure built-in model profiles, custom Hugging Face GGUF repos/files, model
  download location, endpoint/API keys, bbox behavior, and optional server
  commands.
- Copy the current image into an `edit` subfolder for later Photoshop work.
- Sort the image list by name, modified date, missing structured captions, or
  missing original captions.
- Autosave edits and save manually with the Save button.

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

Images are listed even when they do not have caption files yet. A missing
structured caption opens as a blank Ideogram caption, and saving creates a new
file with the selected caption extension.

The separate `Original files` dropdown lets you load and edit plain source
captions such as `photo_001.txt` or `photo_001.original` while keeping the main
structured caption output separate. If the original extension would point at the
same file as the active structured caption extension, the original editor is
disabled to prevent accidental overwrite.

## Auto Captioning

Use `Preferences` to configure the captioning runtime. The default runtime is
`Local llama.cpp`, which downloads the selected Hugging Face GGUF model files,
starts `llama-server.exe`, and sends OpenAI-compatible requests to
`http://127.0.0.1:8000/v1`.

For the local runtime, put `llama-server.exe` beside the app, under `tools/`, or
select it in Preferences. The app checks the model folder before each job and
downloads missing model or `mmproj` files into the configured `models`
directory. The default model profile is Qwen2.5-VL 7B Q4 because it is much more
practical for first-time local use than a 30B model.

Model profiles are loaded from `captioner_model_profiles.json` beside the repo
or packaged `.exe`. This is a local, ignored file so pulls do not overwrite user
edits. The tracked `captioner_model_profiles.example.json` file is the shipped
seed/default copy; if the local file does not exist, the app reads that example
when present, otherwise it falls back to defaults compiled into the app.

Use `Preferences` -> `Models` -> `Open Profiles File` to create/open the local
editable copy in the base folder. A downloadable Hugging Face profile looks
like:

```json
{
  "id": "my-qwen-profile",
  "label": "Download: My Qwen model",
  "tasks": ["caption", "bbox"],
  "kind": "hf",
  "api_model": "my-qwen",
  "hf_repo": "org/repo-name",
  "model_filename": "model-file.gguf",
  "mmproj_filename": "mmproj-file.gguf"
}
```

`tasks` can be `["caption"]`, `["bbox"]`, or both. `kind` can be:

- `hf`: download GGUF files from Hugging Face.
- `local`: use fixed local `local_model_path` and optional
  `local_mmproj_path` from the profile file.
- `server`: do not download or launch a model; just send requests using
  `api_model` as the existing server's model name.
  Use this with the `Connect to existing server` runtime and make sure the base
  URL points at the running server, such as `http://127.0.0.1:8000/v1`.

The profile dropdown also has two one-off choices that are not stored in the
profile file:

- `Custom Hugging Face GGUF`: shows HF repo and filename fields in Preferences.
- `Custom local GGUF files`: shows Browse buttons for a model GGUF and optional
  mmproj file already on your computer.

Prompt text can be overridden with plain `.txt` files in the ignored
`captioner_prompts/` folder, so private prompt changes stay local. Use
`Preferences` -> `Pipeline` -> `Open Prompts Folder` to create/open the folder.
The tracked `captioner_prompts.example/` folder shows every supported filename.
You can override only the files you want; missing files fall back to defaults.

Keep these placeholders if you edit the matching prompt:

- `text_to_json_user.txt`: `{caption}`
- `bbox_user.txt`: `{context_json}`, `{targets_json}`

`creative_directive.txt` or `faithful_directive.txt` is appended to the JSON
system prompt automatically.

The right-side `Auto Captioning` buttons do the following:

- `Text Caption`: generates or replaces the plain original caption sidecar.
- `JSON from Text`: converts the original caption sidecar into Ideogram JSON.
- `JSON from Image`: captions the image directly into Ideogram JSON without
  using the original text caption.
- `Add/redo BBoxes`: localizes existing JSON elements with the configured VLM.
- `Undo AI`: restores sidecar files from before the last auto-captioning job.

Multiple images can be selected in the image list with Shift, Ctrl, or `Ctrl+A`.
The app asks for confirmation before running an auto-captioning job on more than
one image.

Model files are checked before a job starts. Built-in downloadable profiles and
custom Hugging Face profiles are downloaded under the configured `models`
folder, which defaults to a `models` directory beside the repo or packaged
`.exe`. Custom HF profiles need a repo id plus the GGUF model filename, and for
vision models usually an `mmproj` filename.

Advanced runtime options are also available:

- `Connect to existing server`: use LM Studio, llama.cpp, vLLM, Ollama bridges,
  or another OpenAI-compatible endpoint you started yourself.
- `Custom start commands`: let the app start your own command templates. They
  support `{model_path}`, `{mmproj_path}`, `{models_dir}`, `{api_model}`, and
  `{base_url}` placeholders.

A generated local llama.cpp command is equivalent to:

```powershell
llama-server.exe -m "{model_path}" --mmproj "{mmproj_path}" --host 127.0.0.1 --port 8000 --alias qwen25vl
```

If the endpoint is already running, the app reuses it instead of launching a
second server.

## Keyboard Shortcuts

- `Enter`: save and move to the next image.
- `Shift+Enter`: insert a newline in a text field.
- `Tab` / `Shift+Tab`: move forward or backward between fields.
- Arrow keys: navigate images when focus is not inside an input.
- `Ctrl+Up` / `Ctrl+Down`: navigate images even when an input has focus.
- `Ctrl+S`: save.
- `Ctrl+A` in the image list: select all images for batch auto-captioning.
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
