# MILK Thesis V1 Demo

Local desktop application (no browser) for:
- image inference using `efficientnetv2l_multilabel_final.keras`
- thresholded multi-label decisions using `efficientnetv2l_multilabel_final_thresholds.npy`
- automatic HiResCAM overlays for top predicted classes

## Folder

- `desktop_app.py` - local GUI app (`tkinter`)
- `app.py` - older Streamlit version (optional)
- `requirements.txt` - dependencies

## Run Local App

From the `thesis_v1_demo` folder:

```bash
pip install -r requirements.txt
python desktop_app.py
```

If your model was trained in `.venv-tfdml`, use:

```bash
..\.venv-tfdml\Scripts\python.exe desktop_app.py
```

(`run_local_app.bat` already tries this interpreter first.)

## Notes

- By default, the app looks for artifacts in:
  - `../notebooks/efficientnetv2l_multilabel_final.keras` (preferred)
  - `../notebooks/efficientnetv2l_multilabel_final_thresholds.npy`
