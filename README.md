# Assembly Assistant

Local Flask prototype for turning uploaded build media into editable assembly instruction sections and packages.

## Run Locally

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open:

```text
http://127.0.0.1:8765/
```

## Notes

- Runtime data, uploads, extracted media, transcripts, and settings live under `storage/`.
- `storage/` is intentionally ignored by git.
- OpenAI API key and local whisper paths are entered in the app Settings page.
- `ffmpeg` is required for real video/audio processing.
- OpenCV is optional but recommended for better frame scoring.
