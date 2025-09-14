# Bore-Log Extractor (Streamlit)

Streamlit app to extract soil layers and SPT N-values from borehole log PDFs/images.

## Quickstart (local)
```bash
git clone https://github.com/<your-username>/bore-log-extractor.git
cd bore-log-extractor
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Install Tesseract:
#   macOS: brew install tesseract
#   Ubuntu/Debian: sudo apt-get install tesseract-ocr
streamlit run app.py
```

## Deploy (Streamlit Cloud)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io/ and connect the repo.
3. Set `app.py` as the main file.  
4. Keep `packages.txt` (installs tesseract-ocr automatically).

## Usage
1. Upload a PDF or page images.
2. Draw rectangles for `description_col`, `nvalue_col`, and optional header/boxes.
3. Click **Pick Top** and **Pick Bottom** to mark depth span (per page or apply to all).
4. Tune **Separator sensitivity** and **Min band height** → _Preview separators_.
5. **Extract & Download** to get Excel output.

> Notes: OCR quality improves with high-resolution scans.
