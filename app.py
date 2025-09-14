# Bore-Log Extractor — Streamlit (v5 – Clip to Depth)
# --------------------------------------------------------------
# Quickstart:
#   pip install -U -r requirements.txt
#   (or: pip install -U streamlit streamlit-drawable-canvas Pillow numpy pandas openpyxl pytesseract pymupdf)
#   Install Tesseract: macOS `brew install tesseract`; Ubuntu/Debian `sudo apt-get install tesseract-ocr`
#   streamlit run app.py
# --------------------------------------------------------------

from __future__ import annotations
import io
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pytesseract
import fitz  # PyMuPDF

import streamlit as st

# --------------------------------------------------------------------
# Compatibility shim for streamlit-drawable-canvas on newer Streamlit
# (restores expected symbol streamlit.elements.image.image_to_url)
# --------------------------------------------------------------------
import sys, types
try:
    from streamlit.elements import image as _st_image  # noqa: F401
except Exception:
    from streamlit import image_utils as _st_image_utils
    shim = types.ModuleType("streamlit.elements.image")
    shim.image_to_url = _st_image_utils.image_to_url
    sys.modules["streamlit.elements.image"] = shim

from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Bore-Log Extractor v5 — Streamlit", layout="wide")

# -------------------------------
# Data structures
# -------------------------------
RegionName = str
REGION_CHOICES: List[RegionName] = [
    "description_col",
    "nvalue_col",
    "header",
    "bore_box",
    "lat_box",
    "lon_box",
    "water_box",
    "elev_box",
    "footer",
]

@dataclass
class DepthPick:
    top_y: Optional[float] = None  # absolute pixel (page image coordinates)
    bot_y: Optional[float] = None

@dataclass
class PageState:
    w: int
    h: int
    image: Image.Image  # PIL image in RGB

@dataclass
class AppState:
    pages: List[PageState] = field(default_factory=list)
    curr: int = 0
    regions: Dict[int, Dict[RegionName, Tuple[float, float, float, float]]] = field(default_factory=dict)
    depth_px: Dict[int, DepthPick] = field(default_factory=dict)
    page_offsets: Dict[int, float] = field(default_factory=dict)
    # params
    top_ft: float = 0.0
    bot_ft: float = 36.0
    snap_top: bool = True
    snap_bot: bool = True
    clip_depth: bool = True
    sep_frac: float = 0.45
    min_band: int = 40

    def regs(self, i: int) -> Dict[RegionName, Tuple[float, float, float, float]]:
        if i not in self.regions:
            self.regions[i] = {}
        return self.regions[i]

    def dpx(self, i: int) -> DepthPick:
        if i not in self.depth_px:
            self.depth_px[i] = DepthPick()
        return self.depth_px[i]

SS: AppState
if "SS" not in st.session_state:
    st.session_state.SS = AppState()
SS = st.session_state.SS

# -------------------------------
# Utilities
# -------------------------------
def pdf_to_images(file_bytes: bytes) -> List[PageState]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: List[PageState] = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better OCR
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(PageState(w=img.width, h=img.height, image=img))
    return pages

def images_from_upload(files: List[io.BytesIO]) -> List[PageState]:
    out = []
    for f in files:
        im = Image.open(f).convert("RGB")
        out.append(PageState(w=im.width, h=im.height, image=im))
    return out

def crop(page: PageState, rel_rect: Tuple[float, float, float, float]) -> Image.Image:
    x0r, y0r, x1r, y1r = rel_rect
    x0, y0, x1, y1 = int(x0r * page.w), int(y0r * page.h), int(x1r * page.w), int(y1r * page.h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(page.w - 1, x1), min(page.h - 1, y1)
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        x1 = min(page.w, x0 + 2)
        y1 = min(page.h, y0 + 2)
    return page.image.crop((x0, y0, x1, y1))

def image_to_gray_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.uint8)

def split_bands_with_params(img: Image.Image, sep_frac: float, min_band: int):
    """Return {bands, seps}; works in local image pixels (y up-down)."""
    g = image_to_gray_np(img)
    h, w = g.shape
    dark = (g < 190).astype(np.uint8)

    frac = np.zeros(h, dtype=np.float32)
    segs = np.zeros(h, dtype=np.uint16)
    for y in range(h):
        row = dark[y]
        cov = int(row.sum())
        s = 0
        run = 0
        for v in row:
            if v:
                run += 1
            elif run > 0:
                s += 1
                run = 0
        if run > 0:
            s += 1
        frac[y] = cov / w
        segs[y] = s

    smooth = np.copy(frac)
    smooth[1:-1] = (frac[:-2] + frac[1:-1] + frac[2:]) / 3.0

    strong = max(sep_frac, 0.45)
    dashed = 0.25
    dashed_segs = max(10, w // 60)

    mask = np.where((smooth >= strong) | ((smooth >= dashed) & (segs >= dashed_segs)), 1, 0).astype(np.uint8)

    seps: List[int] = []
    s = -1
    for y in range(h):
        if mask[y] and s < 0:
            s = y
        if ((not mask[y]) or y == h - 1) and s >= 0:
            e = y if mask[y] else (y - 1)
            seps.append(int(round((s + e) / 2)))
            s = -1

    cuts = [0] + seps + [h - 1]
    bands: List[Tuple[int, int]] = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        if (b - a) >= max(6, int(min_band)):
            bands.append((a, b))

    return {"bands": bands, "seps": seps}

def ocr_text(img: Image.Image, psm: int = 6) -> str:
    cfg = f"--psm {psm}"
    text = pytesseract.image_to_string(img, config=cfg) or ""
    return " ".join(text.split())

def ocr_digits_with_pos(img: Image.Image):
    """Return list of dicts: {val, x, y, w, h} for pure integer tokens (<=3 digits)."""
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    out = []
    n = len(data.get("text", []))
    for i in range(n):
        s = (data["text"][i] or "").strip()
        if s.isdigit() and 1 <= len(s) <= 3:
            try:
                val = int(s)
            except ValueError:
                continue
            out.append({"val": val, "x": data["left"][i], "y": data["top"][i], "w": data["width"][i], "h": data["height"][i]})
    return out

def depth_at_y(y_abs: float, page_index: int, ss: AppState) -> Optional[float]:
    dp = ss.dpx(page_index)
    if dp.top_y is None or dp.bot_y is None or dp.top_y == dp.bot_y:
        return None
    base = ss.page_offsets.get(page_index, 0.0)
    td = ss.top_ft + base
    bd = ss.bot_ft + base
    return td + ((y_abs - dp.top_y) / (dp.bot_y - dp.top_y)) * (bd - td)

# -------------------------------
# Sidebar — Inputs and controls
# -------------------------------
with st.sidebar:
    st.title("Bore-Log Extractor v5")
    st.caption("Streamlit port — mark regions, pick depth, extract to Excel")

    pdf_file = st.file_uploader("Load PDF", type=["pdf"], accept_multiple_files=False)
    st.write("— or —")
    img_files = st.file_uploader("Load page images (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if st.button("Load"):
        if pdf_file is not None:
            SS.pages = pdf_to_images(pdf_file.read())
        elif img_files:
            SS.pages = images_from_upload([f for f in img_files])
        else:
            st.warning("Please upload a PDF or one or more images.")
        SS.curr = 0

    st.divider()

    apply_all = st.checkbox("Apply to all pages", value=True)
    region_name = st.selectbox("Region", REGION_CHOICES, index=0)
    draw_mode = st.radio("Marking mode", ["Rectangle", "Pick Top", "Pick Bottom"], horizontal=True)

    st.divider()
    SS.top_ft = st.number_input("Top depth (ft)", value=float(SS.top_ft), step=0.1)
    SS.bot_ft = st.number_input("Bottom depth (ft)", value=float(SS.bot_ft), step=0.1)
    SS.snap_top = st.checkbox("Snap first band to Top", value=SS.snap_top)
    SS.snap_bot = st.checkbox("Snap last band to Bottom", value=SS.snap_bot)
    SS.clip_depth = st.checkbox("Clip bands to Top/Bottom", value=SS.clip_depth)

    st.divider()
    SS.sep_frac = st.number_input("Separator sensitivity", min_value=0.10, max_value=0.60, step=0.01, value=float(SS.sep_frac))
    SS.min_band = st.number_input("Min band height (px)", min_value=6, step=1, value=int(SS.min_band))

    st.divider()
    if st.button("Save Config JSON"):
        cfg = {
            "regions": SS.regions,
            "depthPx": {k: v.__dict__ for k, v in SS.depth_px.items()},
            "topFt": SS.top_ft,
            "botFt": SS.bot_ft,
            "sepFrac": SS.sep_frac,
            "minBand": SS.min_band,
        }
        st.download_button(
            "Download borelog_config.json",
            data=json.dumps(cfg, indent=2).encode("utf-8"),
            file_name="borelog_config.json",
            mime="application/json",
        )

    cfg_up = st.file_uploader("Load Config JSON", type=["json"], accept_multiple_files=False, key="cfg")
    if cfg_up is not None and st.button("Apply Loaded Config"):
        cfg = json.loads(cfg_up.read())
        SS.regions = cfg.get("regions", {})
        dp = cfg.get("depthPx", {})
        SS.depth_px = {int(k): DepthPick(**v) for k, v in dp.items()}
        SS.top_ft = float(cfg.get("topFt", SS.top_ft))
        SS.bot_ft = float(cfg.get("botFt", SS.bot_ft))
        SS.sep_frac = float(cfg.get("sepFrac", SS.sep_frac))
        SS.min_band = int(cfg.get("MinBand", SS.min_band)) if "MinBand" in cfg else int(cfg.get("minBand", SS.min_band))
        st.success("Config applied.")

# -------------------------------
# Main layout
# -------------------------------
left, right = st.columns([5, 4], vertical_alignment="top")

with left:
    st.subheader("2) Mark Regions & Depth")
    if not SS.pages:
        st.info("Upload a PDF or image pages on the left, then click Load.")
    else:
        # nav
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("◀ Prev", use_container_width=True, disabled=(SS.curr == 0)):
                SS.curr = max(0, SS.curr - 1)
        with c2:
            st.markdown(f"**Page {SS.curr+1} / {len(SS.pages)}**")
        with c3:
            if st.button("Next ▶", use_container_width=True, disabled=(SS.curr >= len(SS.pages) - 1)):
                SS.curr = min(len(SS.pages) - 1, SS.curr + 1)

        page = SS.pages[SS.curr]

        # Prepare a background image for the canvas
        bg = page.image
        canvas_w = min(1200, page.w)  # cap width for perf
        scale = canvas_w / page.w
        canvas_h = int(page.h * scale)
        bg_disp = bg.resize((canvas_w, canvas_h), Image.BILINEAR)

        # Overlays (rectangles + depth lines)
        overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)

        # Existing rectangles
        regs = SS.regs(SS.curr)
        for name, (x0r, y0r, x1r, y1r) in regs.items():
            x0, y0 = int(x0r * canvas_w), int(y0r * canvas_h)
            x1, y1 = int(x1r * canvas_w), int(y1r * canvas_h)
            color = {
                "description_col": (0, 200, 83, 160),  # green
                "nvalue_col": (255, 23, 68, 160),      # red
            }.get(name, (255, 213, 79, 120))            # amber
            odraw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        # Top/Bottom lines
        dp = SS.dpx(SS.curr)
        if dp.top_y is not None:
            ty = int(dp.top_y * scale)
            odraw.line([(0, ty), (canvas_w, ty)], fill=(0, 255, 255, 200), width=2)
        if dp.bot_y is not None:
            by = int(dp.bot_y * scale)
            odraw.line([(0, by), (canvas_w, by)], fill=(255, 0, 255, 200), width=2)

        # Compose background+overlay → PIL RGB (important for st_canvas)
        bg_for_canvas = Image.alpha_composite(bg_disp.convert("RGBA"), overlay).convert("RGB")

        st.caption("Draw a rectangle for the selected region, or click once to set Top/Bottom when in pick modes.")
        canvas_res = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#4db6ac",
            background_color="#FFFFFF",
            background_image=bg_for_canvas,     # pass PIL RGB image
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="rect" if draw_mode == "Rectangle" else "point",
            key=f"canvas_{SS.curr}_{region_name}_{draw_mode}",
        )

        # Interpret canvas interactions
        if canvas_res.json_data is not None:
            objs = canvas_res.json_data.get("objects", [])
            if draw_mode == "Rectangle":
                rect = None
                for o in objs:
                    if o.get("type") == "rect":
                        left0 = o.get("left", 0.0)
                        top0 = o.get("top", 0.0)
                        w0 = o.get("width", 1.0) * o.get("scaleX", 1.0)
                        h0 = o.get("height", 1.0) * o.get("scaleY", 1.0)
                        rect = (left0, top0, left0 + w0, top0 + h0)
                if rect is not None:
                    x0, y0, x1, y1 = rect
                    rel = (x0 / canvas_w, y0 / canvas_h, x1 / canvas_w, y1 / canvas_h)
                    if apply_all:
                        for i in range(len(SS.pages)):
                            SS.regs(i)[region_name] = rel
                    else:
                        SS.regs(SS.curr)[region_name] = rel
            else:
                pt = None
                for o in objs:
                    if o.get("type") == "circle":
                        left0 = o.get("left", 0.0)
                        top0 = o.get("top", 0.0)
                        r = o.get("radius", 2.0)
                        cx = left0 + r
                        cy = top0 + r
                        pt = (cx, cy)
                if pt is not None:
                    x, y = pt
                    y_abs = (y / canvas_h) * page.h  # convert to page space
                    if draw_mode == "Pick Top":
                        if apply_all:
                            for i in range(len(SS.pages)):
                                SS.dpx(i).top_y = y_abs
                        else:
                            SS.dpx(SS.curr).top_y = y_abs
                    elif draw_mode == "Pick Bottom":
                        if apply_all:
                            for i in range(len(SS.pages)):
                                SS.dpx(i).bot_y = y_abs
                        else:
                            SS.dpx(SS.curr).bot_y = y_abs

        if st.button("Preview separators on this page"):
            r = SS.regs(SS.curr)
            if "description_col" not in r:
                st.warning("Draw the description_col region first.")
            else:
                desc_img = crop(page, r["description_col"])
                info = split_bands_with_params(desc_img, SS.sep_frac, SS.min_band)
                preview = desc_img.convert("RGBA").copy()
                d = ImageDraw.Draw(preview)
                for y in info["seps"]:
                    d.line([(0, y), (preview.width - 1, y)], fill=(0, 229, 255, 200), width=2)
                st.image(preview, caption="Separator preview (description_col)")

with right:
    st.subheader("3) Extract → Excel")

    def compute_page_offsets():
        SS.page_offsets = {}
        span = SS.bot_ft - SS.top_ft
        last_b = None
        acc = 0.0
        for i, pg in enumerate(SS.pages):
            bore = ""
            r = SS.regs(i)
            if r.get("bore_box"):
                t = ocr_text(crop(pg, r["bore_box"]))
                import re
                m = re.search(r"([A-Z]{2}-?\d{1,3})", t, re.I)
                if m:
                    bore = m.group(1).upper()
            if not bore and r.get("header"):
                t = ocr_text(crop(pg, r["header"]))
                import re
                m = re.search(r"\b([A-Z]{2}-?\d{1,3})\b", t)
                if m:
                    bore = m.group(1).upper()
            if last_b is None or bore != last_b:
                acc = 0.0
                last_b = bore
            SS.page_offsets[i] = acc
            acc += span

    def extract_rows() -> List[List]]:
        compute_page_offsets()
        rows: List[List] = []
        import re
        for i, pg in enumerate(SS.pages):
            r = SS.regs(i)
            if "description_col" not in r:
                continue
            bore = ""
            lat = ""
            lon = ""
            elev = ""
            water = "N/E"

            if r.get("header"):
                t = ocr_text(crop(pg, r["header"]))
                m = re.search(r"\b([A-Z]{2}-?\d{1,3})\b", t, re.I)
                bore = (m.group(1) if m else "")
                m = re.search(r"LATITUDE.*?([+\-]?\d+\.\d+)", t, re.I)
                lat = m.group(1) if m else lat
                m = re.search(r"LONGITUDE.*?([+\-]?\d+\.\d+)", t, re.I)
                lon = m.group(1) if m else lon
                m = re.search(r"ELEVATION.*?(\d+\.\d+)", t, re.I)
                elev = m.group(1) if m else elev
                m = re.search(r"DEPTH.*?WATER.*?(?:INITIAL.*?([\d.]+))?(?:.*?AFTER.*?24.*?HOURS.*?([\d.]+))?", t, re.I)
                if m:
                    water = (m.group(1) or m.group(2) or "N/E")

            if r.get("bore_box"):
                t = ocr_text(crop(pg, r["bore_box"]))
                m = re.search(r"([A-Z]{2}-?\d{1,3})", t, re.I)
                if m:
                    bore = m.group(1)

            if r.get("lat_box"):
                t = ocr_text(crop(pg, r["lat_box"]))
                m = re.search(r"([+\-]?\d+\.\d+)", t)
                if m:
                    lat = m.group(1)

            if r.get("lon_box"):
                t = ocr_text(crop(pg, r["lon_box"]))
                m = re.search(r"([+\-]?\d+\.\d+)", t)
                if m:
                    lon = m.group(1)

            if r.get("elev_box"):
                t = ocr_text(crop(pg, r["elev_box"]))
                m = re.search(r"(\d+\.\d+)", t)
                if m:
                    elev = m.group(1)

            if r.get("water_box"):
                t = ocr_text(crop(pg, r["water_box"]))
                m = re.search(r"(\d+\.\d+)", t)
                if m:
                    water = m.group(1)

            # bands from description col
            desc_img = crop(pg, r["description_col"])
            info = split_bands_with_params(desc_img, SS.sep_frac, SS.min_band)
            bands = info["bands"]

            # clip to top/bottom picks in page space
            dp2 = SS.dpx(i)
            rx0, ry0, rx1, ry1 = r["description_col"]
            y0_abs_base = ry0 * pg.h

            if SS.clip_depth and dp2.top_y is not None and dp2.bot_y is not None:
                b2 = []
                for (a, b) in bands:
                    A = y0_abs_base + a
                    B = y0_abs_base + b
                    AA = max(A, dp2.top_y)
                    BB = min(B, dp2.bot_y)
                    if (BB - AA) > max(6, SS.min_band / 2):
                        b2.append((int(AA - y0_abs_base), int(BB - y0_abs_base)))
                bands = b2

            # OCR blocks
            blocks = []
            for (y0, y1) in bands:
                tmp = desc_img.crop((0, y0, desc_img.width, y1))
                t = ocr_text(tmp, psm=4).replace("|", " ").strip()
                if t and t.lower() != "description":
                    blocks.append({
                        "y0_abs": y0_abs_base + y0,
                        "y1_abs": y0_abs_base + y1,
                        "text": t,
                        "nvals": [],
                    })

            # N-value column association
            if r.get("nvalue_col"):
                nimg = crop(pg, r["nvalue_col"])
                nums = ocr_digits_with_pos(nimg)
                ny0_abs = r["nvalue_col"][1] * pg.h
                for blk in blocks:
                    vals = []
                    for n in nums:
                        cy = ny0_abs + n["y"] + n["h"] / 2
                        if blk["y0_abs"] <= cy <= blk["y1_abs"]:
                            vals.append(n["val"])
                    uniq = []
                    for v in vals:
                        if not uniq or uniq[-1] != v:
                            uniq.append(v)
                    blk["nvals"] = uniq

            # Emit rows
            for k, b in enumerate(blocks):
                y0a = b["y0_abs"]
                y1a = b["y1_abs"]
                if k == 0 and SS.snap_top and dp2.top_y is not None:
                    y0a = dp2.top_y
                if k == len(blocks) - 1 and SS.snap_bot and dp2.bot_y is not None:
                    y1a = dp2.bot_y
                d0 = depth_at_y(y0a, i, SS)
                d1 = depth_at_y(y1a, i, SS)
                from_ft = round(d0, 1) if d0 is not None else ""
                to_ft = round(d1, 1) if d1 is not None else ""
                elev_from = round(float(elev) - from_ft, 1) if elev and from_ft != "" else ""
                elev_to = round(float(elev) - to_ft, 1) if elev and to_ft != "" else ""
                rows.append([
                    bore or "",
                    from_ft,
                    to_ft,
                    ", ".join(map(str, b["nvals"])) if b["nvals"] else "N/A",
                    b["text"],
                    lon or "",
                    lat or "",
                    elev or "",
                    water or "N/E",
                    elev_from,
                    elev_to,
                ])
        return rows

    if st.button("Extract & Download"):
        if not SS.pages:
            st.warning("Load pages first.")
        else:
            rows = extract_rows()
            header = [
                "Bore L.",
                "From (ft)",
                "To (ft)",
                "SPT N-Value",
                "Soil Layer Description",
                "Longitude",
                "Latitude",
                "Top Elevation (ft)",
                "Water Table (ft)",
                "Elevation From (ft)",
                "Elevation To (ft)",
            ]
            df = pd.DataFrame(rows, columns=header)
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as xw:
                df.to_excel(xw, sheet_name="Bore Logs", index=False)
            st.download_button(
                "Download bore_logs_v5.xlsx",
                data=out.getvalue(),
                file_name="bore_logs_v5.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.success(f"Done. Rows: {len(rows)}")

st.markdown("""
---
**Notes**
- Load a PDF or page images, then draw regions in **Rectangle** mode and click to set **Top/Bottom** lines.
- For best OCR, upload high-resolution scans.
- *Separator sensitivity* and *Min band height* control band splitting; use **Preview separators** to verify.
""")
