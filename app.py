
# app.py
import io
import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# PDF rendering
import fitz  # PyMuPDF

# ---- Page config
st.set_page_config(page_title="Bore-Log Extractor (Streamlit)", layout="wide")

# ---- Styles
st.markdown("""
<style>
.small-muted { color:#9fb0c8; font-size:0.9em }
.tag { font-size:12px; padding:2px 6px; border:1px solid #3a4b7d; border-radius:8px; margin-left:6px; color:#9fb0c8 }
</style>
""", unsafe_allow_html=True)

st.title("Bore-Log Extractor ")
st.markdown('<span class="tag">Streamlit</span>', unsafe_allow_html=True)

# ---- Helpers / Data structures
REGION_KEYS = [
    "description_col", "nvalue_col", "header", "bore_box",
    "lat_box", "lon_box", "water_box", "elev_box", "footer"
]

@dataclass
class PageState:
    regions: Dict[str, Tuple[int,int,int,int]] = field(default_factory=dict)  # x,y,w,h in image pixels
    top_y: Optional[int] = None
    bot_y: Optional[int] = None

def get_session():
    if "images" not in st.session_state:
        st.session_state.images = []  # list of PIL images (one per page)
    if "pages" not in st.session_state:
        st.session_state.pages = []   # list[PageState]
    if "curr" not in st.session_state:
        st.session_state.curr = 0
    if "page_offsets" not in st.session_state:
        st.session_state.page_offsets = {}
    if "cfg" not in st.session_state:
        st.session_state.cfg = {
            "topFt": 0.0, "botFt": 36.0,
            "snapTop": True, "snapBot": True, "clipDepth": True,
            "sepFrac": 0.45, "minBand": 40
        }
    return st.session_state

S = get_session()

# ---- File loaders
def load_pdf_to_images(file) -> List[Image.Image]:
    data = file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    imgs = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        # 5.0 scale similar to original renderScale
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        imgs.append(img)
    return imgs

def load_images(files) -> List[Image.Image]:
    out = []
    for f in files:
        img = Image.open(f).convert("RGB")
        out.append(img)
    return out

# ---- OCR helpers
def ocr_text(img: Image.Image, psm=6) -> str:
    cfg = f'--psm {psm}'
    text = pytesseract.image_to_string(img, config=cfg, lang="eng") or ""
    return re.sub(r"\s+", " ", text).strip()

def ocr_digits_with_pos(img: Image.Image) -> List[Dict]:
    cfg = '--psm 6 -c tessedit_char_whitelist=0123456789'
    data = pytesseract.image_to_data(img, config=cfg, lang="eng", output_type=pytesseract.Output.DICT)
    out = []
    n = len(data["text"])
    for i in range(n):
        s = (data["text"][i] or "").strip()
        if re.fullmatch(r"\d{1,3}", s):
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            out.append({"val": int(s), "x": x, "y": y, "w": w, "h": h})
    return out

# ---- Band splitting (port of JS logic)
def split_bands_with_params(img: Image.Image, sep_frac: float, min_band_px: int):
    # Convert to grayscale numpy
    g = np.array(ImageOps.grayscale(img), dtype=np.uint8)
    h, w = g.shape

    # "dark" mask
    dark = (g < 190).astype(np.uint8)

    # row-wise coverage and segments
    frac = np.zeros(h, dtype=np.float32)
    segs = np.zeros(h, dtype=np.uint16)
    for y in range(h):
        row = dark[y]
        cov = row.sum()
        # count segments of consecutive 1s
        s = 0; run = 0
        for v in row:
            if v:
                run += 1
            else:
                if run > 0:
                    s += 1
                    run = 0
        if run > 0:
            s += 1
        frac[y] = cov / w
        segs[y] = s

    # smooth
    smooth = np.copy(frac)
    smooth[1:-1] = (frac[:-2] + frac[1:-1] + frac[2:]) / 3.0

    strong = max(sep_frac, 0.45)
    dashed = 0.25
    dashed_segs = max(10, w // 60)

    mask = ((smooth >= strong) | ((smooth >= dashed) & (segs >= dashed_segs))).astype(np.uint8)

    # collect separators as midpoints of contiguous mask runs
    seps = []
    s = -1
    for y in range(h):
        if mask[y] and s < 0:
            s = y
        if ((not mask[y]) or (y == h - 1)) and s >= 0:
            e = y if mask[y] else (y - 1)
            seps.append(int(round((s + e) / 2)))
            s = -1

    cuts = [0] + seps + [h - 1]
    bands = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        if (b - a) >= max(6, min_band_px):
            bands.append((a, b))
    return {"bands": bands, "seps": seps}

# ---- Depth mapping
def depth_at_y(y: int, page_idx: int, top_px: Optional[int], bot_px: Optional[int], top_ft: float, bot_ft: float):
    if top_px is None or bot_px is None or top_px == bot_px:
        return None
    base = S.page_offsets.get(page_idx, 0.0)
    td = top_ft + base
    bd = bot_ft + base
    return td + ((y - top_px) / (bot_px - top_px)) * (bd - td)

# ---- UI: File upload
with st.expander("1) Load PDF or Page Images", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        pdf = st.file_uploader("PDF file", type=["pdf"])
        if st.button("Load PDF") and pdf is not None:
            S.images = load_pdf_to_images(pdf)
            S.pages = [PageState() for _ in S.images]
            S.curr = 0
            st.success(f"Rendered {len(S.images)} page(s).")
    with col2:
        imgs = st.file_uploader("PNG/JPG files (multi-select)", type=["png","jpg","jpeg"], accept_multiple_files=True)
        if st.button("Load Images") and imgs:
            S.images = load_images(imgs)
            S.pages = [PageState() for _ in S.images]
            S.curr = 0
            st.success(f"Loaded {len(S.images)} image page(s).")

    st.caption("Tip: If OCR fails, confirm that system Tesseract is installed and in your PATH.")

if not S.images:
    st.stop()

# ---- Controls
with st.expander("2) Mark Regions & Depth", expanded=True):
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        st.markdown("**Navigation**")
        if st.button("◀ Prev", use_container_width=True):
            S.curr = max(0, S.curr - 1)
        st.write(f"Page **{S.curr+1} / {len(S.images)}**")
        if st.button("Next ▶", use_container_width=True):
            S.curr = min(len(S.images) - 1, S.curr + 1)

    with c2:
        st.markdown("**Depth & Params**")
        cfg = S.cfg
        cfg["topFt"] = st.number_input("Top depth (ft)", value=float(cfg["topFt"]), step=0.1)
        cfg["botFt"] = st.number_input("Bottom depth (ft)", value=float(cfg["botFt"]), step=0.1)
        cfg["snapTop"] = st.checkbox("Snap first band to Top", value=bool(cfg["snapTop"]))
        cfg["snapBot"] = st.checkbox("Snap last band to Bottom", value=bool(cfg["snapBot"]))
        cfg["clipDepth"] = st.checkbox("Clip bands to Top/Bottom", value=bool(cfg["clipDepth"]))
        cfg["sepFrac"] = st.number_input("Separator sensitivity", min_value=0.10, max_value=0.60, step=0.01, value=float(cfg["sepFrac"]))
        cfg["minBand"] = st.number_input("Min band height (px)", min_value=6, step=1, value=int(cfg["minBand"]))
        apply_all = st.checkbox("Apply region to all pages", value=True)
        st.session_state.cfg = cfg

        # Save/Load config
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("Download Config JSON", use_container_width=True):
                cfg_payload = {
                    "regions": {i: vars(p)["regions"] for i, p in enumerate(S.pages)},
                    "depthPx": {i: {"top_y": p.top_y, "bot_y": p.bot_y} for i, p in enumerate(S.pages)},
                    "topFt": cfg["topFt"],
                    "botFt": cfg["botFt"],
                    "sepFrac": cfg["sepFrac"],
                    "minBand": cfg["minBand"],
                }
                st.download_button("Save borelog_config.json",
                                   data=json.dumps(cfg_payload, indent=2).encode("utf-8"),
                                   file_name="borelog_config.json",
                                   mime="application/json")
        with cc2:
            cfg_file = st.file_uploader("Load Config JSON", type=["json"])
            if cfg_file is not None and st.button("Load Config", use_container_width=True):
                cfg_data = json.loads(cfg_file.read())
                # restore
                S.pages = [PageState() for _ in S.images]
                for k, v in (cfg_data.get("regions") or {}).items():
                    idx = int(k)
                    if idx < len(S.pages):
                        S.pages[idx].regions = {rk: tuple(rv) for rk, rv in v.items()}
                for k, v in (cfg_data.get("depthPx") or {}).items():
                    idx = int(k)
                    if idx < len(S.pages):
                        S.pages[idx].top_y = v.get("top_y")
                        S.pages[idx].bot_y = v.get("bot_y")
                S.cfg["topFt"] = cfg_data.get("topFt", S.cfg["topFt"])
                S.cfg["botFt"] = cfg_data.get("botFt", S.cfg["botFt"])
                S.cfg["sepFrac"] = cfg_data.get("sepFrac", S.cfg["sepFrac"])
                S.cfg["minBand"] = cfg_data.get("minBand", S.cfg["minBand"])
                st.success("Config loaded.")

    with c3:
        st.markdown("**Region / Depth Picker**")
        img = S.images[S.curr]
        w, h = img.size
        region_name = st.selectbox("Region", REGION_KEYS, index=0)
        mode = st.radio("Mode", ["Rectangle", "Pick Top", "Pick Bottom"], horizontal=True)
        stroke_width = 2

        # Current region overlay
        overlays = []
        pg = S.pages[S.curr]
        # draw existing region as hint
        if region_name in pg.regions:
            x, y, rw, rh = pg.regions[region_name]
            overlays.append({"type":"rect", "left":x, "top":y, "width":rw, "height":rh})

        # Draw canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 255, 0.3)",
            stroke_width=stroke_width,
            background_image=img,
            height=h,
            width=w,
            drawing_mode="rect" if mode == "Rectangle" else "transform",
            initial_drawing=overlays,
            key=f"canvas-{S.curr}-{region_name}"
        )

        # Capture new rectangle (take the last object)
        if mode == "Rectangle" and canvas_result.json_data is not None:
            objs = canvas_result.json_data.get("objects", [])
            if objs:
                last = objs[-1]
                if last.get("type") == "rect":
                    x, y = int(last["left"]), int(last["top"])
                    rw, rh = int(last["width"]), int(last["height"])
                    if apply_all:
                        for p in S.pages:
                            p.regions[region_name] = (x, y, rw, rh)
                    else:
                        pg.regions[region_name] = (x, y, rw, rh)

        # Separate "click" canvas to pick top/bottom lines
        if mode in ("Pick Top", "Pick Bottom"):
            st.info("Click approximately where the line should be; drag makes no difference.")
            click = st.image(img, use_column_width=False)
            # Use a slider to set Y quickly (a lightweight alternative for click events).
            yy = st.slider("Y position (pixels)", min_value=0, max_value=h-1, value=int(h*0.1 if mode=="Pick Top" else h*0.9))
            if st.button("Set Line"):
                if mode == "Pick Top":
                    if apply_all:
                        for p in S.pages: p.top_y = yy
                    else:
                        pg.top_y = yy
                else:
                    if apply_all:
                        for p in S.pages: p.bot_y = yy
                    else:
                        pg.bot_y = yy

# ---- Preview separators
with st.expander("Preview Separators (Current Page)", expanded=False):
    pg = S.pages[S.curr]
    desc = pg.regions.get("description_col")
    if desc:
        x, y, rw, rh = desc
        crop = S.images[S.curr].crop((x, y, x+rw, y+rh))
        sb = split_bands_with_params(crop, S.cfg["sepFrac"], int(S.cfg["minBand"]))
        st.image(crop, caption="Description Column (Cropped)", use_column_width=True)
        st.write(f"Detected separators: {len(sb['seps'])} | Bands: {len(sb['bands'])}")
    else:
        st.warning("Set the description_col region first.")

# ---- Extract → Excel
with st.expander("3) Extract → Excel", expanded=True):
    if st.button("Extract & Download", type="primary"):
        rows = []
        # build page offsets based on span
        S.page_offsets = {}
        span = S.cfg["botFt"] - S.cfg["topFt"]
        last_bore = None
        acc = 0.0
        # Pre-pass to compute offsets
        for i in range(len(S.images)):
            bore = ""
            pg = S.pages[i]
            if "bore_box" in pg.regions:
                x,y,w,h = pg.regions["bore_box"]
                t = ocr_text(S.images[i].crop((x,y,x+w,y+h)), psm=6)
                m = re.search(r"([A-Z]{2}-?\d{1,3})", t, flags=re.I)
                if m: bore = m.group(1).upper()
            if not bore and "header" in pg.regions:
                x,y,w,h = pg.regions["header"]
                t = ocr_text(S.images[i].crop((x,y,x+w,y+h)), psm=6)
                m = re.search(r"\b([A-Z]{2}-?\d{1,3})\b", t)
                if m: bore = m.group(1).upper()
            if last_bore is None or bore != last_bore:
                acc = 0.0
                last_bore = bore
            S.page_offsets[i] = acc
            acc += span

        for i in range(len(S.images)):
            pg = S.pages[i]
            img = S.images[i]
            if "description_col" not in pg.regions:
                continue

            bore = lat = lon = elev = ""
            water = "N/E"
            # header / boxes
            if "header" in pg.regions:
                x,y,w,h = pg.regions["header"]
                t = ocr_text(img.crop((x,y,x+w,y+h)), psm=6)
                m = re.search(r"\b([A-Z]{2}-?\d{1,3})\b", t, flags=re.I)
                if m: bore = m.group(1)
                m = re.search(r"LATITUDE.*?([+\-]?\d+\.\d+)", t, flags=re.I)
                if m: lat = m.group(1)
                m = re.search(r"LONGITUDE.*?([+\-]?\d+\.\d+)", t, flags=re.I)
                if m: lon = m.group(1)
                m = re.search(r"ELEVATION.*?(\d+\.\d+)", t, flags=re.I)
                if m: elev = m.group(1)
                m = re.search(r"DEPTH.*?WATER.*?(?:INITIAL.*?([\d.]+))?(?:.*?AFTER.*?24.*?HOURS.*?([\d.]+))?", t, flags=re.I)
                if m:
                    water = m.group(1) or m.group(2) or "N/E"
            if "bore_box" in pg.regions:
                x,y,w,h = pg.regions["bore_box"]
                t = ocr_text(img.crop((x,y,x+w,y+h)), psm=6)
                m = re.search(r"([A-Z]{2}-?\d{1,3})", t, flags=re.I)
                if m: bore = m.group(1)
            if "lat_box" in pg.regions:
                x,y,w,h = pg.regions["lat_box"]
                m = re.search(r"([+\-]?\d+\.\d+)", ocr_text(img.crop((x,y,x+w,y+h)), psm=6))
                if m: lat = m.group(1)
            if "lon_box" in pg.regions:
                x,y,w,h = pg.regions["lon_box"]
                m = re.search(r"([+\-]?\d+\.\d+)", ocr_text(img.crop((x,y,x+w,y+h)), psm=6))
                if m: lon = m.group(1)
            if "elev_box" in pg.regions:
                x,y,w,h = pg.regions["elev_box"]
                m = re.search(r"(\d+\.\d+)", ocr_text(img.crop((x,y,x+w,y+h)), psm=6))
                if m: elev = m.group(1)
            if "water_box" in pg.regions:
                x,y,w,h = pg.regions["water_box"]
                m = re.search(r"(\d+\.\d+)", ocr_text(img.crop((x,y,x+w,y+h)), psm=6))
                if m: water = m.group(1)

            # description bands
            x,y,w,h = pg.regions["description_col"]
            desc_crop = img.crop((x,y,x+w,y+h))
            sb = split_bands_with_params(desc_crop, float(S.cfg["sepFrac"]), int(S.cfg["minBand"]))
            bands = sb["bands"][:]

            # depth clipping
            if S.cfg["clipDepth"] and pg.top_y is not None and pg.bot_y is not None:
                b2 = []
                for a,b in bands:
                    A = y + a
                    B = y + b
                    AA = max(A, pg.top_y)
                    BB = min(B, pg.bot_y)
                    if (BB - AA) > max(6, int(S.cfg["minBand"]/2)):
                        b2.append((AA - y, BB - y))
                bands = b2

            # prepare text blocks
            blocks = []
            for (y0,y1) in bands:
                tmp = desc_crop.crop((0, y0, desc_crop.width, y1))
                t = ocr_text(tmp, psm=4).replace("|", " ").strip()
                if t and not re.fullmatch(r"Description", t, flags=re.I):
                    blocks.append({"y0_abs": y + y0, "y1_abs": y + y1, "text": t, "nvals": []})

            # N-values
            if "nvalue_col" in pg.regions:
                nx,ny,nw,nh = pg.regions["nvalue_col"]
                n_crop = img.crop((nx,ny,nx+nw,ny+nh))
                nums = ocr_digits_with_pos(n_crop)
                for blk in blocks:
                    vals = []
                    for n in nums:
                        cy = ny + n["y"] + n["h"]/2
                        if blk["y0_abs"] <= cy <= blk["y1_abs"]:
                            vals.append(n["val"])
                    uniq = []
                    for v in vals:
                        if not uniq or uniq[-1] != v:
                            uniq.append(v)
                    blk["nvals"] = uniq

            # Output rows
            for k, b in enumerate(blocks):
                y0a, y1a = b["y0_abs"], b["y1_abs"]
                # Snap first/last bands
                if k == 0 and S.cfg["snapTop"] and pg.top_y is not None:
                    y0a = pg.top_y
                if k == len(blocks) - 1 and S.cfg["snapBot"] and pg.bot_y is not None:
                    y1a = pg.bot_y

                d0 = depth_at_y(y0a, i, pg.top_y, pg.bot_y, float(S.cfg["topFt"]), float(S.cfg["botFt"]))
                d1 = depth_at_y(y1a, i, pg.top_y, pg.bot_y, float(S.cfg["topFt"]), float(S.cfg["botFt"]))
                from_ft = round(d0, 1) if d0 is not None else ""
                to_ft   = round(d1, 1) if d1 is not None else ""

                elev_from = round(float(elev) - from_ft, 1) if elev and from_ft != "" else ""
                elev_to   = round(float(elev) - to_ft, 1) if elev and to_ft != "" else ""

                rows.append([bore or "", from_ft, to_ft,
                             ", ".join(map(str, b["nvals"])) if b["nvals"] else "N/A",
                             b["text"], lon or "", lat or "", elev or "",
                             water or "N/E", elev_from, elev_to])

        if not rows:
            st.warning("No rows were extracted. Ensure regions and depth lines are set.")
        else:
            df = pd.DataFrame(rows, columns=[
                "Bore L.","From (ft)","To (ft)","SPT N-Value","Soil Layer Description",
                "Longitude","Latitude","Top Elevation (ft)","Water Table (ft)","Elevation From (ft)","Elevation To (ft)"
            ])
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Bore Logs")
            st.download_button("Download bore_logs_v5.xlsx", data=buf.getvalue(), file_name="bore_logs_v5.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.success(f"Done. Rows: {len(rows)}")
