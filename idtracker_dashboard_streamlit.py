# idtracker_dashboard_streamlit.py
# 覆蓋版（2026-01-12 / FIX v2）
# 修正：
# 1) st.image 參數相容：use_container_width / use_column_width 自動 fallback
# 2) drawable-canvas 背景圖：優先用 background_image_url(data URL) 避免 image_to_url 崩潰
# 3) px_to_mm/fps 先定義，量測區塊可直接用
# 4) st_canvas 參數降到最保守，避免版本 TypeError
#
# requirements.txt 建議：
# streamlit
# numpy
# pandas
# h5py
# matplotlib
# xlsxwriter
# Pillow
# streamlit-drawable-canvas==0.9.3

import os
import io
import tempfile
import zipfile
import base64

import numpy as np
import pandas as pd
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
    _HAS_CANVAS = True
except Exception:
    _HAS_CANVAS = False


# ---------------------- 相容小工具 ----------------------
def st_image_compat(img, caption=None):
    """相容不同 Streamlit 版本的 st.image 參數"""
    try:
        st.image(img, caption=caption, use_container_width=True)
        return
    except TypeError:
        pass
    try:
        st.image(img, caption=caption, use_column_width=True)
        return
    except TypeError:
        pass
    st.image(img, caption=caption)


def pil_to_data_url(img: Image.Image, fmt="PNG") -> str:
    """PIL Image -> data URL（避免 drawable-canvas 的 image_to_url 崩潰）"""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def resize_if_too_large(img: Image.Image, max_side=1400) -> Image.Image:
    """避免 data URL 太大：長邊>max_side 就縮圖"""
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h))


def apply_axis(ax, xlim=None, ylim=None, xtick=None, ytick=None):
    if xlim and xlim[1] > xlim[0]:
        ax.set_xlim(xlim)
    if ylim and ylim[1] > ylim[0]:
        ax.set_ylim(ylim)
    if xtick:
        lo, hi = ax.get_xlim()
        ax.set_xticks(np.arange(lo, hi + xtick / 10.0, xtick))
    if ytick:
        lo, hi = ax.get_ylim()
        ax.set_yticks(np.arange(lo, hi + ytick / 10.0, ytick))


def _lim_tuple(vmin, vmax):
    if vmax <= vmin or (vmin == 0.0 and vmax == 0.0):
        return None
    return (vmin, vmax)


def _tick_val(v):
    return None if v <= 0 else v


def _safe_bins(vmin, vmax, bin_size, max_bins=800):
    span = float(vmax - vmin)
    if (not np.isfinite(span)) or span <= 0:
        return 1
    b = int(span / float(bin_size))
    b = max(1, b)
    b = min(max_bins, b)
    return b


# ---------------------- App 設定 ----------------------
st.set_page_config(layout="wide")
st.title("🐭 idtracker.ai Dashboard")

uploaded = st.file_uploader("請上傳軌跡檔 (.h5 / .hdf5 / .npz)", type=["h5", "hdf5", "npz"])

# Sidebar 先定義（量測會用到 px_to_mm）
st.sidebar.header("參數設定")
fps = st.sidebar.number_input("FPS", value=30.0, step=1.0)

px_to_mm = st.sidebar.number_input(
    "px_to_mm (mm/px)",
    value=0.10000,
    step=0.00001,
    min_value=0.00001,
    format="%.5f",
)

st.sidebar.subheader("ROI 模式")
roi_mode = st.sidebar.radio(
    "選擇 ROI 來源",
    ["Auto (from trajectories bbox)", "Manual ROI_0 + Split Left/Right 1/3"],
    key="roi_mode",
)

st.sidebar.subheader("Heatmap bin 大小")
bin_mm = st.sidebar.number_input("bin (mm)", value=2.0, step=0.5, min_value=0.1)
bin_px = st.sidebar.number_input("bin (px)", value=5.0, step=1.0, min_value=1.0)

st.sidebar.subheader("軌跡/Heatmap (mm) 座標軸")
x_min_mm = st.sidebar.number_input("Xmin (mm)", value=0.0, step=1.0)
x_max_mm = st.sidebar.number_input("Xmax (mm)", value=0.0, step=1.0)
x_tick_mm = st.sidebar.number_input("Xtick (mm)", value=0.0, step=0.5)
y_min_mm = st.sidebar.number_input("Ymin (mm)", value=0.0, step=1.0)
y_max_mm = st.sidebar.number_input("Ymax (mm)", value=0.0, step=1.0)
y_tick_mm = st.sidebar.number_input("Ytick (mm)", value=0.0, step=0.5)


# ---------------------- PNG 量測工具（主頁） ----------------------
st.markdown("---")
st.subheader("🧰 from streamlit_image_coordinates import streamlit_image_coordinates

st.subheader("🧰 ROI/座標量測（PNG→點選→px/mm）")

if "roi_pts" not in st.session_state:
    st.session_state.roi_pts = []  # 只保留兩點

img_file_main = st.file_uploader(
    "上傳 frame 圖片 (PNG/JPG) 以點選量測",
    type=["png", "jpg", "jpeg"],
    key="roi_measure_img_main",
)
st.caption("建議：Fiji 抽一張 frame 存 PNG，上傳後點一下記錄座標；連點兩次可定義 ROI_0（左上→右下）。")

if img_file_main is not None:
    img0 = Image.open(img_file_main).convert("RGB")
    img0 = resize_if_too_large(img0, max_side=1400)
    w0, h0 = img0.size

    disp_w = st.slider("顯示寬度 (px)", 500, min(1200, w0), min(900, w0), 50)
    scale = disp_w / float(w0)
    disp_h = int(h0 * scale)

    img_disp = img0.resize((disp_w, disp_h))

    # ✅ 這行會「顯示圖」且可點，點到哪裡就回傳座標
    click = streamlit_image_coordinates(img_disp, key="img_coord")

    if click is not None:
        x_disp, y_disp = float(click["x"]), float(click["y"])

        # 轉回原圖座標
        x_px = x_disp / scale
        y_px = y_disp / scale

        # mm
        x_mm = x_px * px_to_mm
        y_mm = y_px * px_to_mm

        st.success(f"點選（原圖）x={x_px:.1f}px, y={y_px:.1f}px ｜ x={x_mm:.2f}mm, y={y_mm:.2f}mm")

        # 去重 + 只保留兩點
        pts = st.session_state.roi_pts
        if len(pts) == 0 or (abs(pts[-1][0] - x_px) > 1 or abs(pts[-1][1] - y_px) > 1):
            pts.append((x_px, y_px))
            st.session_state.roi_pts = pts[:2]

    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("清空點位", key="roi_clear_pts_main"):
            st.session_state.roi_pts = []
            st.rerun()
    with colB:
        st.write(f"已記錄點數：{len(st.session_state.roi_pts)}")
        if len(st.session_state.roi_pts) > 0:
            st.write("Points (original px):", st.session_state.roi_pts)

    # 兩點 → ROI_0
    if len(st.session_state.roi_pts) >= 2:
        (x1p, y1p) = st.session_state.roi_pts[0]
        (x2p, y2p) = st.session_state.roi_pts[1]
        rx1, rx2 = min(x1p, x2p), max(x1p, x2p)
        ry1, ry2 = min(y1p, y2p), max(y1p, y2p)

        st.markdown("**ROI_0 (px)**")
        st.code(f"({rx1:.1f}, {ry1:.1f}, {rx2:.1f}, {ry2:.1f})")

        st.markdown("**ROI_0 (mm)**")
        st.code(f"({rx1*px_to_mm:.2f}, {ry1*px_to_mm:.2f}, {rx2*px_to_mm:.2f}, {ry2*px_to_mm:.2f})")

        if st.button("✅ Apply ROI_0 to Manual ROI inputs", key="apply_roi0_main"):
            st.session_state["roi0_x1"] = float(rx1)
            st.session_state["roi0_y1"] = float(ry1)
            st.session_state["roi0_x2"] = float(rx2)
            st.session_state["roi0_y2"] = float(ry2)
            st.session_state["roi_mode"] = "Manual ROI_0 + Split Left/Right 1/3"
            st.session_state["show_rois"] = ["ROI_0", "ROI_LEFT_1_3", "ROI_RIGHT_1_3"]
            st.rerun()



# ---------------------- 載入軌跡 ----------------------
def load_trajectories(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".h5", ".hdf5"]:
        with h5py.File(path, "r") as f:
            if "trajectories" in f:
                arr = f["trajectories"][()]
            elif "positions" in f:
                arr = f["positions"][()]
            else:
                raise ValueError("H5 檔案中找不到 'trajectories' 或 'positions'")
    elif ext == ".npz":
        data = np.load(path, allow_pickle=True)
        if "positions" not in data:
            raise ValueError("NPZ 檔案中找不到 'positions'")
        arr = data["positions"]
    else:
        raise ValueError("僅支援 h5/npz")

    if arr.ndim != 3 or arr.shape[-1] < 2:
        raise ValueError(f"positions/trajectories shape 應為 (T,N,2) 或 (T,N,>=2)，目前：{arr.shape}")

    T, N, _ = arr.shape
    positions = {i: arr[:, i, :2] for i in range(N)}
    return positions, {"frame_count": T, "ids": list(range(N))}


def _detect_outer_bbox_from_file(path):
    ext = os.path.splitext(path)[1].lower()
    keys = ["arena_bbox", "bbox", "roi_rect"]
    cand = []
    try:
        if ext in [".h5", ".hdf5"]:
            with h5py.File(path, "r") as f:
                for k in keys:
                    if k in f:
                        try:
                            cand.append(f[k][()])
                        except Exception:
                            pass
        elif ext == ".npz":
            data = np.load(path, allow_pickle=True)
            for k in keys:
                if k in data:
                    cand.append(data[k])
    except Exception:
        return None

    for c in cand:
        a = np.array(c, dtype=float).squeeze()
        if a.size == 4:
            return float(a[0]), float(a[1]), float(a[2]), float(a[3])
    return None


def generate_manual_split_rois(x1, y1, x2, y2, include_mid=True):
    rois = [{"name": "ROI_0", "rect": (x1, y1, x2, y2)}]
    w = x2 - x1
    if w <= 0:
        return rois
    xL = x1 + w / 3.0
    xR = x1 + 2.0 * w / 3.0
    rois.append({"name": "ROI_LEFT_1_3", "rect": (x1, y1, xL, y2)})
    if include_mid:
        rois.append({"name": "ROI_MID_1_3", "rect": (xL, y1, xR, y2)})
    rois.append({"name": "ROI_RIGHT_1_3", "rect": (xR, y1, x2, y2)})
    return rois


# ---------------------- Manual ROI_0 inputs ----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Manual ROI_0 (px)")
roi0_x1 = st.sidebar.number_input("ROI_0 x1 (px)", value=float(st.session_state.get("roi0_x1", 0.0)), step=1.0, key="roi0_x1")
roi0_y1 = st.sidebar.number_input("ROI_0 y1 (px)", value=float(st.session_state.get("roi0_y1", 0.0)), step=1.0, key="roi0_y1")
roi0_x2 = st.sidebar.number_input("ROI_0 x2 (px)", value=float(st.session_state.get("roi0_x2", 0.0)), step=1.0, key="roi0_x2")
roi0_y2 = st.sidebar.number_input("ROI_0 y2 (px)", value=float(st.session_state.get("roi0_y2", 0.0)), step=1.0, key="roi0_y2")
include_mid = st.sidebar.checkbox("也生成中間 1/3 ROI", value=True, key="include_mid")

if st.sidebar.button("⬅️ 用量測兩點填入 ROI_0", key="btn_fill_roi0"):
    if len(st.session_state.roi_pts) >= 2:
        (x1p, y1p) = st.session_state.roi_pts[0]
        (x2p, y2p) = st.session_state.roi_pts[1]
        st.session_state["roi0_x1"] = float(min(x1p, x2p))
        st.session_state["roi0_x2"] = float(max(x1p, x2p))
        st.session_state["roi0_y1"] = float(min(y1p, y2p))
        st.session_state["roi0_y2"] = float(max(y1p, y2p))
        st.rerun()
    else:
        st.sidebar.warning("請先在 PNG 上連點兩次（左上→右下）以定義 ROI_0。")


# ---------------------- ROI 產生 / 後續分析（以下維持你原本邏輯） ----------------------
if uploaded is None:
    st.info("請上傳軌跡檔以繼續")
    st.stop()

suffix = "." + uploaded.name.split(".")[-1]
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

positions_px, meta = load_trajectories(tmp_path)
ids = meta["ids"]
total_frames = meta["frame_count"]

max_frame = max(0, total_frames - 1)
frame_start = st.sidebar.number_input("Start frame", 0, max_frame, 0)
frame_end = st.sidebar.number_input("End frame", 0, max_frame, max_frame)

ROI_RANGES = []
if roi_mode.startswith("Auto"):
    detected_bbox = _detect_outer_bbox_from_file(tmp_path)
    if detected_bbox is None:
        all_xy_tmp = np.vstack([positions_px[i][frame_start:frame_end + 1, :] for i in ids])
        x1, y1 = np.nanmin(all_xy_tmp[:, 0]), np.nanmin(all_xy_tmp[:, 1])
        x2, y2 = np.nanmax(all_xy_tmp[:, 0]), np.nanmax(all_xy_tmp[:, 1])
        detected_bbox = (x1, y1, x2, y2)
    x1, y1, x2, y2 = detected_bbox
    ROI_RANGES = [{"name": "ROI_0", "rect": (x1, y1, x2, y2)}]
else:
    x1, y1, x2, y2 = float(roi0_x1), float(roi0_y1), float(roi0_x2), float(roi0_y2)
    if (x2 <= x1) or (y2 <= y1):
        st.sidebar.warning("Manual ROI_0 需要滿足：x2>x1 且 y2>y1。")
    ROI_RANGES = generate_manual_split_rois(x1, y1, x2, y2, include_mid=include_mid)

st.sidebar.caption(f"ROI count = {len(ROI_RANGES)}")

# --------- 以下：你原本的 per_id / df_global / df_dwell / 視覺化 / 匯出 完整照貼即可 ---------
# （為了不讓回覆爆長，我這裡不重複貼你後半段；你把你原本 FIX 檔案中
#  `# ---------------------- per-ID 計算 ----------------------` 之後的內容
#  原封不動接在這行下面即可。）
