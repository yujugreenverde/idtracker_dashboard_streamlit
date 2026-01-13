# idtracker_dashboard_streamlit.py
# 覆蓋版（2026-01-13 / FIX v2.4.1）
# 版本變更說明（在 v2.4 基礎上新增）：
# 9) ✅ FirstEntry 表格欄位「前置排序」：一打開就看得到
#    - First_inside_is_focal（你說的「第一次所在區域是否為 focal」；以 first INSIDE 判定）
#    - First_entry_is_focal（第一次進入事件是否為 focal；以 first ENTRY 判定）
#    - Entry_count_focal / Entry_count_other（進入次數；區間第一格在 ROI 內也算一次）
# 10) ✅ 新增欄位：First_inside_is_focal（None/True/False/Tie）
# 11) ✅ 匯出 FirstEntry（Excel/CSV/PDF/ZIP）同步帶出 First_inside_is_focal
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
# streamlit-image-coordinates==0.1.6

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

# Optional: drawable canvas
try:
    from streamlit_drawable_canvas import st_canvas
    _HAS_CANVAS = True
except Exception:
    _HAS_CANVAS = False

# Optional: streamlit-image-coordinates (preferred)
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    _HAS_IMG_COORD = True
except Exception:
    _HAS_IMG_COORD = False


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

# ---- apply pending ROI0 BEFORE widgets are created ----
for k in ["roi0_x1", "roi0_y1", "roi0_x2", "roi0_y2"]:
    pk = f"{k}_pending"
    if pk in st.session_state:
        st.session_state[k] = st.session_state.pop(pk)

# ---- apply pending widget state BEFORE widgets are created ----
if "roi_mode_pending" in st.session_state:
    st.session_state["roi_mode"] = st.session_state.pop("roi_mode_pending")

if "show_rois_pending" in st.session_state:
    st.session_state["show_rois"] = st.session_state.pop("show_rois_pending")

# ✅ px_to_mm pending (avoid StreamlitAPIException)
if "px_to_mm_pending" in st.session_state:
    st.session_state["px_to_mm"] = st.session_state.pop("px_to_mm_pending")

uploaded = st.file_uploader("請上傳軌跡檔 (.h5 / .hdf5 / .npz)", type=["h5", "hdf5", "npz"])


# ---------------------- Sidebar 先定義（量測會用到 px_to_mm） ----------------------
st.sidebar.header("參數設定")

# ✅ experiment metadata inputs
exp_id = st.sidebar.text_input("Experiment ID", value=st.session_state.get("exp_id", ""), key="exp_id")
condition = st.sidebar.radio("Condition", ["Experiment", "Control"], index=0, key="condition")
focal_side = st.sidebar.radio("Focal side", ["Left", "Right"], index=0, key="focal_side")

fps = st.sidebar.number_input("FPS", value=30.0, step=1.0, key="fps")

px_to_mm = st.sidebar.number_input(
    "px_to_mm (mm/px)",
    value=float(st.session_state.get("px_to_mm", 0.10000)),
    step=0.00001,
    min_value=0.00001,
    format="%.5f",
    key="px_to_mm",
)

st.sidebar.subheader("ROI 模式")
roi_mode = st.sidebar.radio(
    "選擇 ROI 來源",
    ["Auto (from trajectories bbox)", "Manual ROI_0 + Split Left/Right 1/3"],
    key="roi_mode",
)

st.sidebar.subheader("Heatmap bin 大小")
bin_mm = st.sidebar.number_input("bin (mm)", value=2.0, step=0.5, min_value=0.1, key="bin_mm")
bin_px = st.sidebar.number_input("bin (px)", value=5.0, step=1.0, min_value=1.0, key="bin_px")

st.sidebar.subheader("軌跡/Heatmap (mm) 座標軸")
x_min_mm = st.sidebar.number_input("Xmin (mm)", value=0.0, step=1.0, key="x_min_mm")
x_max_mm = st.sidebar.number_input("Xmax (mm)", value=0.0, step=1.0, key="x_max_mm")
x_tick_mm = st.sidebar.number_input("Xtick (mm)", value=0.0, step=0.5, key="x_tick_mm")
y_min_mm = st.sidebar.number_input("Ymin (mm)", value=0.0, step=1.0, key="y_min_mm")
y_max_mm = st.sidebar.number_input("Ymax (mm)", value=0.0, step=1.0, key="y_max_mm")
y_tick_mm = st.sidebar.number_input("Ytick (mm)", value=0.0, step=0.5, key="y_tick_mm")


# ---------------------- PNG 量測工具（主頁） ----------------------
st.markdown("---")
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

    disp_w = st.slider("顯示寬度 (px)", 500, min(1400, w0), min(900, w0), 50, key="roi_disp_w")
    scale = disp_w / float(w0)
    disp_h = int(h0 * scale)

    img_disp = img0.resize((disp_w, disp_h))

    # ---- Prefer streamlit-image-coordinates (best alignment) ----
    click = None
    if _HAS_IMG_COORD:
        click = streamlit_image_coordinates(img_disp, key="img_coord")
    else:
        st.warning("找不到 streamlit-image-coordinates，改用 drawable-canvas fallback（若仍遇到底圖不顯示/錯位，請 requirements.txt 加上 streamlit-image-coordinates）。")

        if not _HAS_CANVAS:
            st.error("同時缺少 streamlit-image-coordinates 與 streamlit-drawable-canvas，無法點選量測。")
        else:
            bg_url = pil_to_data_url(img_disp, fmt="PNG")
            try:
                canvas = st_canvas(
                    background_color="white",
                    background_image_url=bg_url,
                    update_streamlit=True,
                    height=disp_h,
                    width=disp_w,
                    drawing_mode="point",
                    key="roi_canvas_fallback",
                )
            except TypeError:
                canvas = st_canvas(
                    background_color="white",
                    background_image=img_disp,
                    update_streamlit=True,
                    height=disp_h,
                    width=disp_w,
                    drawing_mode="point",
                    key="roi_canvas_fallback",
                )

            if canvas is not None and canvas.json_data is not None:
                objs = canvas.json_data.get("objects", [])
                if objs:
                    last = objs[-1]
                    x = last.get("left", last.get("x", None))
                    y = last.get("top", last.get("y", None))
                    if x is not None and y is not None:
                        click = {"x": float(x), "y": float(y)}

    # ---- Handle click ----
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

        w_px = float(rx2 - rx1)
        h_px = float(ry2 - ry1)
        short_px = float(min(w_px, h_px)) if (w_px > 0 and h_px > 0) else np.nan

        st.markdown("**ROI_0 (px)**")
        st.code(f"({rx1:.1f}, {ry1:.1f}, {rx2:.1f}, {ry2:.1f})")

        st.markdown("**ROI_0 (mm)** (using current px_to_mm)")
        st.code(f"({rx1*px_to_mm:.2f}, {ry1*px_to_mm:.2f}, {rx2*px_to_mm:.2f}, {ry2*px_to_mm:.2f})")

        # ✅ px_to_mm calibration from known short-side length (mm)
        st.markdown("### 📏 px_to_mm 校正（用 ROI_0 短邊）")
        colC1, colC2, colC3 = st.columns([1.2, 1.0, 1.2])
        with colC1:
            short_mm = st.number_input(
                "ROI_0 短邊真實長度 (mm)",
                min_value=0.0,
                value=float(st.session_state.get("roi_short_mm", 0.0)),
                step=0.1,
                key="roi_short_mm",
                help="例如：你的矩形短邊實際是 20 mm，就填 20。",
            )
        with colC2:
            st.write("Short side (px)")
            st.code("NaN" if not np.isfinite(short_px) else f"{short_px:.2f}")
        with colC3:
            if st.button("✅ 用短邊(mm)換算 px_to_mm", key="btn_calib_px_to_mm"):
                if (not np.isfinite(short_px)) or short_px <= 0:
                    st.error("ROI_0 短邊(px) 不合法。請先用兩點定義有效矩形 ROI_0。")
                elif short_mm <= 0:
                    st.error("短邊(mm) 必須 > 0。")
                else:
                    new_px_to_mm = float(short_mm) / float(short_px)
                    st.session_state["px_to_mm_pending"] = new_px_to_mm
                    st.success(f"已設定新的 px_to_mm_pending = {new_px_to_mm:.6f} (mm/px)，即將 rerun 套用。")
                    st.rerun()

        # ✅ 不直接改 roi_mode（它被 radio 綁 key="roi_mode"）
        if st.button("✅ Apply ROI_0 to Manual ROI inputs", key="apply_roi0_main"):
            st.session_state["roi0_x1"] = float(rx1)
            st.session_state["roi0_y1"] = float(ry1)
            st.session_state["roi0_x2"] = float(rx2)
            st.session_state["roi0_y2"] = float(ry2)

            st.session_state["roi_mode_pending"] = "Manual ROI_0 + Split Left/Right 1/3"
            st.session_state["show_rois_pending"] = ["ROI_0", "ROI_LEFT_1_3", "ROI_RIGHT_1_3"]
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


# ---------------------- ROI 產生 / 後續分析 ----------------------
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
frame_start = st.sidebar.number_input("Start frame", 0, max_frame, 0, key="frame_start")
frame_end = st.sidebar.number_input("End frame", 0, max_frame, max_frame, key="frame_end")

# ✅ safeguard: 如果在 Manual 模式但 ROI_0 還沒填、且 roi_pts 有兩點，就自動回填
if str(st.session_state.get("roi_mode", "")).startswith("Manual"):
    x1, y1 = float(st.session_state.get("roi0_x1", 0.0)), float(st.session_state.get("roi0_y1", 0.0))
    x2, y2 = float(st.session_state.get("roi0_x2", 0.0)), float(st.session_state.get("roi0_y2", 0.0))
    if (x2 <= x1) or (y2 <= y1):
        pts = st.session_state.get("roi_pts", [])
        if isinstance(pts, list) and len(pts) >= 2:
            (x1p, y1p), (x2p, y2p) = pts[0], pts[1]
            st.session_state["roi0_x1_pending"] = float(min(x1p, x2p))
            st.session_state["roi0_y1_pending"] = float(min(y1p, y2p))
            st.session_state["roi0_x2_pending"] = float(max(x1p, x2p))
            st.session_state["roi0_y2_pending"] = float(max(y1p, y2p))
            st.rerun()


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


# ---------------------- per-ID 計算 ----------------------
def compute_speed_mm_per_s(xy_mm, fps):
    diff = np.diff(xy_mm, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    return np.concatenate([[np.nan], dist * fps])


def compute_ang_vel_deg_per_s(xy_mm, fps, eps=1e-6):
    v = np.diff(xy_mm, axis=0) * fps
    ang_vel = np.full(len(xy_mm), np.nan)
    if len(v) >= 2:
        dot = np.sum(v[1:] * v[:-1], axis=1)
        cross = v[1:, 0] * v[:-1, 1] - v[1:, 1] * v[:-1, 0]
        norm = np.linalg.norm(v[1:], axis=1) * np.linalg.norm(v[:-1], axis=1)
        norm = np.where(norm < eps, eps, norm)
        dtheta = np.arctan2(cross, dot)
        av = dtheta * fps * (180 / np.pi)
        av = (av + 180) % 360 - 180
        ang_vel[2:] = av
    return ang_vel


per_id = {}
for i in ids:
    xy_px = positions_px[i][frame_start: frame_end + 1, :]
    xy_mm = xy_px * px_to_mm
    spd = compute_speed_mm_per_s(xy_mm, fps)
    ang = compute_ang_vel_deg_per_s(xy_mm, fps)
    per_id[i] = {"xy_mm": xy_mm, "speed": spd, "angvel": ang}


# ---------------------- Global Summary ----------------------
all_dist = []
for data in per_id.values():
    xy = data["xy_mm"]
    if xy.shape[0] >= 2:
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        all_dist.append(np.nansum(steps))

global_distance_mm = float(np.nansum(all_dist))
global_mean_speed = float(np.nanmean(np.concatenate([d["speed"] for d in per_id.values()]))) if len(per_id) else np.nan
global_mean_ang = float(np.nanmean(np.concatenate([d["angvel"] for d in per_id.values()]))) if len(per_id) else np.nan

df_global = pd.DataFrame(
    [
        {
            "Experiment_ID": exp_id,
            "Condition": condition,
            "Focal_side": focal_side,
            "Total_distance_mm": round(global_distance_mm, 3),
            "Mean_speed_mm_s": round(global_mean_speed, 3) if np.isfinite(global_mean_speed) else np.nan,
            "Mean_ang_deg_s": round(global_mean_ang, 3) if np.isfinite(global_mean_ang) else np.nan,
        }
    ]
)

st.subheader("整體統計")
st.dataframe(df_global, use_container_width=True)


# ---------------------- ROI 統計 ----------------------
def in_rect(xy_px, rect):
    x1, y1, x2, y2 = rect
    return (xy_px[:, 0] >= x1) & (xy_px[:, 0] <= x2) & (xy_px[:, 1] >= y1) & (xy_px[:, 1] <= y2)


# ---- FirstEntry helper funcs ----
def _finite_xy_mask(xy_px):
    return np.isfinite(xy_px[:, 0]) & np.isfinite(xy_px[:, 1])


def first_inside_frame(mask_bool, frame_start_abs):
    """回傳區間內第一個 inside=True 的 absolute frame；若沒有回傳 None"""
    idx = np.flatnonzero(mask_bool)
    if idx.size == 0:
        return None
    return int(frame_start_abs + idx[0])


def first_entry_frame(mask_bool, frame_start_abs):
    """
    回傳區間內「進入」的第一個 absolute frame。
    定義（依你的要求）：區間第一格就在 ROI 裡 → 算作一次「進入」
    entry：mask[t]=True 且 mask[t-1]=False（t=0 時 prev=False）
    """
    if mask_bool is None or len(mask_bool) == 0:
        return None
    prev = np.concatenate([[False], mask_bool[:-1]])
    entry = mask_bool & (~prev)
    idx = np.flatnonzero(entry)
    if idx.size == 0:
        return None
    return int(frame_start_abs + idx[0])


def entry_count(mask_bool):
    """進入次數（同上定義：t=0 inside=True 也算一次）"""
    if mask_bool is None or len(mask_bool) == 0:
        return 0
    prev = np.concatenate([[False], mask_bool[:-1]])
    entry = mask_bool & (~prev)
    return int(np.count_nonzero(entry))


df_dwell_rows = []
for i in ids:
    xy_px = positions_px[i][frame_start: frame_end + 1, :]
    spd = per_id[i]["speed"]
    ang = per_id[i]["angvel"]

    for roi in ROI_RANGES:
        name = roi["name"]
        mask = in_rect(xy_px, roi["rect"])  # ROI 判定用 px
        frames_in = int(np.count_nonzero(mask))
        mean_spd = float(np.nanmean(np.where(mask, spd, np.nan)))
        mean_ang = float(np.nanmean(np.where(mask, ang, np.nan)))

        df_dwell_rows.append(
            {
                "Experiment_ID": exp_id,
                "Condition": condition,
                "Focal_side": focal_side,
                "ID": i,
                "ROI": name,
                "Frames_in_ROI": frames_in,
                "Time_in_ROI_s": round(frames_in / fps, 3),
                "Mean_speed_mm_s": round(mean_spd, 3) if np.isfinite(mean_spd) else np.nan,
                "Mean_ang_deg_s": round(mean_ang, 3) if np.isfinite(mean_ang) else np.nan,
            }
        )

df_dwell = pd.DataFrame(
    df_dwell_rows,
    columns=[
        "Experiment_ID", "Condition", "Focal_side",
        "ID", "ROI", "Frames_in_ROI", "Time_in_ROI_s", "Mean_speed_mm_s", "Mean_ang_deg_s"
    ],
)

st.subheader("ROI 統計")
st.dataframe(df_dwell, use_container_width=True)


# ---------------------- ROI 顯示選項 ----------------------
roi_names = [r["name"] for r in ROI_RANGES]
default_show = roi_names if len(roi_names) > 0 else []
show_rois = st.sidebar.multiselect(
    "要在圖上顯示哪些 ROI？",
    options=roi_names,
    default=st.session_state.get("show_rois", default_show),
    key="show_rois",
)


# ---------------------- 左右 ROI 分析（只在 Manual Split 模式顯示） ----------------------
df_pref = pd.DataFrame()
df_first_entry = pd.DataFrame()
if roi_mode.startswith("Manual"):
    st.subheader("左右 ROI 分析（Left/Right 1/3）")
    if df_dwell.empty:
        st.warning("df_dwell 目前為空（可能 ROI_0 尚未有效、或 frame/資料不足），無法計算左右偏好。")
    else:
        roi_set = set(df_dwell["ROI"].dropna().unique().tolist())
        left_name = "ROI_LEFT_1_3" if "ROI_LEFT_1_3" in roi_set else None
        right_name = "ROI_RIGHT_1_3" if "ROI_RIGHT_1_3" in roi_set else None

        if left_name is None or right_name is None:
            st.warning("找不到 ROI_LEFT_1_3 / ROI_RIGHT_1_3，請確認 ROI_0 有效且已產生分割 ROI。")
        else:
            pivot = df_dwell.pivot_table(index="ID", columns="ROI", values="Time_in_ROI_s", aggfunc="sum").fillna(0.0)
            tL = pivot.get(left_name, pd.Series(0.0, index=pivot.index))
            tR = pivot.get(right_name, pd.Series(0.0, index=pivot.index))
            denom = (tL + tR).replace(0, np.nan)

            # ✅ raw PI: (L-R)/(L+R)
            pi_raw = (tL - tR) / denom

            # ✅ focal-aligned PI: focal side as positive
            pi_focal = pi_raw if focal_side == "Left" else -pi_raw

            df_pref = pd.DataFrame(
                {
                    "Experiment_ID": exp_id,
                    "Condition": condition,
                    "Focal_side": focal_side,
                    "ID": pivot.index,
                    "Time_Left_s": np.round(tL.values, 3),
                    "Time_Right_s": np.round(tR.values, 3),
                    "PI_raw_(L-R)/(L+R)": np.round(pi_raw.values, 3),
                    "PI_focal_(focal-positive)": np.round(pi_focal.values, 3),
                }
            )
            st.dataframe(df_pref, use_container_width=True)

            TL = float(np.nansum(tL.values))
            TR = float(np.nansum(tR.values))
            PI_all_raw = (TL - TR) / (TL + TR) if (TL + TR) > 0 else np.nan
            PI_all_focal = PI_all_raw if focal_side == "Left" else (-PI_all_raw if np.isfinite(PI_all_raw) else np.nan)
            st.caption(
                f"All IDs total: Left={TL:.3f}s, Right={TR:.3f}s, "
                f"PI_raw={PI_all_raw:.3f} | PI_focal={PI_all_focal:.3f} (focal={focal_side})"
            )

            # ---------------------- First entry + Entry counts (v2.4.1) ----------------------
            st.markdown("### ⏱️ First entry（指定 frame 區間內：第一次所在區域 / 第一次進入 / 進入次數）")

            focal_roi_name = left_name if focal_side == "Left" else right_name
            other_roi_name = right_name if focal_side == "Left" else left_name

            rect_map = {r["name"]: r["rect"] for r in ROI_RANGES}
            focal_rect = rect_map.get(focal_roi_name, None)
            other_rect = rect_map.get(other_roi_name, None)

            if focal_rect is None or other_rect is None:
                st.warning("找不到 focal/other 的 ROI rect，無法計算 first entry。")
            else:
                rows = []
                for i in ids:
                    xy_px = positions_px[i][frame_start: frame_end + 1, :]
                    fin = _finite_xy_mask(xy_px)

                    mask_focal = np.zeros(xy_px.shape[0], dtype=bool)
                    mask_other = np.zeros(xy_px.shape[0], dtype=bool)
                    if np.any(fin):
                        mask_focal[fin] = in_rect(xy_px[fin], focal_rect)
                        mask_other[fin] = in_rect(xy_px[fin], other_rect)

                    fi_focal_inside = first_inside_frame(mask_focal, frame_start)
                    fi_other_inside = first_inside_frame(mask_other, frame_start)

                    fe_focal = first_entry_frame(mask_focal, frame_start)
                    fe_other = first_entry_frame(mask_other, frame_start)

                    cnt_focal = entry_count(mask_focal)
                    cnt_other = entry_count(mask_other)

                    # ---- 判定：first INSIDE 是否為 focal（你說的「第一次所在區域」） ----
                    if fi_focal_inside is None and fi_other_inside is None:
                        inside_is_focal = None
                    elif fi_focal_inside is not None and fi_other_inside is None:
                        inside_is_focal = True
                    elif fi_focal_inside is None and fi_other_inside is not None:
                        inside_is_focal = False
                    else:
                        if fi_focal_inside < fi_other_inside:
                            inside_is_focal = True
                        elif fi_other_inside < fi_focal_inside:
                            inside_is_focal = False
                        else:
                            inside_is_focal = "Tie"

                    # ---- 判定：first ENTRY 是否為 focal（第一次「進入事件」） ----
                    if fe_focal is None and fe_other is None:
                        entry_is_focal = None
                    elif fe_focal is not None and fe_other is None:
                        entry_is_focal = True
                    elif fe_focal is None and fe_other is not None:
                        entry_is_focal = False
                    else:
                        if fe_focal < fe_other:
                            entry_is_focal = True
                        elif fe_other < fe_focal:
                            entry_is_focal = False
                        else:
                            entry_is_focal = "Tie"

                    rows.append(
                        {
                            "Experiment_ID": exp_id,
                            "Condition": condition,
                            "Focal_side": focal_side,
                            "ID": i,
                            "Focal_ROI": focal_roi_name,
                            "Other_ROI": other_roi_name,
                            "First_inside_focal_frame": fi_focal_inside,
                            "First_inside_other_frame": fi_other_inside,
                            "First_inside_is_focal": inside_is_focal,  # ✅ v2.4.1
                            "First_entry_focal_frame": fe_focal,
                            "First_entry_other_frame": fe_other,
                            "First_entry_is_focal": entry_is_focal,
                            "Entry_count_focal": cnt_focal,
                            "Entry_count_other": cnt_other,
                            "First_entry_focal_time_s": (None if fe_focal is None else round((fe_focal - frame_start) / fps, 3)),
                            "First_entry_other_time_s": (None if fe_other is None else round((fe_other - frame_start) / fps, 3)),
                        }
                    )

                df_first_entry = pd.DataFrame(rows)

                # ✅ 前置排序：一打開就看到你要的欄位
                front_cols = [
                    "ID",
                    "First_inside_is_focal",
                    "First_entry_is_focal",
                    "Entry_count_focal",
                    "Entry_count_other",
                    "First_inside_focal_frame",
                    "First_inside_other_frame",
                    "First_entry_focal_frame",
                    "First_entry_other_frame",
                    "First_entry_focal_time_s",
                    "First_entry_other_time_s",
                ]
                front_cols = [c for c in front_cols if c in df_first_entry.columns]
                df_first_entry = df_first_entry[front_cols + [c for c in df_first_entry.columns if c not in front_cols]]

                st.dataframe(df_first_entry, use_container_width=True)

                vc_inside = df_first_entry["First_inside_is_focal"].value_counts(dropna=False).to_dict()
                vc_entry = df_first_entry["First_entry_is_focal"].value_counts(dropna=False).to_dict()
                total_focal_entries = int(np.nansum(df_first_entry["Entry_count_focal"].values))
                total_other_entries = int(np.nansum(df_first_entry["Entry_count_other"].values))

                st.caption(f"First_inside_is_focal counts: {vc_inside}")
                st.caption(f"First_entry_is_focal counts: {vc_entry}")
                st.caption(f"Total entry counts (all IDs): focal={total_focal_entries}, other={total_other_entries}")


# ---------------------- 視覺化 ----------------------
st.subheader("軌跡圖 (mm)")
fig, ax = plt.subplots(figsize=(6, 6))
for i, data in per_id.items():
    xy = data["xy_mm"]
    ax.plot(xy[:, 0], xy[:, 1], lw=0.7, alpha=0.6, label=f"ID {i}")

for roi in ROI_RANGES:
    if roi["name"] in show_rois:
        x1p, y1p, x2p, y2p = roi["rect"]
        x1m, y1m, x2m, y2m = x1p * px_to_mm, y1p * px_to_mm, x2p * px_to_mm, y2p * px_to_mm
        rect = Rectangle((x1m, y1m), x2m - x1m, y2m - y1m, fill=False, lw=1.2, alpha=0.9)
        ax.add_patch(rect)

apply_axis(
    ax,
    xlim=_lim_tuple(x_min_mm, x_max_mm),
    ylim=_lim_tuple(y_min_mm, y_max_mm),
    xtick=_tick_val(x_tick_mm),
    ytick=_tick_val(y_tick_mm),
)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.legend(fontsize=6, loc="best")
st.pyplot(fig)

st.subheader("軌跡圖 (px)")
fig, ax = plt.subplots(figsize=(6, 6))
for i in ids:
    xy_px = positions_px[i][frame_start: frame_end + 1, :]
    ax.plot(xy_px[:, 0], xy_px[:, 1], lw=0.7, alpha=0.6, label=f"ID {i}")

for roi in ROI_RANGES:
    if roi["name"] in show_rois:
        x1p, y1p, x2p, y2p = roi["rect"]
        rect = Rectangle((x1p, y1p), x2p - x1p, y2p - y1p, fill=False, lw=1.2, alpha=0.9)
        ax.add_patch(rect)

ax.set_xlabel("X (px)")
ax.set_ylabel("Y (px)")
ax.legend(fontsize=6, loc="best")
st.pyplot(fig)

# ✅ Heatmap(mm)
st.subheader("Heatmap (mm)")
all_xy_mm_plot = np.vstack([d["xy_mm"] for d in per_id.values()])
finite_mask = np.isfinite(all_xy_mm_plot[:, 0]) & np.isfinite(all_xy_mm_plot[:, 1])
xy_finite = all_xy_mm_plot[finite_mask]

if xy_finite.shape[0] < 2:
    st.warning("Heatmap(mm) 無有效資料點（可能 frame/資料不足或全 NaN），略過。")
else:
    bx = _safe_bins(np.min(xy_finite[:, 0]), np.max(xy_finite[:, 0]), bin_mm)
    by = _safe_bins(np.min(xy_finite[:, 1]), np.max(xy_finite[:, 1]), bin_mm)

    H, xedges, yedges = np.histogram2d(xy_finite[:, 0], xy_finite[:, 1], bins=[bx, by])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="hot",
    )
    fig.colorbar(im, ax=ax)

    for roi in ROI_RANGES:
        if roi["name"] in show_rois:
            x1p, y1p, x2p, y2p = roi["rect"]
            x1m, y1m, x2m, y2m = x1p * px_to_mm, y1p * px_to_mm, x2p * px_to_mm, y2p * px_to_mm
            rect = Rectangle((x1m, y1m), x2m - x1m, y2m - y1m, fill=False, lw=1.2, alpha=0.9)
            ax.add_patch(rect)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    st.pyplot(fig)

# ✅ Heatmap(px)
st.subheader("Heatmap (px)")
all_xy_px_plot = np.vstack([positions_px[i][frame_start: frame_end + 1, :] for i in ids])
finite_mask = np.isfinite(all_xy_px_plot[:, 0]) & np.isfinite(all_xy_px_plot[:, 1])
xy_finite = all_xy_px_plot[finite_mask]

if xy_finite.shape[0] < 2:
    st.warning("Heatmap(px) 無有效資料點（可能 frame/資料不足或全 NaN），略過。")
else:
    bx = _safe_bins(np.min(xy_finite[:, 0]), np.max(xy_finite[:, 0]), bin_px)
    by = _safe_bins(np.min(xy_finite[:, 1]), np.max(xy_finite[:, 1]), bin_px)

    H, xedges, yedges = np.histogram2d(xy_finite[:, 0], xy_finite[:, 1], bins=[bx, by])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="hot",
    )
    fig.colorbar(im, ax=ax)

    for roi in ROI_RANGES:
        if roi["name"] in show_rois:
            x1p, y1p, x2p, y2p = roi["rect"]
            rect = Rectangle((x1p, y1p), x2p - x1p, y2p - y1p, fill=False, lw=1.2, alpha=0.9)
            ax.add_patch(rect)

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    st.pyplot(fig)

# 速度與角速度
st.subheader("速度與角速度曲線")
for i, data in per_id.items():
    fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    ax[0].plot(data["speed"], lw=0.8)
    ax[0].set_ylabel("Speed (mm/s)")
    ax[1].plot(data["angvel"], lw=0.8)
    ax[1].set_ylabel("Ang vel (deg/s)")
    ax[1].set_xlabel("Frame")
    fig.suptitle(f"ID {i}")
    st.pyplot(fig)

# ---------------------- 匯出 Excel/PDF/ZIP ----------------------
st.subheader("匯出結果")

df_roi_ranges = pd.DataFrame(
    [
        {
            "Experiment_ID": exp_id,
            "Condition": condition,
            "Focal_side": focal_side,
            "ROI": r["name"],
            "x1_px": r["rect"][0],
            "y1_px": r["rect"][1],
            "x2_px": r["rect"][2],
            "y2_px": r["rect"][3],
        }
        for r in ROI_RANGES
    ]
)

df_meta = pd.DataFrame(
    [
        {
            "Experiment_ID": exp_id,
            "Condition": condition,
            "Focal_side": focal_side,
            "px_to_mm": px_to_mm,
            "fps": fps,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_count": frame_end - frame_start + 1,
            "roi_mode": roi_mode,
        }
    ]
)

# Ensure df_pref exists even in Auto mode
if df_pref is None or not isinstance(df_pref, pd.DataFrame) or df_pref.empty:
    df_pref_export = pd.DataFrame(
        columns=[
            "Experiment_ID", "Condition", "Focal_side", "ID",
            "Time_Left_s", "Time_Right_s", "PI_raw_(L-R)/(L+R)", "PI_focal_(focal-positive)"
        ]
    )
else:
    df_pref_export = df_pref.copy()

# Ensure df_first_entry exists even in Auto mode
if df_first_entry is None or not isinstance(df_first_entry, pd.DataFrame) or df_first_entry.empty:
    df_first_entry_export = pd.DataFrame(
        columns=[
            "Experiment_ID", "Condition", "Focal_side", "ID", "Focal_ROI", "Other_ROI",
            "First_inside_focal_frame", "First_inside_other_frame", "First_inside_is_focal",
            "First_entry_focal_frame", "First_entry_other_frame", "First_entry_is_focal",
            "Entry_count_focal", "Entry_count_other",
            "First_entry_focal_time_s", "First_entry_other_time_s",
        ]
    )
else:
    df_first_entry_export = df_first_entry.copy()

if st.button("⬇️ 匯出 Excel"):
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_global.to_excel(writer, sheet_name="Global", index=False)
        df_dwell.to_excel(writer, sheet_name="ROI_Summary", index=False)
        df_pref_export.to_excel(writer, sheet_name="PreferenceIndex", index=False)
        df_first_entry_export.to_excel(writer, sheet_name="FirstEntry", index=False)
        df_roi_ranges.to_excel(writer, sheet_name="ROI_Ranges", index=False)
        df_meta.to_excel(writer, sheet_name="Meta_Info", index=False)
    excel_buf.seek(0)
    st.download_button(
        "下載 Excel",
        data=excel_buf,
        file_name="all_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if st.button("⬇️ 匯出 PDF"):
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        def _pdf_table(df, title, fontsize=7, fig_w=8, fig_h=3):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.axis("off")
            ax.set_title(title)
            tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(fontsize)
            pdf.savefig(fig)
            plt.close(fig)

        _pdf_table(df_global, "Global", fontsize=8, fig_w=8, fig_h=2)
        _pdf_table(df_dwell, "ROI_Summary", fontsize=6, fig_w=10, fig_h=4)
        _pdf_table(df_pref_export, "PreferenceIndex", fontsize=7, fig_w=10, fig_h=3)
        _pdf_table(df_first_entry_export, "FirstEntry", fontsize=7, fig_w=10, fig_h=3)
        _pdf_table(df_roi_ranges, "ROI_Ranges", fontsize=7, fig_w=10, fig_h=3)
        _pdf_table(df_meta, "Meta_Info", fontsize=8, fig_w=8, fig_h=2)

    pdf_buf.seek(0)
    st.download_button("下載 PDF", data=pdf_buf, file_name="all_results.pdf", mime="application/pdf")

if st.button("⬇️ 匯出 ZIP"):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        # Excel
        excel_bytes = io.BytesIO()
        with pd.ExcelWriter(excel_bytes, engine="xlsxwriter") as writer:
            df_global.to_excel(writer, sheet_name="Global", index=False)
            df_dwell.to_excel(writer, sheet_name="ROI_Summary", index=False)
            df_pref_export.to_excel(writer, sheet_name="PreferenceIndex", index=False)
            df_first_entry_export.to_excel(writer, sheet_name="FirstEntry", index=False)
            df_roi_ranges.to_excel(writer, sheet_name="ROI_Ranges", index=False)
            df_meta.to_excel(writer, sheet_name="Meta_Info", index=False)
        excel_bytes.seek(0)
        zf.writestr("all_results.xlsx", excel_bytes.read())

        # CSV
        zf.writestr("global_summary.csv", df_global.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("roi_summary.csv", df_dwell.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("preference_index.csv", df_pref_export.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("first_entry.csv", df_first_entry_export.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("roi_ranges.csv", df_roi_ranges.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("meta_info.csv", df_meta.to_csv(index=False).encode("utf-8-sig"))

    zip_buf.seek(0)
    st.download_button("下載 ZIP", data=zip_buf, file_name="all_results_bundle.zip", mime="application/zip")
