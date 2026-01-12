# idtracker_dashboard_streamlit.py
# 功能總覽（2025-09-29 完整版：ROI_0~F、自動外框、軌跡/Heatmap mm+px、速度角速度、匯出工具）
# 2026-01-12 覆蓋版修正：
# 1) Heatmap bins 防呆：避免 int(dx/bin)=0 造成 np.histogram2d ValueError
# 2) Heatmap / bbox 計算忽略 NaN/inf：避免 max/min 爆掉
# 3) Heatmap 資料點不足時改為提示，不直接 crash
# 4) 新增 Two-ROI 模式：可針對「左右兩個 ROI（polygon）」做分析與呈現
#    - Sidebar 可貼 ROI polygon 座標（px）
#    - 輸出 Left/Right dwell、Preference Index
#    - 圖上可顯示 ROI polygon 框線

import os
import io
import json
import tempfile
import zipfile
import numpy as np
import pandas as pd
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path

# ---------------------- 小工具 ----------------------
def fig_to_png_bytes(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

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

def _finite_xy(xy: np.ndarray) -> np.ndarray:
    """Return only finite rows of shape (k,2)."""
    if xy is None or xy.size == 0:
        return xy
    m = np.isfinite(xy).all(axis=1)
    return xy[m]

def _safe_hist2d(x: np.ndarray, y: np.ndarray, bin_size: float):
    """
    Create 2D histogram with safe bins:
    - filters NaN/inf
    - ensures nx, ny >= 1
    Returns: (H, xedges, yedges, nx, ny, n_points)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n_points = int(x.size)

    if n_points < 2:
        return None, None, None, 0, 0, n_points

    dx = float(np.max(x) - np.min(x))
    dy = float(np.max(y) - np.min(y))

    bin_size = float(bin_size) if float(bin_size) > 0 else 1.0
    nx = max(1, int(dx / bin_size))
    ny = max(1, int(dy / bin_size))

    H, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny])
    return H, xedges, yedges, nx, ny, n_points

def _parse_polygon_text(txt: str):
    """
    Accept formats:
    - JSON list: [[x,y],[x,y],...]
    - or python-like list
    Returns np.ndarray shape (K,2)
    """
    if txt is None:
        return None
    s = txt.strip()
    if not s:
        return None
    try:
        poly = json.loads(s)
    except Exception:
        # fallback: try python literal style safely-ish
        try:
            import ast
            poly = ast.literal_eval(s)
        except Exception as e:
            raise ValueError(f"ROI polygon 解析失敗：請用 [[x,y],[x,y],...] 格式。錯誤：{e}")

    arr = np.array(poly, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError("ROI polygon 需要至少 3 個點，格式為 [[x,y],[x,y],...]")
    if not np.isfinite(arr).all():
        raise ValueError("ROI polygon 內含 NaN/inf，請檢查座標。")
    return arr

def in_polygon(xy, poly_xy):
    """Boolean mask for points in polygon (including boundary approximately)."""
    if poly_xy is None or len(poly_xy) < 3:
        return np.zeros((xy.shape[0],), dtype=bool)
    p = Path(poly_xy, closed=True)
    # contains_points 不含邊界；加一點點 radius 容忍
    return p.contains_points(xy, radius=1e-9)

# ---------------------- Streamlit App 設定 ----------------------
st.set_page_config(layout="wide")
st.title("🐭 idtracker.ai Dashboard")

uploaded = st.file_uploader("請上傳軌跡檔 (.h5 / .hdf5 / .npz)", type=["h5", "hdf5", "npz"])

# Sidebar
st.sidebar.header("參數設定")
fps = st.sidebar.number_input("FPS", value=30.0, step=1.0)
px_to_mm = st.sidebar.number_input(
    "px_to_mm (mm/px)", value=0.10000, step=0.00001, min_value=0.00001, format="%.5f"
)

roi_mode = st.sidebar.radio(
    "ROI 模式",
    ["Auto (ROI_0~F 自動切)", "Two-ROI (左右兩個 ROI)"],
    index=1,
)

target_side = st.sidebar.radio("實驗 Target 在哪一側？", ["左", "右"], horizontal=True)

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

# Two-ROI polygon inputs (px)
poly_left_px = None
poly_right_px = None
if roi_mode.startswith("Two-ROI"):
    st.sidebar.subheader("Two-ROI：貼上 ROI polygon 座標 (px)")
    st.sidebar.caption("格式：[[x,y],[x,y],[x,y],[x,y]]（至少 3 點）。你可以直接貼 TRex 裡 ROI 列表的座標。")

    # 依你截圖裡的例子，先放一組預設值（左 ROI x 較小、右 ROI x 較大）
    default_left = "[[274.1,236.2],[504.0,231.8],[508.5,479.3],[267.5,474.9]]"
    default_right = "[[769.3,242.8],[767.1,474.9],[988.2,477.1],[999.2,245.0]]"

    txt_left = st.sidebar.text_area("Left ROI polygon (px)", value=default_left, height=80)
    txt_right = st.sidebar.text_area("Right ROI polygon (px)", value=default_right, height=80)

    try:
        poly_left_px = _parse_polygon_text(txt_left)
        poly_right_px = _parse_polygon_text(txt_right)
    except Exception as e:
        st.sidebar.error(str(e))
        poly_left_px = None
        poly_right_px = None

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
    T, N, _ = arr.shape
    positions = {i: arr[:, i, :] for i in range(N)}
    return positions, {"frame_count": T, "ids": list(range(N))}

def _detect_outer_bbox_from_file(path):
    keys = ["arena_bbox", "bbox", "roi_rect"]
    cand = []
    ext = os.path.splitext(path)[1].lower()
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

    for c in cand:
        a = np.array(c, dtype=float).squeeze()
        if a.size == 4 and np.isfinite(a).all():
            return float(a[0]), float(a[1]), float(a[2]), float(a[3])
    return None

def generate_auto_rois(x1, y1, x2, y2, target_side="左"):
    x3 = (x1 + x2) / 2
    y3 = (y1 + y2) / 2
    rois = [{"name": "ROI_0", "rect": (x1, y1, x2, y2), "type": "rect"}]
    if target_side == "左":
        rois += [
            {"name": "ROI_A", "rect": (x1, y1, x3, y2), "type": "rect"},
            {"name": "ROI_B", "rect": (x3, y1, x2, y2), "type": "rect"},
            {"name": "ROI_E", "rect": (x1, y1, x3, y3), "type": "rect"},
            {"name": "ROI_F", "rect": (x3, y1, x2, y3), "type": "rect"},
        ]
    else:
        rois += [
            {"name": "ROI_A", "rect": (x3, y1, x2, y2), "type": "rect"},
            {"name": "ROI_B", "rect": (x1, y1, x3, y2), "type": "rect"},
            {"name": "ROI_E", "rect": (x3, y1, x2, y3), "type": "rect"},
            {"name": "ROI_F", "rect": (x1, y1, x3, y3), "type": "rect"},
        ]
    rois.append({"name": "ROI_C", "rect": (x1, y1, x2, y3), "type": "rect"})
    rois.append({"name": "ROI_D", "rect": (x1, y3, x2, y2), "type": "rect"})
    return rois

if uploaded is None:
    st.info("請上傳檔案以繼續")
    st.stop()

suffix = "." + uploaded.name.split(".")[-1]
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

positions_px, meta = load_trajectories(tmp_path)
ids = meta["ids"]
total_frames = meta["frame_count"]

detected_bbox = _detect_outer_bbox_from_file(tmp_path)
if detected_bbox is None:
    all_xy_tmp = np.vstack([positions_px[i] for i in ids])
    all_xy_tmp = _finite_xy(all_xy_tmp)
    if all_xy_tmp is None or all_xy_tmp.size == 0:
        raise ValueError("軌跡資料全為 NaN/inf，無法計算自動外框 bbox。")
    x1, y1 = float(np.min(all_xy_tmp[:, 0])), float(np.min(all_xy_tmp[:, 1]))
    x2, y2 = float(np.max(all_xy_tmp[:, 0])), float(np.max(all_xy_tmp[:, 1]))
    detected_bbox = (x1, y1, x2, y2)

x1, y1, x2, y2 = detected_bbox

# ROI definitions
ROI_RANGES = []
if roi_mode.startswith("Auto"):
    ROI_RANGES = generate_auto_rois(x1, y1, x2, y2, target_side)
else:
    # Two ROI (polygon) defined in px; also define an overall bbox ROI_0 (rect) for reference
    ROI_RANGES.append({"name": "ROI_0", "rect": (x1, y1, x2, y2), "type": "rect"})
    if poly_left_px is not None:
        ROI_RANGES.append({"name": "ROI_LEFT", "poly_px": poly_left_px, "type": "poly"})
    if poly_right_px is not None:
        ROI_RANGES.append({"name": "ROI_RIGHT", "poly_px": poly_right_px, "type": "poly"})

# ---------------------- Frame 範圍 ----------------------
max_frame = max(0, total_frames - 1)
frame_start = st.sidebar.number_input("Start frame", 0, max_frame, 0)
frame_end = st.sidebar.number_input("End frame", 0, max_frame, max_frame)

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
    xy_px = positions_px[i][frame_start : frame_end + 1, :]
    xy_mm = xy_px * px_to_mm
    spd = compute_speed_mm_per_s(xy_mm, fps)
    ang = compute_ang_vel_deg_per_s(xy_mm, fps)
    per_id[i] = {"xy_px": xy_px, "xy_mm": xy_mm, "speed": spd, "angvel": ang}

# ---------------------- Global Summary ----------------------
all_dist = []
for data in per_id.values():
    xy = data["xy_mm"]
    if xy.shape[0] >= 2:
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        all_dist.append(np.nansum(steps))

global_distance_mm = float(np.nansum(all_dist))
global_mean_speed = float(np.nanmean(np.concatenate([d["speed"] for d in per_id.values()])))
global_mean_ang = float(np.nanmean(np.concatenate([d["angvel"] for d in per_id.values()])))

df_global = pd.DataFrame(
    [
        {
            "Total_distance_mm": round(global_distance_mm, 3),
            "Mean_speed_mm_s": round(global_mean_speed, 3),
            "Mean_ang_deg_s": round(global_mean_ang, 3),
        }
    ]
)
st.subheader("整體統計")
st.dataframe(df_global, use_container_width=True)

# ---------------------- ROI 統計 ----------------------
def in_rect(xy, rect):
    x1, y1, x2, y2 = rect
    return (xy[:, 0] >= x1) & (xy[:, 0] <= x2) & (xy[:, 1] >= y1) & (xy[:, 1] <= y2)

df_dwell = []
for i, data in per_id.items():
    xy_mm = data["xy_mm"]
    xy_px = data["xy_px"]

    for roi in ROI_RANGES:
        name = roi["name"]

        if roi.get("type") == "rect":
            mask = in_rect(xy_mm, roi["rect"])  # rect ROI is stored in bbox px-space but we apply on mm? -> bbox derived from px; keep consistent by converting bbox to mm
            # NOTE: bbox came from px; but xy_mm is mm. Convert rect px->mm:
            rx1, ry1, rx2, ry2 = roi["rect"]
            rect_mm = (rx1 * px_to_mm, ry1 * px_to_mm, rx2 * px_to_mm, ry2 * px_to_mm)
            mask = in_rect(xy_mm, rect_mm)
        else:
            poly_px = roi["poly_px"]
            mask = in_polygon(xy_px, poly_px)

        frames_in = int(np.count_nonzero(mask))
        mean_spd = float(np.nanmean(np.where(mask, data["speed"], np.nan)))
        mean_ang = float(np.nanmean(np.where(mask, data["angvel"], np.nan)))

        df_dwell.append(
            {
                "ID": i,
                "ROI": name,
                "Frames_in_ROI": frames_in,
                "Time_in_ROI_s": round(frames_in / fps, 3),
                "Mean_speed_mm_s": round(mean_spd, 3),
                "Mean_ang_deg_s": round(mean_ang, 3),
            }
        )

df_dwell = pd.DataFrame(df_dwell)
st.subheader("ROI 統計")
st.dataframe(df_dwell, use_container_width=True)

# ---------------------- Two-ROI Preference Summary ----------------------
df_pref = None
if roi_mode.startswith("Two-ROI") and {"ROI_LEFT", "ROI_RIGHT"}.issubset(set(df_dwell["ROI"].unique())):
    pivot = df_dwell.pivot_table(index="ID", columns="ROI", values="Time_in_ROI_s", aggfunc="sum").fillna(0.0)
    tL = pivot.get("ROI_LEFT", pd.Series(0.0, index=pivot.index))
    tR = pivot.get("ROI_RIGHT", pd.Series(0.0, index=pivot.index))
    denom = (tL + tR).replace(0, np.nan)
    pi = (tL - tR) / denom

    df_pref = pd.DataFrame(
        {
            "ID": pivot.index,
            "Time_Left_s": np.round(tL.values, 3),
            "Time_Right_s": np.round(tR.values, 3),
            "PreferenceIndex_(L-R)/(L+R)": np.round(pi.values, 3),
        }
    )

    st.subheader("Two-ROI 偏好指標 (Preference Index)")
    st.dataframe(df_pref, use_container_width=True)

    # aggregate
    TL = float(np.nansum(tL.values))
    TR = float(np.nansum(tR.values))
    PI_all = (TL - TR) / (TL + TR) if (TL + TR) > 0 else np.nan
    st.caption(f"All IDs total: Left={TL:.3f}s, Right={TR:.3f}s, PI={PI_all:.3f}")

# ---------------------- Sidebar: ROI 顯示選項 ----------------------
roi_names = [r["name"] for r in ROI_RANGES]
show_rois = st.sidebar.multiselect("要在圖上顯示哪些 ROI 框線？", options=roi_names, default=roi_names)

# ---------------------- 視覺化 ----------------------
st.subheader("軌跡圖 (mm)")
fig, ax = plt.subplots(figsize=(6, 6))
for i, data in per_id.items():
    xy = data["xy_mm"]
    ax.plot(xy[:, 0], xy[:, 1], lw=0.7, alpha=0.6, label=f"ID {i}")

# draw ROIs
for roi in ROI_RANGES:
    if roi["name"] not in show_rois:
        continue
    if roi.get("type") == "rect":
        rx1, ry1, rx2, ry2 = roi["rect"]
        rect_mm = (rx1 * px_to_mm, ry1 * px_to_mm, rx2 * px_to_mm, ry2 * px_to_mm)
        x1m, y1m, x2m, y2m = rect_mm
        ax.add_patch(Rectangle((x1m, y1m), x2m - x1m, y2m - y1m, fill=False, lw=1.0, alpha=0.8))
    else:
        poly_px = roi["poly_px"]
        poly_mm = poly_px * px_to_mm
        ax.add_patch(Polygon(poly_mm, closed=True, fill=False, lw=1.2, alpha=0.9))

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
    xy_px = positions_px[i][frame_start : frame_end + 1, :]
    ax.plot(xy_px[:, 0], xy_px[:, 1], lw=0.7, alpha=0.6, label=f"ID {i}")

for roi in ROI_RANGES:
    if roi["name"] not in show_rois:
        continue
    if roi.get("type") == "rect":
        rx1, ry1, rx2, ry2 = roi["rect"]
        ax.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, fill=False, lw=1.0, alpha=0.8))
    else:
        poly_px = roi["poly_px"]
        ax.add_patch(Polygon(poly_px, closed=True, fill=False, lw=1.2, alpha=0.9))

ax.set_xlabel("X (px)")
ax.set_ylabel("Y (px)")
ax.legend(fontsize=6, loc="best")
st.pyplot(fig)

# ---------------------- Heatmap (mm) ----------------------
st.subheader("Heatmap (mm)")
all_xy_mm_plot = np.vstack([d["xy_mm"] for d in per_id.values()])
fig, ax = plt.subplots(figsize=(6, 6))

H, xedges, yedges, nx, ny, npts = _safe_hist2d(all_xy_mm_plot[:, 0], all_xy_mm_plot[:, 1], bin_mm)
if H is None:
    st.warning("Heatmap(mm) 資料點太少或包含無效值（NaN/inf），無法繪製。")
else:
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="hot",
    )
    fig.colorbar(im, ax=ax)

    for roi in ROI_RANGES:
        if roi["name"] not in show_rois:
            continue
        if roi.get("type") == "rect":
            rx1, ry1, rx2, ry2 = roi["rect"]
            rect_mm = (rx1 * px_to_mm, ry1 * px_to_mm, rx2 * px_to_mm, ry2 * px_to_mm)
            x1m, y1m, x2m, y2m = rect_mm
            ax.add_patch(Rectangle((x1m, y1m), x2m - x1m, y2m - y1m, fill=False, lw=1.0, alpha=0.8, color="cyan"))
        else:
            poly_mm = roi["poly_px"] * px_to_mm
            ax.add_patch(Polygon(poly_mm, closed=True, fill=False, lw=1.2, alpha=0.9, color="cyan"))

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    st.caption(f"bins = ({nx}, {ny}), points = {npts}")
    st.pyplot(fig)

# ---------------------- Heatmap (px) ----------------------
st.subheader("Heatmap (px)")
all_xy_px_plot = np.vstack([positions_px[i][frame_start : frame_end + 1, :] for i in ids])
fig, ax = plt.subplots(figsize=(6, 6))

H, xedges, yedges, nx, ny, npts = _safe_hist2d(all_xy_px_plot[:, 0], all_xy_px_plot[:, 1], bin_px)
if H is None:
    st.warning("Heatmap(px) 資料點太少或包含無效值（NaN/inf），無法繪製。")
else:
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="hot",
    )
    fig.colorbar(im, ax=ax)

    for roi in ROI_RANGES:
        if roi["name"] not in show_rois:
            continue
        if roi.get("type") == "rect":
            rx1, ry1, rx2, ry2 = roi["rect"]
            ax.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, fill=False, lw=1.0, alpha=0.8, color="cyan"))
        else:
            ax.add_patch(Polygon(roi["poly_px"], closed=True, fill=False, lw=1.2, alpha=0.9, color="cyan"))

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    st.caption(f"bins = ({nx}, {ny}), points = {npts}")
    st.pyplot(fig)

# ---------------------- 速度與角速度 ----------------------
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

# ROI ranges export：同時支援 rect(mm) 與 poly(px)
rows_roi = []
for r in ROI_RANGES:
    if r.get("type") == "rect":
        rx1, ry1, rx2, ry2 = r["rect"]
        rows_roi.append({"ROI": r["name"], "type": "rect_px", "x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2, "poly_px": ""})
    else:
        poly = r["poly_px"]
        rows_roi.append({"ROI": r["name"], "type": "poly_px", "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan, "poly_px": json.dumps(poly.tolist())})

df_roi_ranges = pd.DataFrame(rows_roi)

df_meta = pd.DataFrame(
    [
        {
            "px_to_mm": px_to_mm,
            "fps": fps,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_count": frame_end - frame_start + 1,
            "roi_mode": roi_mode,
            "target_side": target_side,
        }
    ]
)

if st.button("⬇️ 匯出 Excel"):
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_global.to_excel(writer, sheet_name="Global", index=False)
        df_dwell.to_excel(writer, sheet_name="ROI_Summary", index=False)
        df_roi_ranges.to_excel(writer, sheet_name="ROI_Definitions", index=False)
        df_meta.to_excel(writer, sheet_name="Meta_Info", index=False)
        if df_pref is not None:
            df_pref.to_excel(writer, sheet_name="TwoROI_Preference", index=False)
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
        # Global
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.axis("off")
        tbl = ax.table(cellText=df_global.values, colLabels=df_global.columns, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        pdf.savefig(fig)
        plt.close(fig)

        # ROI Summary
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        tbl = ax.table(cellText=df_dwell.values, colLabels=df_dwell.columns, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6)
        pdf.savefig(fig)
        plt.close(fig)

        # TwoROI preference
        if df_pref is not None:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis("off")
            tbl = ax.table(cellText=df_pref.values, colLabels=df_pref.columns, loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            pdf.savefig(fig)
            plt.close(fig)

        # ROI definitions
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        tbl = ax.table(cellText=df_roi_ranges.values, colLabels=df_roi_ranges.columns, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6)
        pdf.savefig(fig)
        plt.close(fig)

        # Meta Info
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        tbl = ax.table(cellText=df_meta.values, colLabels=df_meta.columns, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        pdf.savefig(fig)
        plt.close(fig)

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
            df_roi_ranges.to_excel(writer, sheet_name="ROI_Definitions", index=False)
            df_meta.to_excel(writer, sheet_name="Meta_Info", index=False)
            if df_pref is not None:
                df_pref.to_excel(writer, sheet_name="TwoROI_Preference", index=False)
        excel_bytes.seek(0)
        zf.writestr("all_results.xlsx", excel_bytes.read())

        # CSV
        zf.writestr("global_summary.csv", df_global.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("roi_summary.csv", df_dwell.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("roi_definitions.csv", df_roi_ranges.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("meta_info.csv", df_meta.to_csv(index=False).encode("utf-8-sig"))
        if df_pref is not None:
            zf.writestr("two_roi_preference.csv", df_pref.to_csv(index=False).encode("utf-8-sig"))

    zip_buf.seek(0)
    st.download_button("下載 ZIP", data=zip_buf, file_name="all_results_bundle.zip", mime="application/zip")
