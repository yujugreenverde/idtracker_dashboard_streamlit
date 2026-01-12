# idtracker_dashboard_streamlit.py
# 2026-01-12 覆蓋版（新增：手動大矩形 ROI_0 → 左右 1/3 分割 Two-ROI 分析 + heatmap 防呆）

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
    if xy is None or xy.size == 0:
        return xy
    m = np.isfinite(xy).all(axis=1)
    return xy[m]

def _safe_hist2d(x: np.ndarray, y: np.ndarray, bin_size: float):
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
    if txt is None:
        return None
    s = txt.strip()
    if not s:
        return None
    try:
        poly = json.loads(s)
    except Exception:
        import ast
        poly = ast.literal_eval(s)
    arr = np.array(poly, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError("ROI polygon 需要至少 3 點，格式 [[x,y],[x,y],...]")
    if not np.isfinite(arr).all():
        raise ValueError("ROI polygon 內含 NaN/inf")
    return arr

def in_polygon(xy, poly_xy):
    if poly_xy is None or len(poly_xy) < 3:
        return np.zeros((xy.shape[0],), dtype=bool)
    p = Path(poly_xy, closed=True)
    return p.contains_points(xy, radius=1e-9)

def in_rect_xy(xy, rect):
    x1, y1, x2, y2 = rect
    return (xy[:, 0] >= x1) & (xy[:, 0] <= x2) & (xy[:, 1] >= y1) & (xy[:, 1] <= y2)

# ---------------------- Streamlit App 設定 ----------------------
st.set_page_config(layout="wide")
st.title("🐭 idtracker.ai Dashboard")

uploaded = st.file_uploader("請上傳軌跡檔 (.h5 / .hdf5 / .npz)", type=["h5", "hdf5", "npz"])

st.sidebar.header("參數設定")
fps = st.sidebar.number_input("FPS", value=30.0, step=1.0)
px_to_mm = st.sidebar.number_input("px_to_mm (mm/px)", value=0.10000, step=0.00001, min_value=0.00001, format="%.5f")

roi_mode = st.sidebar.radio("ROI 模式", ["Auto (ROI_0~F)", "Two-ROI (左右偏好)"], index=1)
two_roi_method = None

if roi_mode.startswith("Two-ROI"):
    two_roi_method = st.sidebar.radio(
        "Two-ROI 取得方式",
        ["Rect → Left 1/3 & Right 1/3 (推薦)", "Paste polygons (px)"],
        index=0
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

# Two-ROI inputs
poly_left_px = None
poly_right_px = None
manual_roi0_rect_px = None  # (x1,y1,x2,y2)

if roi_mode.startswith("Two-ROI"):
    if two_roi_method.startswith("Rect"):
        st.sidebar.subheader("輸入大矩形 ROI_0 (px)")
        st.sidebar.caption("請用你畫面上的 arena 外框：x1,y1,x2,y2（左上/右下）")
        rx1 = st.sidebar.number_input("ROI_0 x1 (px)", value=0.0, step=1.0)
        ry1 = st.sidebar.number_input("ROI_0 y1 (px)", value=0.0, step=1.0)
        rx2 = st.sidebar.number_input("ROI_0 x2 (px)", value=0.0, step=1.0)
        ry2 = st.sidebar.number_input("ROI_0 y2 (px)", value=0.0, step=1.0)

        if rx2 > rx1 and ry2 > ry1:
            manual_roi0_rect_px = (float(rx1), float(ry1), float(rx2), float(ry2))
        else:
            st.sidebar.warning("ROI_0 需要 x2>x1 且 y2>y1，否則不會啟用分割。")

        st.sidebar.subheader("分割設定")
        inner_margin = st.sidebar.number_input("左右 ROI 內縮 margin (px)", value=0.0, step=1.0, min_value=0.0)
        use_middle = st.sidebar.checkbox("也計算中間 1/3 (ROI_MID_1_3)", value=False)
    else:
        st.sidebar.subheader("貼上左右 ROI polygon (px)")
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

df_global = pd.DataFrame([{
    "Total_distance_mm": round(float(np.nansum(all_dist)), 3),
    "Mean_speed_mm_s": round(float(np.nanmean(np.concatenate([d["speed"] for d in per_id.values()]))), 3),
    "Mean_ang_deg_s": round(float(np.nanmean(np.concatenate([d["angvel"] for d in per_id.values()]))), 3),
}])
st.subheader("整體統計")
st.dataframe(df_global, use_container_width=True)

# ---------------------- ROI 定義 ----------------------
ROI_RANGES = []

if roi_mode.startswith("Auto"):
    # 用所有軌跡的 min/max 當 bbox
    all_xy_tmp = np.vstack([positions_px[i] for i in ids])
    all_xy_tmp = _finite_xy(all_xy_tmp)
    if all_xy_tmp is None or all_xy_tmp.size == 0:
        raise ValueError("軌跡資料全為 NaN/inf，無法計算 bbox。")
    x1, y1 = float(np.min(all_xy_tmp[:, 0])), float(np.min(all_xy_tmp[:, 1]))
    x2, y2 = float(np.max(all_xy_tmp[:, 0])), float(np.max(all_xy_tmp[:, 1]))
    ROI_RANGES.append({"name": "ROI_0", "type": "rect_px", "rect_px": (x1, y1, x2, y2)})
else:
    # Two ROI
    if two_roi_method.startswith("Rect"):
        if manual_roi0_rect_px is not None:
            x1, y1, x2, y2 = manual_roi0_rect_px
            ROI_RANGES.append({"name": "ROI_0", "type": "rect_px", "rect_px": (x1, y1, x2, y2)})

            W = x2 - x1
            # thirds split
            a = x1 + W / 3.0
            b = x1 + 2.0 * W / 3.0

            m = float(inner_margin)
            # Left 1/3
            ROI_RANGES.append({"name": "ROI_LEFT_1_3", "type": "rect_px", "rect_px": (x1 + m, y1 + m, a - m, y2 - m)})
            # Middle 1/3 (optional)
            if use_middle:
                ROI_RANGES.append({"name": "ROI_MID_1_3", "type": "rect_px", "rect_px": (a + m, y1 + m, b - m, y2 - m)})
            # Right 1/3
            ROI_RANGES.append({"name": "ROI_RIGHT_1_3", "type": "rect_px", "rect_px": (b + m, y1 + m, x2 - m, y2 - m)})
        else:
            st.warning("Two-ROI Rect 模式：請先在 sidebar 輸入 ROI_0 x1,y1,x2,y2。")
    else:
        ROI_RANGES.append({"name": "ROI_LEFT", "type": "poly_px", "poly_px": poly_left_px})
        ROI_RANGES.append({"name": "ROI_RIGHT", "type": "poly_px", "poly_px": poly_right_px})

# ---------------------- ROI 統計 ----------------------
df_dwell = []
for i, data in per_id.items():
    xy_px = data["xy_px"]

    for roi in ROI_RANGES:
        name = roi["name"]
        if roi["type"] == "rect_px":
            rect = roi["rect_px"]
            mask = in_rect_xy(xy_px, rect)
        else:
            poly = roi["poly_px"]
            if poly is None:
                continue
            mask = in_polygon(xy_px, poly)

        frames_in = int(np.count_nonzero(mask))
        mean_spd = float(np.nanmean(np.where(mask, data["speed"], np.nan)))
        mean_ang = float(np.nanmean(np.where(mask, data["angvel"], np.nan)))

        df_dwell.append({
            "ID": i,
            "ROI": name,
            "Frames_in_ROI": frames_in,
            "Time_in_ROI_s": round(frames_in / fps, 3),
            "Mean_speed_mm_s": round(mean_spd, 3),
            "Mean_ang_deg_s": round(mean_ang, 3),
        })

df_dwell = pd.DataFrame(df_dwell)
st.subheader("ROI 統計")
st.dataframe(df_dwell, use_container_width=True)

# ---------------------- Two-ROI Preference ----------------------
df_pref = None
if roi_mode.startswith("Two-ROI"):
    # 防呆：df_dwell 必須存在且包含 ROI 欄位
    if (df_dwell is None) or (df_dwell.empty) or ("ROI" not in df_dwell.columns):
        st.warning("尚未產生 ROI 統計（可能 ROI_0 尚未有效輸入、或 frame/軌跡資料不足），因此略過 Preference Index 計算。")
    else:
        roi_set = set(df_dwell["ROI"].dropna().unique().tolist())

        # 支援兩種 naming
        left_name = "ROI_LEFT_1_3" if "ROI_LEFT_1_3" in roi_set else ("ROI_LEFT" if "ROI_LEFT" in roi_set else None)
        right_name = "ROI_RIGHT_1_3" if "ROI_RIGHT_1_3" in roi_set else ("ROI_RIGHT" if "ROI_RIGHT" in roi_set else None)

        if left_name is None or right_name is None:
            st.warning("df_dwell 中找不到左右 ROI（ROI_LEFT_1_3 / ROI_RIGHT_1_3 或 ROI_LEFT / ROI_RIGHT），略過 Preference Index。")
        else:
            pivot = (
                df_dwell.pivot_table(index="ID", columns="ROI", values="Time_in_ROI_s", aggfunc="sum")
                .fillna(0.0)
            )
            tL = pivot.get(left_name, pd.Series(0.0, index=pivot.index))
            tR = pivot.get(right_name, pd.Series(0.0, index=pivot.index))

            denom = (tL + tR).replace(0, np.nan)
            pi = (tL - tR) / denom

            df_pref = pd.DataFrame({
                "ID": pivot.index,
                "Time_Left_s": np.round(tL.values, 3),
                "Time_Right_s": np.round(tR.values, 3),
                "PreferenceIndex_(L-R)/(L+R)": np.round(pi.values, 3),
            })

            st.subheader("左右偏好指標 (Preference Index)")
            st.dataframe(df_pref, use_container_width=True)

            TL = float(np.nansum(tL.values))
            TR = float(np.nansum(tR.values))
            PI_all = (TL - TR) / (TL + TR) if (TL + TR) > 0 else np.nan
            st.caption(f"All IDs total: Left={TL:.3f}s, Right={TR:.3f}s, PI={PI_all:.3f}")

# ---------------------- ROI 顯示選項 ----------------------
roi_names = [r["name"] for r in ROI_RANGES]
st.sidebar.caption(f"ROI count = {len(ROI_RANGES)}")

default_show = roi_names if len(roi_names) > 0 else []
show_rois = st.sidebar.multiselect(
    "要在圖上顯示哪些 ROI？",
    options=roi_names,
    default=default_show
)

# ---------------------- 視覺化：軌跡 (mm) ----------------------
st.subheader("軌跡圖 (mm)")
fig, ax = plt.subplots(figsize=(6, 6))
for i, data in per_id.items():
    xy = data["xy_mm"]
    ax.plot(xy[:, 0], xy[:, 1], lw=0.7, alpha=0.6, label=f"ID {i}")

for roi in ROI_RANGES:
    if roi["name"] not in show_rois:
        continue
    if roi["type"] == "rect_px":
        x1, y1, x2, y2 = roi["rect_px"]
        x1m, y1m, x2m, y2m = x1 * px_to_mm, y1 * px_to_mm, x2 * px_to_mm, y2 * px_to_mm
        ax.add_patch(Rectangle((x1m, y1m), x2m - x1m, y2m - y1m, fill=False, lw=1.0, alpha=0.9))
    else:
        poly = roi["poly_px"]
        if poly is None:
            continue
        poly_mm = poly * px_to_mm
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

# ---------------------- 視覺化：軌跡 (px) ----------------------
st.subheader("軌跡圖 (px)")
fig, ax = plt.subplots(figsize=(6, 6))
for i in ids:
    xy_px = positions_px[i][frame_start : frame_end + 1, :]
    ax.plot(xy_px[:, 0], xy_px[:, 1], lw=0.7, alpha=0.6, label=f"ID {i}")

for roi in ROI_RANGES:
    if roi["name"] not in show_rois:
        continue
    if roi["type"] == "rect_px":
        x1, y1, x2, y2 = roi["rect_px"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0, alpha=0.9))
    else:
        poly = roi["poly_px"]
        if poly is None:
            continue
        ax.add_patch(Polygon(poly, closed=True, fill=False, lw=1.2, alpha=0.9))

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
    st.warning("Heatmap(mm) 資料點太少或包含 NaN/inf，無法繪製。")
else:
    im = ax.imshow(H.T, origin="lower",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   aspect="auto", cmap="hot")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    st.caption(f"bins = ({nx},{ny}), points={npts}")
    st.pyplot(fig)

# ---------------------- Heatmap (px) ----------------------
st.subheader("Heatmap (px)")
all_xy_px_plot = np.vstack([positions_px[i][frame_start : frame_end + 1, :] for i in ids])
fig, ax = plt.subplots(figsize=(6, 6))
H, xedges, yedges, nx, ny, npts = _safe_hist2d(all_xy_px_plot[:, 0], all_xy_px_plot[:, 1], bin_px)
if H is None:
    st.warning("Heatmap(px) 資料點太少或包含 NaN/inf，無法繪製。")
else:
    im = ax.imshow(H.T, origin="lower",
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   aspect="auto", cmap="hot")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    st.caption(f"bins = ({nx},{ny}), points={npts}")
    st.pyplot(fig)

# ---------------------- 匯出 ----------------------
st.subheader("匯出結果")

rows_roi = []
for r in ROI_RANGES:
    if r["type"] == "rect_px":
        x1, y1, x2, y2 = r["rect_px"]
        rows_roi.append({"ROI": r["name"], "type": "rect_px", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "poly_px": ""})
    else:
        poly = r["poly_px"]
        rows_roi.append({"ROI": r["name"], "type": "poly_px", "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan, "poly_px": json.dumps(poly.tolist() if poly is not None else [])})

df_roi_ranges = pd.DataFrame(rows_roi)
df_meta = pd.DataFrame([{
    "px_to_mm": px_to_mm,
    "fps": fps,
    "frame_start": frame_start,
    "frame_end": frame_end,
    "frame_count": frame_end - frame_start + 1,
    "roi_mode": roi_mode,
    "two_roi_method": two_roi_method if two_roi_method else "",
}])

if st.button("⬇️ 匯出 Excel"):
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_global.to_excel(writer, sheet_name="Global", index=False)
        df_dwell.to_excel(writer, sheet_name="ROI_Summary", index=False)
        df_roi_ranges.to_excel(writer, sheet_name="ROI_Definitions", index=False)
        df_meta.to_excel(writer, sheet_name="Meta_Info", index=False)
        if df_pref is not None:
            df_pref.to_excel(writer, sheet_name="Preference", index=False)
    excel_buf.seek(0)
    st.download_button("下載 Excel", data=excel_buf, file_name="all_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if st.button("⬇️ 匯出 PDF"):
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        for df, title, fs in [
            (df_global, "Global", 8),
            (df_dwell, "ROI Summary", 6),
            (df_pref, "Preference", 7),
            (df_roi_ranges, "ROI Definitions", 6),
            (df_meta, "Meta Info", 8),
        ]:
            if df is None:
                continue
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis("off")
            tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(fs)
            ax.set_title(title)
            pdf.savefig(fig)
            plt.close(fig)
    pdf_buf.seek(0)
    st.download_button("下載 PDF", data=pdf_buf, file_name="all_results.pdf", mime="application/pdf")

if st.button("⬇️ 匯出 ZIP"):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        # CSVs
        zf.writestr("global_summary.csv", df_global.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("roi_summary.csv", df_dwell.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("roi_definitions.csv", df_roi_ranges.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("meta_info.csv", df_meta.to_csv(index=False).encode("utf-8-sig"))
        if df_pref is not None:
            zf.writestr("preference.csv", df_pref.to_csv(index=False).encode("utf-8-sig"))
    zip_buf.seek(0)
    st.download_button("下載 ZIP", data=zip_buf, file_name="all_results_bundle.zip", mime="application/zip")
