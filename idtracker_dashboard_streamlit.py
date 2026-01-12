# idtracker_dashboard_streamlit.py
# 覆蓋版（2026-01-12 / FIX）
# 修正：
# A) streamlit-drawable-canvas 在 Cloud 可能因 background_image 轉 URL 掛掉
#    → 改用 data URL（background_image_url）避免 image_to_url 崩潰
# B) Apply ROI to plots 之前放在 expander 外、且用到 rx1/rx2 造成 NameError
#    → 改成「按鈕只用 session_state.roi_pts 重新算 ROI_0」，並放在 expander 內（img_file 有效才顯示）
# C) include_mid / roi_mode / show_rois 一鍵同步 OK
# D) Heatmap bins 防呆（保留）
#
# 依賴（Streamlit Cloud requirements.txt）：
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
import math
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


def _safe_bins(vmin, vmax, bin_size, max_bins=800):
    """給 np.histogram2d 的 bins 防呆：至少 1，且避免太大"""
    span = float(vmax - vmin)
    if (not np.isfinite(span)) or span <= 0:
        return 1
    b = int(span / float(bin_size))
    b = max(1, b)
    b = min(max_bins, b)
    return b


def pil_to_data_url(img: Image.Image, fmt="PNG") -> str:
    """把 PIL Image 轉成 data URL（避免 drawable-canvas 在 Cloud 的 image_to_url 崩潰）"""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def resize_if_too_large(img: Image.Image, max_side=1600) -> Image.Image:
    """避免 data URL 太大：長邊>max_side 就縮圖"""
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h))


# ---------------------- Streamlit App 設定 ----------------------
st.set_page_config(layout="wide")
st.title("🐭 idtracker.ai Dashboard")

uploaded = st.file_uploader("請上傳軌跡檔 (.h5 / .hdf5 / .npz)", type=["h5", "hdf5", "npz"])

# Sidebar
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
    """ROI_0 + Left/Right 1/3（可選中間 1/3）"""
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


# ---------------------- PNG 量測工具（Sidebar Expander） ----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🧰 ROI/座標量測（PNG→點選→px/mm）")

if "roi_pts" not in st.session_state:
    st.session_state.roi_pts = []  # 用來存兩點 (x,y)


with st.sidebar.expander("打開量測工具", expanded=False):
    if not _HAS_CANVAS:
        st.error("缺少 streamlit-drawable-canvas。請在 requirements.txt 加上 streamlit-drawable-canvas==0.9.3 後重新部署。")
    else:
        img_file = st.file_uploader("上傳 frame 圖片 (PNG/JPG)", type=["png", "jpg", "jpeg"], key="roi_measure_img")
        st.caption("建議：用 Fiji 先從影片 Duplicate current slice → Save As PNG，再上傳來點。")

        if img_file is not None:
            img = Image.open(img_file).convert("RGB")
            img = resize_if_too_large(img, max_side=1600)
            w, h = img.size

            st.write(f"Image size: {w}×{h} px")
            st.caption("點一下回報座標；若要定 ROI_0，請連點兩次（左上→右下）。")

            bg_url = pil_to_data_url(img, fmt="PNG")

            # ✅ 關鍵：用 background_image_url（data URL），避免 Cloud 的 image_to_url 出錯
            canvas = st_canvas(
                fill_color="rgba(255, 0, 0, 0.0)",
                stroke_width=2,
                stroke_color="rgba(0, 255, 255, 1.0)",
                background_image_url=bg_url,
                update_streamlit=True,
                height=h,
                width=w,
                drawing_mode="point",
                point_display_radius=4,
                key="roi_measure_canvas",
            )

            # 讀取最後一個點
            if canvas.json_data is not None:
                objs = canvas.json_data.get("objects", [])
                if len(objs) > 0:
                    last = objs[-1]
                    x_px = float(last.get("left", np.nan))
                    y_px = float(last.get("top", np.nan))

                    if np.isfinite(x_px) and np.isfinite(y_px):
                        x_mm = x_px * px_to_mm
                        y_mm = y_px * px_to_mm
                        st.success(f"點選：x={x_px:.1f}px, y={y_px:.1f}px  ｜  x={x_mm:.2f}mm, y={y_mm:.2f}mm")

                        # 只在點位與上一點差異>1px 時才加入，避免重複累積
                        pts = st.session_state.roi_pts
                        if len(pts) == 0 or (abs(pts[-1][0] - x_px) > 1 or abs(pts[-1][1] - y_px) > 1):
                            pts.append((x_px, y_px))
                            # 只保留前 2 點（ROI_0）
                            st.session_state.roi_pts = pts[:2]

            colA, colB = st.columns([1, 2])
            with colA:
                if st.button("清空點位", key="roi_clear_pts"):
                    st.session_state.roi_pts = []
                    st.rerun()
            with colB:
                st.write(f"已記錄點數：{len(st.session_state.roi_pts)}")
                if len(st.session_state.roi_pts) > 0:
                    st.write("Points (px):", st.session_state.roi_pts)

            # ROI_0 兩點 → 數值 + 預覽 + Apply
            if len(st.session_state.roi_pts) >= 2:
                (x1p, y1p) = st.session_state.roi_pts[0]
                (x2p, y2p) = st.session_state.roi_pts[1]
                rx1, rx2 = min(x1p, x2p), max(x1p, x2p)
                ry1, ry2 = min(y1p, y2p), max(y1p, y2p)

                st.markdown("**ROI_0 (px)**")
                st.code(f"({rx1:.1f}, {ry1:.1f}, {rx2:.1f}, {ry2:.1f})")

                st.markdown("**ROI_0 (mm)**")
                st.code(f"({rx1*px_to_mm:.2f}, {ry1*px_to_mm:.2f}, {rx2*px_to_mm:.2f}, {ry2*px_to_mm:.2f})")

                # 分割線預覽（用 matplotlib 疊在 PNG 上）
                figp, axp = plt.subplots(figsize=(6, 4))
                axp.imshow(img, origin="upper")
                axp.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, fill=False, lw=2))
                w0 = (rx2 - rx1)
                xL = rx1 + w0 / 3.0
                xR = rx1 + 2.0 * w0 / 3.0
                axp.axvline(xL, lw=2)
                axp.axvline(xR, lw=2)
                axp.set_title("ROI_0 + split 1/3 preview (image coords)")
                axp.set_xlabel("X (px)")
                axp.set_ylabel("Y (px)")
                st.pyplot(figp)

                # ✅ Apply 按鈕（只用 session_state.roi_pts 計算，不會 NameError）
                col_ap1, col_ap2 = st.columns([1, 2])
                with col_ap1:
                    if st.button("✅ Apply ROI to plots", key="btn_apply_roi_to_plots"):
                        st.session_state["roi0_x1"] = float(rx1)
                        st.session_state["roi0_y1"] = float(ry1)
                        st.session_state["roi0_x2"] = float(rx2)
                        st.session_state["roi0_y2"] = float(ry2)

                        # 切到 Manual 模式
                        st.session_state["roi_mode"] = "Manual ROI_0 + Split Left/Right 1/3"

                        # 自動只顯示 ROI_0 + Left/Right（含 mid 視 include_mid）
                        wanted = ["ROI_0", "ROI_LEFT_1_3", "ROI_RIGHT_1_3"]
                        if st.session_state.get("include_mid", True):
                            wanted.insert(2, "ROI_MID_1_3")
                        st.session_state["show_rois"] = wanted

                        st.rerun()
                with col_ap2:
                    st.caption("按下後：ROI_0 會自動填入 Manual ROI_0，並讓下方 px 軌跡圖只顯示 ROI_0 + Left/Right（可選含 MID）。")

                st.caption("若你要分析左右兩側：建議 ROI_0 框包含完整 device 外框，左右 1/3 會自動從 ROI_0 切出。")
            else:
                st.info("提示：要啟用 Apply ROI，請先在 PNG 上點兩次定義 ROI_0。")


# ---------------------- Manual ROI_0 inputs（可被量測工具一鍵填入） ----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Manual ROI_0 (px)")

# 這四個 number_input 用 key 綁 session_state，才能被按鈕寫入
roi0_x1 = st.sidebar.number_input("ROI_0 x1 (px)", value=float(st.session_state.get("roi0_x1", 0.0)), step=1.0, key="roi0_x1")
roi0_y1 = st.sidebar.number_input("ROI_0 y1 (px)", value=float(st.session_state.get("roi0_y1", 0.0)), step=1.0, key="roi0_y1")
roi0_x2 = st.sidebar.number_input("ROI_0 x2 (px)", value=float(st.session_state.get("roi0_x2", 0.0)), step=1.0, key="roi0_x2")
roi0_y2 = st.sidebar.number_input("ROI_0 y2 (px)", value=float(st.session_state.get("roi0_y2", 0.0)), step=1.0, key="roi0_y2")

include_mid = st.sidebar.checkbox("也生成中間 1/3 ROI", value=True, key="include_mid")

col_fill, col_hint = st.sidebar.columns([1, 1])
with col_fill:
    if st.button("⬅️ 用量測兩點填入 ROI_0", key="btn_fill_roi0"):
        if len(st.session_state.roi_pts) >= 2:
            (x1p, y1p) = st.session_state.roi_pts[0]
            (x2p, y2p) = st.session_state.roi_pts[1]
            st.session_state["roi0_x1"] = float(min(x1p, x2p))
            st.session_state["roi0_x2"] = float(max(x1p, x2p))
            st.session_state["roi0_y1"] = float(min(y1p, y2p))
            st.session_state["roi0_y2"] = float(max(y1p, y2p))
            st.rerun()
        else:
            st.warning("請先在 PNG 上連點兩次（左上→右下）以定義 ROI_0。")
with col_hint:
    st.caption("量測兩點 → 一鍵填入")


# ---------------------- ROI 產生 ----------------------
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

# Frame 範圍
max_frame = max(0, total_frames - 1)
frame_start = st.sidebar.number_input("Start frame", 0, max_frame, 0)
frame_end = st.sidebar.number_input("End frame", 0, max_frame, max_frame)

# ROI
ROI_RANGES = []
if roi_mode.startswith("Auto"):
    detected_bbox = _detect_outer_bbox_from_file(tmp_path)
    if detected_bbox is None:
        all_xy_tmp = np.vstack([positions_px[i][frame_start : frame_end + 1, :] for i in ids])
        x1, y1 = np.nanmin(all_xy_tmp[:, 0]), np.nanmin(all_xy_tmp[:, 1])
        x2, y2 = np.nanmax(all_xy_tmp[:, 0]), np.nanmax(all_xy_tmp[:, 1])
        detected_bbox = (x1, y1, x2, y2)

    x1, y1, x2, y2 = detected_bbox
    ROI_RANGES = [{"name": "ROI_0", "rect": (x1, y1, x2, y2)}]
else:
    # Manual ROI_0
    x1, y1, x2, y2 = float(roi0_x1), float(roi0_y1), float(roi0_x2), float(roi0_y2)
    if (x2 <= x1) or (y2 <= y1):
        st.sidebar.warning("Manual ROI_0 需要滿足：x2>x1 且 y2>y1。否則只會生成 ROI_0（且可能無效）。")
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
    xy_px = positions_px[i][frame_start : frame_end + 1, :]
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


df_dwell_rows = []
for i in ids:
    xy_px = positions_px[i][frame_start : frame_end + 1, :]
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
                "ID": i,
                "ROI": name,
                "Frames_in_ROI": frames_in,
                "Time_in_ROI_s": round(frames_in / fps, 3),
                "Mean_speed_mm_s": round(mean_spd, 3) if np.isfinite(mean_spd) else np.nan,
                "Mean_ang_deg_s": round(mean_ang, 3) if np.isfinite(mean_ang) else np.nan,
            }
        )

# 確保 df_dwell 即使空也有 columns
df_dwell = pd.DataFrame(
    df_dwell_rows,
    columns=["ID", "ROI", "Frames_in_ROI", "Time_in_ROI_s", "Mean_speed_mm_s", "Mean_ang_deg_s"],
)

st.subheader("ROI 統計")
st.dataframe(df_dwell, use_container_width=True)


# ---------------------- Sidebar: ROI 顯示選項 ----------------------
roi_names = [r["name"] for r in ROI_RANGES]
default_show = roi_names if len(roi_names) > 0 else []
show_rois = st.sidebar.multiselect(
    "要在圖上顯示哪些 ROI？",
    options=roi_names,
    default=default_show,
    key="show_rois",
)


# ---------------------- 左右 ROI 分析（只在 Manual Split 模式顯示） ----------------------
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
            pi = (tL - tR) / denom

            df_pref = pd.DataFrame(
                {
                    "ID": pivot.index,
                    "Time_Left_s": np.round(tL.values, 3),
                    "Time_Right_s": np.round(tR.values, 3),
                    "PreferenceIndex_(L-R)/(L+R)": np.round(pi.values, 3),
                }
            )
            st.dataframe(df_pref, use_container_width=True)

            TL = float(np.nansum(tL.values))
            TR = float(np.nansum(tR.values))
            PI_all = (TL - TR) / (TL + TR) if (TL + TR) > 0 else np.nan
            st.caption(f"All IDs total: Left={TL:.3f}s, Right={TR:.3f}s, PI={PI_all:.3f}")


# ---------------------- 視覺化 ----------------------
st.subheader("軌跡圖 (mm)")
fig, ax = plt.subplots(figsize=(6, 6))
for i, data in per_id.items():
    xy = data["xy_mm"]
    ax.plot(xy[:, 0], xy[:, 1], lw=0.7, alpha=0.6, label=f"ID {i}")

# ROI：把 px ROI 轉成 mm 畫上去
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
    xy_px = positions_px[i][frame_start : frame_end + 1, :]
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

st.subheader("Heatmap (mm)")
all_xy_mm_plot = np.vstack([d["xy_mm"] for d in per_id.values()]) if len(per_id) else np.zeros((0, 2))
fig, ax = plt.subplots(figsize=(6, 6))
if all_xy_mm_plot.shape[0] < 2:
    st.warning("Heatmap(mm) 資料不足（點數太少），略過。")
else:
    bx = _safe_bins(np.nanmin(all_xy_mm_plot[:, 0]), np.nanmax(all_xy_mm_plot[:, 0]), bin_mm)
    by = _safe_bins(np.nanmin(all_xy_mm_plot[:, 1]), np.nanmax(all_xy_mm_plot[:, 1]), bin_mm)

    H, xedges, yedges = np.histogram2d(all_xy_mm_plot[:, 0], all_xy_mm_plot[:, 1], bins=[bx, by])
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
            rect = Rectangle((x1m, y1m), x2m - x1m, y2m - y1m, fill=False, lw=1.2, alpha=0.9, color="cyan")
            ax.add_patch(rect)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    st.pyplot(fig)

st.subheader("Heatmap (px)")
all_xy_px_plot = np.vstack([positions_px[i][frame_start : frame_end + 1, :] for i in ids]) if len(ids) else np.zeros((0, 2))
fig, ax = plt.subplots(figsize=(6, 6))
if all_xy_px_plot.shape[0] < 2:
    st.warning("Heatmap(px) 資料不足（點數太少），略過。")
else:
    bx = _safe_bins(np.nanmin(all_xy_px_plot[:, 0]), np.nanmax(all_xy_px_plot[:, 0]), bin_px)
    by = _safe_bins(np.nanmin(all_xy_px_plot[:, 1]), np.nanmax(all_xy_px_plot[:, 1]), bin_px)

    H, xedges, yedges = np.histogram2d(all_xy_px_plot[:, 0], all_xy_px_plot[:, 1], bins=[bx, by])
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
            rect = Rectangle((x1p, y1p), x2p - x1p, y2p - y1p, fill=False, lw=1.2, alpha=0.9, color="cyan")
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
    [{"ROI": r["name"], "x1_px": r["rect"][0], "y1_px": r["rect"][1], "x2_px": r["rect"][2], "y2_px": r["rect"][3]} for r in ROI_RANGES]
)
df_meta = pd.DataFrame(
    [
        {
            "px_to_mm": px_to_mm,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_count": frame_end - frame_start + 1,
            "roi_mode": roi_mode,
        }
    ]
)

if st.button("⬇️ 匯出 Excel"):
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_global.to_excel(writer, sheet_name="Global", index=False)
        df_dwell.to_excel(writer, sheet_name="ROI_Summary", index=False)
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

        # ROI Ranges
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        tbl = ax.table(cellText=df_roi_ranges.values, colLabels=df_roi_ranges.columns, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
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
            df_roi_ranges.to_excel(writer, sheet_name="ROI_Ranges", index=False)
            df_meta.to_excel(writer, sheet_name="Meta_Info", index=False)
        excel_bytes.seek(0)
        zf.writestr("all_results.xlsx", excel_bytes.read())

        # CSV
        zf.writestr("global_summary.csv", df_global.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("roi_summary.csv", df_dwell.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("roi_ranges.csv", df_roi_ranges.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("meta_info.csv", df_meta.to_csv(index=False).encode("utf-8-sig"))

    zip_buf.seek(0)
    st.download_button("下載 ZIP", data=zip_buf, file_name="all_results_bundle.zip", mime="application/zip")
