# idtracker_dashboard_streamlit.py
# 功能總覽（2025-09-29 完整版：ROI_0~F、自動外框、軌跡/Heatmap mm+px、速度角速度、匯出工具）

import os
import io
import math
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
from matplotlib.patches import Rectangle

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
        ax.set_xticks(np.arange(lo, hi+xtick/10.0, xtick))
    if ytick:
        lo, hi = ax.get_ylim()
        ax.set_yticks(np.arange(lo, hi+ytick/10.0, ytick))

def _lim_tuple(vmin, vmax):
    if vmax <= vmin or (vmin==0.0 and vmax==0.0):
        return None
    return (vmin, vmax)

def _tick_val(v):
    return None if v <= 0 else v

# ---------------------- Streamlit App 設定 ----------------------
st.set_page_config(layout="wide")
st.title("🐭 idtracker.ai Dashboard")

uploaded = st.file_uploader("請上傳軌跡檔 (.h5 / .hdf5 / .npz)", type=["h5","hdf5","npz"])

# Sidebar
st.sidebar.header("參數設定")
fps = st.sidebar.number_input("FPS", value=30.0, step=1.0)
px_to_mm = st.sidebar.number_input("px_to_mm (mm/px)", value=0.10000, step=0.00001, min_value=0.00001, format="%.5f")
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

# ---------------------- 載入軌跡 ----------------------
def load_trajectories(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".h5",".hdf5"]:
        with h5py.File(path,"r") as f:
            if "trajectories" in f:
                arr = f["trajectories"][()]
            elif "positions" in f:
                arr = f["positions"][()]
            else:
                raise ValueError("H5 檔案中找不到 'trajectories' 或 'positions'")
    elif ext==".npz":
        data = np.load(path, allow_pickle=True)
        if "positions" not in data:
            raise ValueError("NPZ 檔案中找不到 'positions'")
        arr = data["positions"]
    else:
        raise ValueError("僅支援 h5/npz")
    T,N,_ = arr.shape
    positions = {i:arr[:,i,:] for i in range(N)}
    return positions, {"frame_count":T, "ids":list(range(N))}

def _detect_outer_bbox_from_file(path):
    import numpy as np
    ext = os.path.splitext(path)[1].lower()
    keys = ["arena_bbox","bbox","roi_rect"]
    cand = []
    if ext in [".h5",".hdf5"]:
        with h5py.File(path,"r") as f:
            for k in keys:
                if k in f: 
                    try: cand.append(f[k][()])
                    except: pass
    elif ext==".npz":
        data = np.load(path, allow_pickle=True)
        for k in keys:
            if k in data: cand.append(data[k])
    for c in cand:
        a = np.array(c,dtype=float).squeeze()
        if a.size==4:
            return float(a[0]),float(a[1]),float(a[2]),float(a[3])
    return None

def generate_auto_rois(x1,y1,x2,y2,target_side="左"):
    x3=(x1+x2)/2; y3=(y1+y2)/2
    rois=[{"name":"ROI_0","rect":(x1,y1,x2,y2)}]
    if target_side=="左":
        rois += [
            {"name":"ROI_A","rect":(x1,y1,x3,y2)},
            {"name":"ROI_B","rect":(x3,y1,x2,y2)},
            {"name":"ROI_E","rect":(x1,y1,x3,y3)},
            {"name":"ROI_F","rect":(x3,y1,x2,y3)}
        ]
    else:
        rois += [
            {"name":"ROI_A","rect":(x3,y1,x2,y2)},
            {"name":"ROI_B","rect":(x1,y1,x3,y2)},
            {"name":"ROI_E","rect":(x3,y1,x2,y3)},
            {"name":"ROI_F","rect":(x1,y1,x3,y3)}
        ]
    rois.append({"name":"ROI_C","rect":(x1,y1,x2,y3)})
    rois.append({"name":"ROI_D","rect":(x1,y3,x2,y2)})
    return rois

if uploaded is None:
    st.info("請上傳檔案以繼續")
    st.stop()

suffix = "."+uploaded.name.split(".")[-1]
with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path=tmp.name

positions_px, meta = load_trajectories(tmp_path)
ids=meta["ids"]; total_frames=meta["frame_count"]

detected_bbox = _detect_outer_bbox_from_file(tmp_path)
if detected_bbox is None:
    all_xy_tmp = np.vstack([positions_px[i] for i in ids])
    x1,y1=np.min(all_xy_tmp[:,0]),np.min(all_xy_tmp[:,1])
    x2,y2=np.max(all_xy_tmp[:,0]),np.max(all_xy_tmp[:,1])
    detected_bbox=(x1,y1,x2,y2)
x1,y1,x2,y2=detected_bbox
ROI_RANGES = generate_auto_rois(x1,y1,x2,y2,target_side)

# ---------------------- Frame 範圍 ----------------------
max_frame=max(0,total_frames-1)
frame_start=st.sidebar.number_input("Start frame",0,max_frame,0)
frame_end=st.sidebar.number_input("End frame",0,max_frame,max_frame)

# ---------------------- per-ID 計算 ----------------------
def compute_speed_mm_per_s(xy_mm,fps):
    diff=np.diff(xy_mm,axis=0)
    dist=np.linalg.norm(diff,axis=1)
    return np.concatenate([[np.nan],dist*fps])

def compute_ang_vel_deg_per_s(xy_mm,fps,eps=1e-6):
    v=np.diff(xy_mm,axis=0)*fps
    ang_vel=np.full(len(xy_mm),np.nan)
    if len(v)>=2:
        dot=np.sum(v[1:]*v[:-1],axis=1)
        cross=v[1:,0]*v[:-1,1]-v[1:,1]*v[:-1,0]
        norm=np.linalg.norm(v[1:],axis=1)*np.linalg.norm(v[:-1],axis=1)
        norm=np.where(norm<eps,eps,norm)
        dtheta=np.arctan2(cross,dot)
        av=dtheta*fps*(180/np.pi)
        av=(av+180)%360-180
        ang_vel[2:]=av
    return ang_vel

per_id={}
for i in ids:
    xy_px=positions_px[i][frame_start:frame_end+1,:]
    xy_mm=xy_px*px_to_mm
    spd=compute_speed_mm_per_s(xy_mm,fps)
    ang=compute_ang_vel_deg_per_s(xy_mm,fps)
    per_id[i]={"xy_mm":xy_mm,"speed":spd,"angvel":ang}

# ---------------------- Global Summary ----------------------
all_dist=[]
for data in per_id.values():
    xy=data["xy_mm"]
    if xy.shape[0]>=2:
        steps=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        all_dist.append(np.nansum(steps))
global_distance_mm=float(np.nansum(all_dist))
global_mean_speed=float(np.nanmean(np.concatenate([d["speed"] for d in per_id.values()])))
global_mean_ang=float(np.nanmean(np.concatenate([d["angvel"] for d in per_id.values()])))

df_global=pd.DataFrame([{
    "Total_distance_mm":round(global_distance_mm,3),
    "Mean_speed_mm_s":round(global_mean_speed,3),
    "Mean_ang_deg_s":round(global_mean_ang,3)
}])
st.subheader("整體統計")
st.dataframe(df_global,use_container_width=True)

# ---------------------- ROI 統計 ----------------------
def in_rect(xy_mm,rect):
    x1,y1,x2,y2=rect
    return (xy_mm[:,0]>=x1)&(xy_mm[:,0]<=x2)&(xy_mm[:,1]>=y1)&(xy_mm[:,1]<=y2)

df_dwell=[]
for i,data in per_id.items():
    xy=data["xy_mm"]
    for roi in ROI_RANGES:
        name=roi["name"]
        mask=in_rect(xy,roi["rect"])
        frames_in=int(np.count_nonzero(mask))
        mean_spd=float(np.nanmean(np.where(mask,data["speed"],np.nan)))
        mean_ang=float(np.nanmean(np.where(mask,data["angvel"],np.nan)))
        df_dwell.append({
            "ID":i,"ROI":name,
            "Frames_in_ROI":frames_in,
            "Time_in_ROI_s":round(frames_in/fps,3),
            "Mean_speed_mm_s":round(mean_spd,3),
            "Mean_ang_deg_s":round(mean_ang,3)
        })

df_dwell=pd.DataFrame(df_dwell)
st.subheader("ROI 統計")
st.dataframe(df_dwell,use_container_width=True)
# ---------------------- Sidebar: ROI 顯示選項 ----------------------
roi_names = [r["name"] for r in ROI_RANGES]
show_rois = st.sidebar.multiselect(
    "要在圖上顯示哪些 ROI 框線？",
    options=roi_names,
    default=roi_names
)

# ---------------------- 視覺化 ----------------------
st.subheader("軌跡圖 (mm)")
fig, ax = plt.subplots(figsize=(6, 6))
for i, data in per_id.items():
    xy = data["xy_mm"]
    ax.plot(xy[:,0], xy[:,1], lw=0.7, alpha=0.6, label=f"ID {i}")
for roi in ROI_RANGES:
    if roi["name"] in show_rois:
        x1,y1,x2,y2 = roi["rect"]
        rect = Rectangle((x1,y1), x2-x1, y2-y1, fill=False, lw=1.0, alpha=0.8)
        ax.add_patch(rect)
apply_axis(ax,
    xlim=_lim_tuple(x_min_mm, x_max_mm),
    ylim=_lim_tuple(y_min_mm, y_max_mm),
    xtick=_tick_val(x_tick_mm), ytick=_tick_val(y_tick_mm))
ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
ax.legend(fontsize=6, loc="best")
st.pyplot(fig)

st.subheader("軌跡圖 (px)")
fig, ax = plt.subplots(figsize=(6, 6))
for i in ids:
    xy_px = positions_px[i][frame_start:frame_end+1, :]
    ax.plot(xy_px[:,0], xy_px[:,1], lw=0.7, alpha=0.6, label=f"ID {i}")
for roi in ROI_RANGES:
    if roi["name"] in show_rois:
        x1,y1,x2,y2 = roi["rect"]
        rect = Rectangle((x1,y1), x2-x1, y2-y1, fill=False, lw=1.0, alpha=0.8)
        ax.add_patch(rect)
ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
ax.legend(fontsize=6, loc="best")
st.pyplot(fig)

st.subheader("Heatmap (mm)")
all_xy_mm_plot = np.vstack([d["xy_mm"] for d in per_id.values()])
fig, ax = plt.subplots(figsize=(6, 6))
H, xedges, yedges = np.histogram2d(all_xy_mm_plot[:,0], all_xy_mm_plot[:,1],
    bins=[int((np.max(all_xy_mm_plot[:,0])-np.min(all_xy_mm_plot[:,0]))/bin_mm),
          int((np.max(all_xy_mm_plot[:,1])-np.min(all_xy_mm_plot[:,1]))/bin_mm)])
im = ax.imshow(H.T, origin="lower",
    extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],
    aspect="auto", cmap="hot")
fig.colorbar(im, ax=ax)
for roi in ROI_RANGES:
    if roi["name"] in show_rois:
        x1,y1,x2,y2 = roi["rect"]
        rect = Rectangle((x1,y1), x2-x1, y2-y1, fill=False, lw=1.0, alpha=0.8, color="cyan")
        ax.add_patch(rect)
ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
st.pyplot(fig)

st.subheader("Heatmap (px)")
all_xy_px_plot = np.vstack([positions_px[i][frame_start:frame_end+1, :] for i in ids])
fig, ax = plt.subplots(figsize=(6, 6))
H, xedges, yedges = np.histogram2d(all_xy_px_plot[:,0], all_xy_px_plot[:,1],
    bins=[int((np.max(all_xy_px_plot[:,0])-np.min(all_xy_px_plot[:,0]))/bin_px),
          int((np.max(all_xy_px_plot[:,1])-np.min(all_xy_px_plot[:,1]))/bin_px)])
im = ax.imshow(H.T, origin="lower",
    extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],
    aspect="auto", cmap="hot")
fig.colorbar(im, ax=ax)
for roi in ROI_RANGES:
    if roi["name"] in show_rois:
        x1,y1,x2,y2 = roi["rect"]
        rect = Rectangle((x1,y1), x2-x1, y2-y1, fill=False, lw=1.0, alpha=0.8, color="cyan")
        ax.add_patch(rect)
ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
st.pyplot(fig)

# 速度與角速度
st.subheader("速度與角速度曲線")
for i, data in per_id.items():
    fig, ax = plt.subplots(2,1, figsize=(8,4), sharex=True)
    ax[0].plot(data["speed"], lw=0.8)
    ax[0].set_ylabel("Speed (mm/s)")
    ax[1].plot(data["angvel"], lw=0.8)
    ax[1].set_ylabel("Ang vel (deg/s)")
    ax[1].set_xlabel("Frame")
    fig.suptitle(f"ID {i}")
    st.pyplot(fig)

# ---------------------- 匯出 Excel/PDF/ZIP ----------------------
st.subheader("匯出結果")

df_roi_ranges = pd.DataFrame([
    {"ROI": r["name"], "x1": r["rect"][0], "y1": r["rect"][1],
     "x2": r["rect"][2], "y2": r["rect"][3]}
    for r in ROI_RANGES
])
df_meta = pd.DataFrame([{
    "px_to_mm": px_to_mm,
    "frame_start": frame_start,
    "frame_end": frame_end,
    "frame_count": frame_end - frame_start + 1
}])

if st.button("⬇️ 匯出 Excel"):
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        df_global.to_excel(writer, sheet_name="Global", index=False)
        df_dwell.to_excel(writer, sheet_name="ROI_Summary", index=False)
        df_roi_ranges.to_excel(writer, sheet_name="ROI_Ranges", index=False)
        df_meta.to_excel(writer, sheet_name="Meta_Info", index=False)
    excel_buf.seek(0)
    st.download_button("下載 Excel", data=excel_buf,
        file_name="all_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if st.button("⬇️ 匯出 PDF"):
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        # Global
        fig, ax = plt.subplots(figsize=(5,2))
        ax.axis("off")
        tbl = ax.table(cellText=df_global.values, colLabels=df_global.columns, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        pdf.savefig(fig); plt.close(fig)
        # ROI Summary
        fig, ax = plt.subplots(figsize=(8,4))
        ax.axis("off")
        tbl = ax.table(cellText=df_dwell.values, colLabels=df_dwell.columns, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(6)
        pdf.savefig(fig); plt.close(fig)
        # ROI Ranges
        fig, ax = plt.subplots(figsize=(8,3))
        ax.axis("off")
        tbl = ax.table(cellText=df_roi_ranges.values, colLabels=df_roi_ranges.columns, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        pdf.savefig(fig); plt.close(fig)
        # Meta Info
        fig, ax = plt.subplots(figsize=(6,2))
        ax.axis("off")
        tbl = ax.table(cellText=df_meta.values, colLabels=df_meta.columns, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        pdf.savefig(fig); plt.close(fig)
    pdf_buf.seek(0)
    st.download_button("下載 PDF", data=pdf_buf,
        file_name="all_results.pdf", mime="application/pdf")

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
    st.download_button("下載 ZIP", data=zip_buf,
        file_name="all_results_bundle.zip", mime="application/zip")
