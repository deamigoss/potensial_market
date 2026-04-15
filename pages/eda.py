import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.preprocessing import preprocessing_pipeline

st.set_page_config(page_title="EDA - Market Intelligence", layout="wide", page_icon="📊")

from components.sidebar import sidebar
sidebar()

# =====================
# CSS
# =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp {
    background: radial-gradient(ellipse at 20% 10%,#1a2744 0%,#0f172a 50%,#0d1117 100%);
    color: #f8fafc;
}
header[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0) !important;
    border-bottom: none !important; box-shadow: none !important;
}
[data-testid="stSidebar"]    { border-right: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 0rem !important; margin-top: -20px !important; }
.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(15,23,42,0.6); padding: 10px;
    border-radius: 20px; gap: 12px; border: 1px solid rgba(255,255,255,0.05);
}
.stTabs [data-baseweb="tab"] {
    height: 46px; background-color: transparent; border-radius: 12px;
    color: #94a3b8; border: none; transition: all 0.4s; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#1d6eef,#972db4) !important;
    color: white !important; box-shadow: 0 5px 20px rgba(29,110,239,0.4);
    transform: translateY(-2px);
}
[data-testid="stDataFrame"] {
    background: rgba(30,41,59,0.3) !important; border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
::-webkit-scrollbar { height: 8px; width: 8px; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(to bottom,#1d6eef,#972db4); border-radius: 10px;
}
.stat-card {
    background: rgba(30,41,59,0.55); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;
    padding: 18px 20px; transition: all 0.3s ease;
}
.stat-card:hover { border-color: rgba(0,212,255,0.35); transform: translateY(-4px); }
.stat-label {
    color: #64748b; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase; display: block; margin-bottom: 6px;
}
.stat-value {
    font-size: 1.8rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg,#00d4ff,#972db4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-sub { color: #64748b; font-size: 0.78rem; margin-top: 4px; display: block; }
.section-header {
    background: linear-gradient(135deg,rgba(29,110,239,0.1),rgba(151,45,180,0.08));
    border: 1px solid rgba(29,110,239,0.2); border-radius: 14px;
    padding: 14px 20px; margin-bottom: 16px;
}
.section-header p { margin: 0; color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    df = pd.read_excel("data/dataset2k22-2k26.xlsx")
    df_proc, _ = preprocessing_pipeline(df)
    return df_proc

df = load_data()

COL_HH     = "Jumlah Household"
COL_SINYAL = "Jumlah Desa/Kelurahan Yang Menerima Sinyal 5G/4G/LTE (Persen)"
COL_UMP    = "UMP (Rupiah)"
COL_PDRB   = "PDRB (Ribu Rp)"
COL_EXP    = "Rata-Rata Pengeluaran per Kapita Bukan Makanan (Rupiah)"
COL_TV     = "Tercover TV Digital?"
numeric_cols = [c for c in [COL_HH,COL_SINYAL,COL_UMP,COL_PDRB,COL_EXP] if c in df.columns]

# =====================
# HEADER
# =====================
st.markdown("""
<div style="background:linear-gradient(135deg,rgba(29,110,239,0.15),rgba(151,45,180,0.12));
            border:1px solid rgba(0,212,255,0.2);border-radius:22px;
            padding:36px 40px;margin-bottom:8px;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-40px;right:-40px;width:220px;height:220px;
              background:radial-gradient(circle,rgba(0,212,255,0.07),transparent 70%);pointer-events:none;"></div>
  <span style="background:rgba(0,212,255,0.12);border:1px solid rgba(0,212,255,0.35);border-radius:20px;
               padding:4px 14px;font-size:0.75rem;font-weight:700;color:#00d4ff;letter-spacing:0.8px;
               text-transform:uppercase;display:inline-block;margin-bottom:12px;">📊 Market Intelligence</span>
  <h1 style="font-size:2rem;font-weight:800;margin:0 0 10px 0;
             background:linear-gradient(90deg,#f8fafc 30%,#00d4ff 70%,#972db4);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    Exploratory Data Analysis
  </h1>
  <p style="color:#94a3b8;margin:0;font-size:0.95rem;line-height:1.7;max-width:680px;">
    Eksplorasi variabel <b>ekonomi</b>, <b>demografi</b>, dan <b>infrastruktur</b> wilayah Indonesia
    sebagai fondasi strategi distribusi produk Transvision.
  </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# =====================
# RINGKASAN STATISTIK
# =====================
st.subheader("📈 Ringkasan Statistik Dataset")
cols_stat = st.columns(4)
stat_items = [
    ("🌍 Total Wilayah",  f"{len(df):,}",                "Baris kab/kota dalam dataset"),
    ("📊 Rata-rata PDRB", f"{df[COL_PDRB].mean():,.0f}" if COL_PDRB in df.columns else "N/A",
                          "PDRB per kapita (Ribu Rp)"),
    ("💰 Rata-rata UMP",  f"Rp {df[COL_UMP].mean():,.0f}" if COL_UMP in df.columns else "N/A",
                          "Upah minimum rata-rata"),
    ("📶 Sinyal 4G/5G",   f"{df[COL_SINYAL].mean():.1f}%" if COL_SINYAL in df.columns else "N/A",
                          "Rata-rata penetrasi sinyal"),
]
for col,(lbl,val,sub) in zip(cols_stat, stat_items):
    with col:
        st.markdown(f"""
        <div class="stat-card">
          <span class="stat-label">{lbl}</span>
          <p class="stat-value">{val}</p>
          <span class="stat-sub">{sub}</span>
        </div>""", unsafe_allow_html=True)

st.divider()

# =====================
# TAB 1 — SCATTER & KORELASI
# =====================
st.subheader("🔗 Korelasi & Hubungan Variabel")
tab1, tab2 = st.tabs(["📍 Scatter Analysis","🔥 Correlation Heatmap"])

with tab1:
    ca, cb = st.columns([1,3])
    with ca:
        st.markdown("#### ⚙️ Control Panel")
        x_ax    = st.selectbox("Sumbu X:",  numeric_cols, index=numeric_cols.index(COL_PDRB) if COL_PDRB in numeric_cols else 0)
        y_ax    = st.selectbox("Sumbu Y:",  numeric_cols, index=numeric_cols.index(COL_HH)   if COL_HH   in numeric_cols else 1)
        col_by  = st.selectbox("Warnai oleh:", [c for c in ["Provinsi","Tahun",COL_TV] if c in df.columns])
        log_x   = st.checkbox("Log scale X", value=True)
        st.markdown("""
        <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);
                    border-radius:10px;padding:10px 12px;margin-top:10px;">
          <p style="color:#94a3b8;font-size:0.8rem;margin:0;line-height:1.5;">
            💡 Ukuran titik = Jumlah Household.<br>
            Hover untuk detail kab/kota.
          </p>
        </div>""", unsafe_allow_html=True)
    with cb:
        fig_sc = px.scatter(
            df, x=x_ax, y=y_ax, color=col_by,
            hover_name="Kota/Kab" if "Kota/Kab" in df.columns else None,
            size=COL_HH if COL_HH in df.columns else None,
            size_max=40, log_x=log_x,
            title=f"{x_ax} vs {y_ax}",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_sc.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                             font=dict(color="#94a3b8"),height=460)
        fig_sc.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        fig_sc.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig_sc, use_container_width=True)

with tab2:
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        # Label pendek untuk keterbacaan
        short = {COL_HH:"Household",COL_SINYAL:"Sinyal 4G",COL_UMP:"UMP",
                 COL_PDRB:"PDRB",COL_EXP:"Pengeluaran"}
        corr.index   = [short.get(c,c) for c in corr.index]
        corr.columns = [short.get(c,c) for c in corr.columns]

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}", colorscale="RdBu_r", zmid=0,
            colorbar=dict(tickfont=dict(color="#94a3b8"),
                          bgcolor="rgba(15,23,42,0.8)",bordercolor="rgba(255,255,255,0.1)",
                          thickness=14)
        ))
        fig_corr.update_layout(
            title="Matriks Korelasi Indikator Strategis",
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#f8fafc",size=13),height=480,
            margin=dict(l=10,r=10,t=50,b=10)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("""
        <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);
                    border-radius:12px;padding:14px 18px;margin-top:4px;">
          <p style="color:#94a3b8;font-size:0.85rem;margin:0;line-height:1.6;">
            <b style="color:#00d4ff;">Merah tua</b> = korelasi positif kuat &nbsp;·&nbsp;
            <b style="color:#60a5fa;">Biru tua</b> = korelasi negatif kuat &nbsp;·&nbsp;
            <b style="color:#94a3b8;">Putih</b> = tidak berkorelasi.
            Variabel dengan korelasi tinggi terhadap satu sama lain berpotensi membawa informasi redundan ke model.
          </p>
        </div>""", unsafe_allow_html=True)

st.divider()

# =====================
# RANKING WILAYAH
# =====================
st.subheader("🏆 Peringkat Wilayah Strategis")
param_r = st.selectbox("Urutkan berdasarkan:", numeric_cols, key="rank")
agg_f   = "sum" if "Jumlah" in param_r else "mean"
lbl_agg = "Total" if agg_f=="sum" else "Rata-Rata"

df_rank  = df.groupby(["Kota/Kab","Tahun"])[param_r].agg(agg_f).reset_index() if "Kota/Kab" in df.columns else None
if df_rank is not None:
    top_cities = df.groupby("Kota/Kab")[param_r].agg(agg_f).nlargest(15).index
    df_plot    = df_rank[df_rank["Kota/Kab"].isin(top_cities)]
    fig_top = px.bar(
        df_plot, x=param_r, y="Kota/Kab", color="Tahun", barmode="group",
        orientation='h', text_auto='.2s',
        title=f"15 Wilayah {lbl_agg} {param_r} Tertinggi",
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Electric_r,
    )
    fig_top.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        yaxis={'categoryorder':'total ascending'},
        height=580,legend_title="Tahun",
        font=dict(color="#94a3b8"),
    )
    fig_top.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    if "Persen" in param_r or "%" in param_r:
        fig_top.update_xaxes(ticksuffix="%")
        fig_top.update_traces(texttemplate='%{x:.1f}%')
    st.plotly_chart(fig_top, use_container_width=True)

st.divider()

# =====================
# TREN WAKTU
# =====================
st.subheader("📈 Tren Waktu per Provinsi")
if "Tahun" in df.columns and "Provinsi" in df.columns:
    sel_tren = st.selectbox("Variabel Tren:", numeric_cols, key="tren")
    all_prov = sorted(df["Provinsi"].unique())
    sel_prov = st.multiselect("Pilih Provinsi (maks 8):", all_prov,
                               default=all_prov[:5] if len(all_prov)>=5 else all_prov)
    if sel_prov:
        sel_prov = sel_prov[:8]
        df_tren  = df[df["Provinsi"].isin(sel_prov)].groupby(["Tahun","Provinsi"])[sel_tren].mean().reset_index()
        fig_line = px.line(
            df_tren, x="Tahun", y=sel_tren, color="Provinsi",
            markers=True, template="plotly_dark",
            title=f"Tren {sel_tren} dari Waktu ke Waktu",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_line.update_traces(line_width=2.5, marker_size=7)
        fig_line.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#94a3b8"),height=420,
            legend=dict(bgcolor="rgba(15,23,42,0.7)",bordercolor="rgba(255,255,255,0.1)",borderwidth=1),
        )
        fig_line.update_xaxes(gridcolor="rgba(255,255,255,0.05)",dtick=1)
        fig_line.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig_line, use_container_width=True)