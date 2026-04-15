import streamlit as st
import pandas as pd
import os
import base64
import plotly.express as px
from utils.preprocessing import preprocessing_pipeline, get_training_data

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Potensi Produk Transvision",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

from components.sidebar import sidebar
sidebar()

# =====================
# CUSTOM CSS
# =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(ellipse at 20% 10%, #1a2744 0%, #0f172a 50%, #0d1117 100%);
    color: #f8fafc;
}
header[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0) !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"]   { border-right: none !important; }
[data-testid="stDecoration"]{ display: none !important; }
.block-container { padding-top: 0rem !important; margin-top: -20px !important; }

/* ── HERO BANNER ── */
.hero-banner {
    background: linear-gradient(135deg,
        rgba(29,110,239,0.18) 0%,
        rgba(151,45,180,0.14) 50%,
        rgba(0,212,255,0.10) 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 24px;
    padding: 40px 44px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem; font-weight: 800; margin: 0 0 10px 0;
    background: linear-gradient(90deg, #f8fafc 30%, #00d4ff 70%, #972db4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.hero-sub {
    color: #94a3b8; font-size: 1rem; margin: 0; line-height: 1.7; max-width: 700px;
}
.hero-badge {
    display: inline-block; background: rgba(0,212,255,0.12);
    border: 1px solid rgba(0,212,255,0.35); border-radius: 20px;
    padding: 4px 14px; font-size: 0.78rem; font-weight: 700;
    color: #00d4ff; letter-spacing: 0.8px; margin-bottom: 14px;
    text-transform: uppercase;
}

/* ── METRIC CARDS ── */
.stMetric {
    background: rgba(30,41,59,0.5) !important;
    backdrop-filter: blur(12px);
    padding: 22px !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: all 0.4s cubic-bezier(0.175,0.885,0.32,1.275);
    animation: float 6s ease-in-out infinite;
}
@keyframes float {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-5px); }
}
.stMetric:hover {
    transform: scale(1.04) translateY(-8px) !important;
    border: 1px solid rgba(0,212,255,0.5) !important;
    box-shadow: 0 0 28px rgba(0,212,255,0.25);
}
[data-testid="stMetricValue"] {
    background: linear-gradient(45deg, #00d4ff, #972db4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800 !important; font-size: 2.2rem !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(15,23,42,0.6);
    padding: 10px; border-radius: 20px; gap: 12px;
    border: 1px solid rgba(255,255,255,0.05);
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

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    background: rgba(30,41,59,0.3) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── PRODUCT CARDS ── */
.product-card {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px; padding: 22px; text-align: center;
    height: 320px; display: flex; flex-direction: column;
    justify-content: space-between; align-items: center;
    position: relative; overflow: hidden;
    transition: all 0.5s cubic-bezier(0.4,0,0.2,1);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
.product-card::before {
    content: ""; position: absolute;
    top:-50%;left:-50%;width:200%;height:200%;
    background: conic-gradient(transparent,rgba(0,212,255,0.15),transparent 30%);
    animation: rotate-glow 5s linear infinite; z-index:0;
}
.product-card::after {
    content:""; position:absolute; inset:2px;
    background:#0f172a; border-radius:20px; z-index:1;
}
@keyframes rotate-glow { from{transform:rotate(0deg);} to{transform:rotate(360deg);} }
.product-card img {
    position:relative;z-index:2;max-width:88%;max-height:165px;
    object-fit:contain;
    filter: drop-shadow(0 0 12px rgba(0,212,255,0.3));
    animation: float-img 4s ease-in-out infinite;
    transition: all 0.5s ease;
}
@keyframes float-img {
    0%,100%{transform:translateY(0) scale(1);}
    50%{transform:translateY(-8px) scale(1.04);}
}
.product-card:hover { transform: translateY(-14px) scale(1.02); border-color:#00d4ff; box-shadow:0 0 35px rgba(0,212,255,0.3); }
.product-card:hover img { filter: drop-shadow(0 0 22px rgba(0,212,255,0.7)); }
.product-name {
    position:relative;z-index:2;font-weight:700;font-size:1.05rem;
    background:linear-gradient(90deg,#00d4ff,#972db4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    text-transform:uppercase;letter-spacing:1.5px;margin:0;
}
.product-category {
    position:relative;z-index:2;font-size:0.72rem;font-weight:600;
    letter-spacing:1px;text-transform:uppercase;padding:3px 12px;
    border-radius:20px;margin-top:4px;
}
.cat-dth { background:rgba(0,212,255,0.12);color:#00d4ff;border:1px solid rgba(0,212,255,0.3); }
.cat-ott { background:rgba(151,45,180,0.12);color:#c084fc;border:1px solid rgba(151,45,180,0.3); }

/* ── INSIGHT CARDS ── */
.insight-card {
    background:rgba(29,110,239,0.07);
    border:1px solid rgba(29,110,239,0.25);
    border-radius:16px;padding:18px 22px;
    transition: all 0.3s ease;
}
.insight-card:hover { border-color:rgba(0,212,255,0.4); box-shadow:0 0 15px rgba(0,212,255,0.1); }

/* ── PIPELINE STEPS ── */
.pipe-step {
    background:rgba(30,41,59,0.5);border:1px solid rgba(255,255,255,0.08);
    border-radius:16px;padding:16px 18px;text-align:center;
    transition:all 0.3s ease;
}
.pipe-step:hover { border-color:rgba(0,212,255,0.35);transform:translateY(-4px); }
.pipe-num {
    width:36px;height:36px;border-radius:50%;
    background:linear-gradient(135deg,#1d6eef,#972db4);
    display:flex;align-items:center;justify-content:center;
    font-weight:800;font-size:1rem;color:white;
    margin:0 auto 10px auto;box-shadow:0 0 15px rgba(29,110,239,0.4);
}

/* SCROLLBAR */
::-webkit-scrollbar{height:8px;width:8px;}
::-webkit-scrollbar-thumb{background:linear-gradient(to bottom,#1d6eef,#972db4);border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD IMAGES
# =====================
def get_base64(path):
    with open(path,'rb') as f:
        return base64.b64encode(f.read()).decode()

@st.cache_resource
def load_images():
    paths = {
        "Nusantara": "assets/nusantara.jpeg",
        "Satellite" : "assets/satellite.jpg",
        "Seru"      : "assets/seru.jpeg",
        "2nd Gen"   : "assets/2nd.jpg",
    }
    return {k: (get_base64(v) if os.path.exists(v) else None) for k,v in paths.items()}

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    df_raw = pd.read_excel("data/dataset2k22-2k26.xlsx")
    df_processed, produk_cols = preprocessing_pipeline(df_raw.copy())
    df_train = get_training_data(df_processed, produk_cols)
    return df_raw, df_processed, df_train, produk_cols

df_raw, df_processed, df_train, produk_cols = load_data()
img_data = load_images()

# =====================
# HERO BANNER
# =====================
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">📡 Dashboard Analitik · Transvision 2026</div>
  <h1 class="hero-title">Pemetaan Potensi Pasar<br>Produk Transvision</h1>
  <p class="hero-sub">
    Platform berbasis <b>Hierarchical Logistic Regression</b> untuk mengklasifikasikan
    potensi pasar produk Transvision di seluruh wilayah Indonesia.
    Dari data historis 2022–2025, sistem memproyeksikan strategi distribusi optimal untuk <b>2026</b>.
  </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# =====================
# METRICS
# =====================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🌍 Total Records",    f"{len(df_processed):,}", help="Total data historis 2022–2025")
with col2:
    st.metric("🧠 Data Training",    f"{len(df_train):,}",    help="Baris berlabel untuk training model")
with col3:
    st.metric("🏝️ Cakupan Provinsi", f"{df_raw['Provinsi'].nunique()}")
with col4:
    st.metric("🎯 Target Proyeksi",  "2026")

st.write("")
st.divider()

# =====================
# PIPELINE SECTION
# =====================
st.subheader("⚙️ Alur Sistem Klasifikasi")
p1,p2,p3,p4,p5 = st.columns(5)
steps = [
    ("1","📥","Input Data","Data sosio-ekonomi kota/kab: PDRB, UMP, sinyal, household, pengeluaran."),
    ("2","⚙️","Preprocessing","Normalisasi, encoding TV Digital, pembersihan angka, pembuatan label."),
    ("3","🔬","Seleksi Fitur","Backward Wald mengeliminasi variabel tidak signifikan per provinsi."),
    ("4","🤖","Model Hierarkis","Multinomial → DTH/OTT/Hybrid. Lalu biner → produk spesifik."),
    ("5","🎯","Rekomendasi","Output: strategi + produk (Satellite/Nusantara/Seru/2nd Gen+Seru)."),
]
for col, (num, icon, title, desc) in zip([p1,p2,p3,p4,p5], steps):
    with col:
        st.markdown(f"""
        <div class="pipe-step">
          <div class="pipe-num">{num}</div>
          <div style="font-size:1.5rem;margin-bottom:6px;">{icon}</div>
          <div style="color:#f8fafc;font-weight:700;font-size:0.88rem;margin-bottom:6px;">{title}</div>
          <div style="color:#64748b;font-size:0.78rem;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")
st.divider()

# =====================
# TABS — DATA EXPLORATION
# =====================
st.subheader("🔎 Eksplorasi Dataset")

if 'sample_idx' not in st.session_state:
    st.session_state.sample_idx = df_raw.sample(10).index

sample_raw  = df_raw.loc[st.session_state.sample_idx].sort_index()
sample_proc = df_processed.loc[st.session_state.sample_idx].sort_index()

tab1, tab2, tab3 = st.tabs(["📑 Preview Data Asli", "⚙️ Data Terproses (ML)", "📊 Distribusi Historis"])

with tab1:
    c_refresh, _ = st.columns([1,4])
    with c_refresh:
        if st.button("🔄 Ambil Sampel Baru"):
            st.session_state.sample_idx = df_raw.sample(10).index
            st.rerun()
    st.caption("10 sampel acak dari dataset mentah.")
    st.dataframe(sample_raw, use_container_width=True,
                 column_config={"OTT": st.column_config.TextColumn("OTT",width="large"),
                                "DTH/ MASS": st.column_config.TextColumn("DTH/ MASS",width="large")})

with tab2:
    st.caption("Data yang sama setelah normalisasi, encoding, dan pelabelan.")
    st.dataframe(sample_proc, use_container_width=True,
                 column_config={
                     "label_kategori": st.column_config.NumberColumn("Label Multinomial",width="medium",
                         help="1=DTH Only · 2=OTT Only · 3=Hybrid"),
                     "label_dth": st.column_config.NumberColumn("Sub-Label DTH",width="medium",
                         help="0=Satellite · 1=Nusantara"),
                     "label_ott": st.column_config.NumberColumn("Sub-Label OTT",width="medium",
                         help="0=Seru · 1=2nd Gen"),
                 })

with tab3:
    label_map_disp = {1.0:"DTH Only",2.0:"OTT Only",3.0:"Hybrid Market"}
    list_tahun = sorted(df_train["Tahun"].unique())
    opsi_tahun = ["Semua Tahun"] + [str(t) for t in list_tahun]
    sel_tahun  = st.selectbox("📅 Pilih Tahun:", opsi_tahun)
    df_fil     = df_train if sel_tahun=="Semua Tahun" else df_train[df_train["Tahun"]==int(sel_tahun)]

    df_stats = df_fil["label_kategori"].map(label_map_disp).value_counts().reset_index()
    df_stats.columns = ["Kategori","Jumlah"]

    ch1, ch2 = st.columns([5,3])
    with ch1:
        fig_pie = px.pie(df_stats, values="Jumlah", names="Kategori", hole=0.52,
                         template="plotly_dark",
                         color_discrete_sequence=["#00d4ff","#c084fc","#60a5fa"],
                         title=f"Proporsi Strategi Wilayah — {sel_tahun}")
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color="#f8fafc"))
        st.plotly_chart(fig_pie, use_container_width=True)
    with ch2:
        st.write("")
        total = df_stats["Jumlah"].sum()
        avg_ump  = df_fil["UMP (Rupiah)"].mean() if "UMP (Rupiah)" in df_fil.columns else 0
        avg_pdrb = df_fil["PDRB (Ribu Rp)"].mean() if "PDRB (Ribu Rp)" in df_fil.columns else 0
        st.markdown(f"""
        <div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);
                    border-radius:14px;padding:18px 20px;">
          <p style="color:#64748b;font-size:0.72rem;font-weight:700;letter-spacing:1px;
                    text-transform:uppercase;margin:0 0 10px 0;">Summary {sel_tahun}</p>
          <p style="color:#f8fafc;font-size:0.9rem;margin:0 0 6px 0;">
            📍 Wilayah Terpetakan: <b style="color:#00d4ff;">{total}</b>
          </p>
          <p style="color:#f8fafc;font-size:0.9rem;margin:0 0 6px 0;">
            💰 Rata-rata UMP: <b style="color:#00d4ff;">Rp {avg_ump:,.0f}</b>
          </p>
          <p style="color:#f8fafc;font-size:0.9rem;margin:0 0 10px 0;">
            📈 Rata-rata PDRB: <b style="color:#00d4ff;">{avg_pdrb:,.2f}</b>
          </p>
          <p style="color:#94a3b8;font-size:0.8rem;line-height:1.6;margin:0;">
            Data historis ini menjadi acuan model untuk mengenali pola wilayah
            dan memproyeksikan strategi 2026.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    # Bar chart distribusi produk
    if "DTH/ MASS" in df_fil.columns and "OTT" in df_fil.columns:
        c_sat  = df_fil["DTH/ MASS"].str.contains("satellite",  na=False).sum()
        c_nus  = df_fil["DTH/ MASS"].str.contains("nusantara",  na=False).sum()
        c_seru = df_fil["OTT"].str.contains("seru",             na=False).sum()
        c_2nd  = df_fil["OTT"].str.contains("2nd gen",          na=False).sum()
        df_prod = pd.DataFrame({
            "Produk" : ["Satellite","Nusantara","Seru","2nd Gen"],
            "Jumlah" : [c_sat,c_nus,c_seru,c_2nd],
            "Kategori": ["DTH","DTH","OTT","OTT"],
        }).sort_values("Jumlah",ascending=True)
        fig_bar = px.bar(df_prod,x="Jumlah",y="Produk",orientation='h',
                         text="Jumlah",color="Kategori",
                         color_discrete_map={"DTH":"#00d4ff","OTT":"#c084fc"},
                         title=f"Distribusi Produk Historis ({sel_tahun})",
                         template="plotly_dark")
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                              showlegend=True,legend_title_text="Kategori",
                              margin=dict(l=10,r=30,t=40,b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# =====================
# PRODUCT SHOWCASE
# =====================
st.subheader("🎯 Lini Produk Transvision")

produk_info = [
    ("Nusantara", "DTH", "Paket DTH premium dengan channel eksklusif untuk segmen menengah-atas."),
    ("Satellite",  "DTH", "Layanan DTH berbasis satelit untuk wilayah dengan keterbatasan infrastruktur internet."),
    ("Seru",       "OTT", "Platform streaming OTT entry-level, konten lokal dan hiburan keluarga."),
    ("2nd Gen",    "OTT", "OTT generasi terkini — bundel wajib dengan Seru sebagai paket lengkap."),
]

cols = st.columns(4)
for i,(produk, cat, desc) in enumerate(produk_info):
    with cols[i]:
        cat_class = "cat-dth" if cat=="DTH" else "cat-ott"
        img_src   = f'data:image/jpeg;base64,{img_data[produk]}' if img_data.get(produk) else ""
        img_tag   = f'<img src="{img_src}" alt="{produk}">' if img_src else f'<div style="font-size:3rem;z-index:2;position:relative;">📦</div>'
        st.markdown(f"""
        <div class="product-card">
          {img_tag}
          <div style="position:relative;z-index:2;width:100%;">
            <p class="product-name">{produk}</p>
            <span class="product-category {cat_class}">{cat}</span>
            <p style="color:#64748b;font-size:0.75rem;margin:8px 0 0 0;line-height:1.5;">{desc}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.write("")
st.divider()

# =====================
# QUICK INSIGHT CARDS
# =====================
st.subheader("💡 Insight Cepat Dataset")
ic1, ic2, ic3 = st.columns(3)

prov_terbanyak = df_raw.groupby("Provinsi").size().idxmax() if "Provinsi" in df_raw.columns else "N/A"
tahun_terbaru  = df_raw["Tahun"].max() if "Tahun" in df_raw.columns else "N/A"
pct_labeled    = f"{len(df_train)/len(df_processed)*100:.1f}%" if len(df_processed)>0 else "0%"

with ic1:
    st.markdown(f"""
    <div class="insight-card">
      <p style="color:#64748b;font-size:0.72rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin:0 0 8px 0;">📍 Provinsi Data Terbanyak</p>
      <p style="color:#00d4ff;font-size:1.3rem;font-weight:700;margin:0 0 6px 0;">{prov_terbanyak}</p>
      <p style="color:#94a3b8;font-size:0.82rem;margin:0;">Provinsi dengan jumlah kab/kota terlengkap dalam dataset historis.</p>
    </div>""", unsafe_allow_html=True)
with ic2:
    st.markdown(f"""
    <div class="insight-card">
      <p style="color:#64748b;font-size:0.72rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin:0 0 8px 0;">🏷️ Coverage Label</p>
      <p style="color:#c084fc;font-size:1.3rem;font-weight:700;margin:0 0 6px 0;">{pct_labeled}</p>
      <p style="color:#94a3b8;font-size:0.82rem;margin:0;">Proporsi data yang sudah berlabel dan siap digunakan untuk training model.</p>
    </div>""", unsafe_allow_html=True)
with ic3:
    st.markdown(f"""
    <div class="insight-card">
      <p style="color:#64748b;font-size:0.72rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin:0 0 8px 0;">📅 Data Terkini</p>
      <p style="color:#60a5fa;font-size:1.3rem;font-weight:700;margin:0 0 6px 0;">Tahun {tahun_terbaru}</p>
      <p style="color:#94a3b8;font-size:0.82rem;margin:0;">Tahun observasi terakhir dalam dataset. Model diproyeksikan ke 2026.</p>
    </div>""", unsafe_allow_html=True)