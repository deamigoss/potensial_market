import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from utils.preprocessing import preprocessing_pipeline, get_training_data
from components.sidebar import sidebar

st.set_page_config(page_title="Model Performance", layout="wide", page_icon="⚙️")
sidebar()

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at top left,#1e293b,#0f172a);color:#f8fafc;}
header[data-testid="stHeader"]{background-color:rgba(0,0,0,0)!important;border-bottom:none!important;box-shadow:none!important;}
[data-testid="stSidebar"]{border-right:none!important;}
.block-container{padding-top:0rem!important;margin-top:-20px!important;}
[data-testid="stDecoration"]{display:none!important;}
.stTabs [data-baseweb="tab-list"]{background-color:rgba(15,23,42,0.6);padding:10px;border-radius:20px;gap:15px;border:1px solid rgba(255,255,255,0.05);}
.stTabs [data-baseweb="tab"]{height:50px;background-color:transparent;border-radius:12px;color:#94a3b8;border:none;transition:all 0.4s;font-weight:600;}
.stTabs [aria-selected="true"]{background:linear-gradient(90deg,#1d6eef,#972db4)!important;color:white!important;box-shadow:0 5px 20px rgba(29,110,239,0.4);transform:translateY(-2px);}
[data-testid="stDataFrame"]{background:rgba(30,41,59,0.3)!important;border-radius:20px!important;border:1px solid rgba(255,255,255,0.1)!important;padding:10px;}
::-webkit-scrollbar{height:8px;width:8px;}
::-webkit-scrollbar-thumb{background:linear-gradient(to bottom,#1d6eef,#972db4);border-radius:10px;}
.metric-card{background:rgba(30,41,59,0.6);backdrop-filter:blur(15px);border:1px solid rgba(0,212,255,0.25);border-radius:20px;padding:22px 15px;text-align:center;box-shadow:0 10px 30px rgba(0,0,0,0.3);transition:all 0.4s ease-in-out;}
.metric-card:hover{transform:translateY(-8px);border:1px solid #00d4ff;box-shadow:0 0 25px rgba(0,212,255,0.2);}
.metric-label{color:#94a3b8;font-size:0.85rem;font-weight:600;display:block;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px;}
.metric-value{font-size:2.4rem;font-weight:850;margin:0;background:linear-gradient(90deg,#00d4ff,#972db4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.2;}
.metric-sub{color:#64748b;font-size:0.78rem;margin-top:6px;display:block;}
.insight-box{background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.25);border-radius:14px;padding:16px 20px;margin-top:12px;}
.insight-box p{margin:0;color:#e2e8f0;font-size:0.93rem;line-height:1.75;}
.warn-box{background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);border-radius:14px;padding:16px 20px;margin-top:12px;}
.warn-box p{margin:0;color:#fde68a;font-size:0.93rem;line-height:1.75;}
.flow-step{background:rgba(29,110,239,0.08);border:1px solid rgba(29,110,239,0.3);border-radius:14px;padding:16px 18px;}
.flow-step h4{color:#60a5fa;margin:0 0 6px 0;font-size:0.95rem;}
.flow-step p{color:#cbd5e1;margin:0;font-size:0.86rem;line-height:1.6;}
.status-badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:0.78rem;font-weight:700;letter-spacing:0.5px;}
.badge-ready{background:rgba(34,197,94,0.15);color:#4ade80;border:1px solid rgba(34,197,94,0.35);}
.badge-missing{background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.35);}
.model-badge{background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.3);border-radius:12px;height:45px;display:flex;align-items:center;justify-content:center;padding:0 20px;}
.model-badge-text{color:#00d4ff;font-size:0.92rem;font-weight:600;margin:0;}
.stExpander{background:rgba(30,41,59,0.4)!important;border:1px solid rgba(0,212,255,0.2)!important;border-radius:12px!important;}
</style>
""", unsafe_allow_html=True)

COL_HH     = "Jumlah Household"
COL_SINYAL = "Jumlah Desa/Kelurahan Yang Menerima Sinyal 5G/4G/LTE (Persen)"
COL_UMP    = "UMP (Rupiah)"
COL_PDRB   = "PDRB (Ribu Rp)"
COL_EXP    = "Rata-Rata Pengeluaran per Kapita Bukan Makanan (Rupiah)"
COL_TV     = "Tercover TV Digital?"
feature_cols = [COL_HH, COL_SINYAL, COL_UMP, COL_PDRB, COL_EXP, COL_TV]
label_map    = {1: "DTH", 2: "OTT", 3: "Hybrid"}
label_color  = {"DTH": "#00d4ff", "OTT": "#c084fc", "Hybrid": "#60a5fa"}
FEATURE_BISNIS = {
    COL_HH    : ("🏠","Jumlah Rumah Tangga","Proxy ukuran pasar. Semakin banyak household, semakin besar potensi pelanggan."),
    COL_SINYAL: ("📶","Penetrasi Sinyal 4G/5G","Infrastruktur internet kuat mendorong adopsi produk OTT berbasis streaming."),
    COL_UMP   : ("💰","Upah Minimum Provinsi","Mencerminkan daya beli — UMP tinggi lebih receptive terhadap produk premium."),
    COL_PDRB  : ("📈","PDRB per Kapita","Kemakmuran ekonomi wilayah. PDRB tinggi berkorelasi dengan konsumsi hiburan lebih tinggi."),
    COL_EXP   : ("🛒","Pengeluaran Non-Makanan","Proxy langsung kesediaan membayar layanan hiburan & berlangganan TV."),
    COL_TV    : ("📺","Cakupan TV Digital","Wilayah TV Digital lebih siap menerima produk DTH dan memiliki infrastruktur media matang."),
}

def backward_wald(X, y, threshold=0.05):
    X_const   = sm.add_constant(X)
    variables = list(X_const.columns)
    while True:
        mdl     = sm.Logit(y, X_const[variables]).fit(disp=0)
        pvalues = mdl.pvalues
        max_p   = pvalues.max()
        if max_p > threshold:
            remove_var = pvalues.idxmax()
            if remove_var == "const":
                break
            variables.remove(remove_var)
        else:
            break
    return variables

def build_model_pack(df_scope, df_train_full, vars_strategi, scaler_strategi, model_strategi, provinsi_name):
    df_dth = df_scope[df_scope["label_dth"].notna()]
    if len(df_dth["label_dth"].unique()) >= 2:
        X_dth, y_dth, dth_src = df_dth[vars_strategi], df_dth["label_dth"], "lokal"
    else:
        df_nas = df_train_full[df_train_full["label_dth"].notna()]
        X_dth, y_dth, dth_src = df_nas[vars_strategi], df_nas["label_dth"], "nasional"
    sc_dth  = StandardScaler()
    mdl_dth = LogisticRegression(max_iter=2000).fit(sc_dth.fit_transform(X_dth), y_dth)
    df_ott = df_scope[df_scope["label_ott"].notna()]
    if len(df_ott["label_ott"].unique()) >= 2:
        X_ott, y_ott, ott_src = df_ott[vars_strategi], df_ott["label_ott"], "lokal"
    else:
        df_nas = df_train_full[df_train_full["label_ott"].notna()]
        X_ott, y_ott, ott_src = df_nas[vars_strategi], df_nas["label_ott"], "nasional"
    sc_ott  = StandardScaler()
    mdl_ott = LogisticRegression(max_iter=2000).fit(sc_ott.fit_transform(X_ott), y_ott)
    return {
        'model_strategi': model_strategi, 'scaler_strategi': scaler_strategi,
        'model_produk_dth': mdl_dth, 'scaler_dth': sc_dth, 'label_map_dth': {0:"Satellite",1:"Nusantara"},
        'model_produk_ott': mdl_ott, 'scaler_ott': sc_ott, 'label_map_ott': {0:"Seru",1:"2nd Gen"},
        'selected_vars': vars_strategi, 'label_map_strategi': label_map,
        'provinsi_terpilih': provinsi_name, 'dth_source': dth_src, 'ott_source': ott_src,
    }

@st.cache_resource(show_spinner=False)
def pretrain_all_models():
    df_raw          = pd.read_excel("data/dataset2k22-2k26.xlsx")
    df_processed, _ = preprocessing_pipeline(df_raw.copy())
    df_train        = get_training_data(df_processed, None)
    all_models      = {}
    list_prov       = list(df_train["Provinsi"].dropna().unique())
    scopes          = [("Seluruh Indonesia", df_train)] + [(p, df_train[df_train["Provinsi"]==p]) for p in list_prov]
    for prov_name, df_scope in scopes:
        if df_scope.empty or len(df_scope["label_kategori"].unique()) < 2:
            continue
        X = df_scope[feature_cols]
        y = df_scope["label_kategori"]
        try:
            vars_sel = backward_wald(X, y)
            if "const" in vars_sel: vars_sel.remove("const")
            if not vars_sel: vars_sel = feature_cols
        except:
            vars_sel = feature_cols
        sc  = StandardScaler()
        mdl = LogisticRegression(solver="lbfgs", max_iter=2000)
        mdl.fit(sc.fit_transform(X[vars_sel]), y)
        all_models[prov_name] = build_model_pack(
            df_scope=df_scope, df_train_full=df_train,
            vars_strategi=vars_sel, scaler_strategi=sc,
            model_strategi=mdl, provinsi_name=prov_name
        )
    return all_models, df_train

# ── RENDER ───────────────────────────────────────────────────────────────────
st.title("🤖 Model Performance & Evaluasi")

st.markdown("""
<div style="background:rgba(30,41,59,0.45);padding:22px 26px;border-radius:18px;
            border-left:5px solid #1d6eef;border:1px solid rgba(255,255,255,0.06);margin-bottom:8px;">
  <p style="margin:0 0 8px 0;font-size:1.1rem;font-weight:700;color:#f8fafc;">🧠 Sistem Pre-Training Otomatis</p>
  <p style="margin:0;font-size:0.93rem;color:#cbd5e1;line-height:1.75;">
    Semua model provinsi dilatih <b>otomatis saat aplikasi pertama dibuka</b>.<br>
    Model disimpan di <b>memory (cache)</b> dan tetap tersedia selama sesi aktif.<br>
    Sistem menggunakan <b>Hierarchical Logistic Regression</b> dengan seleksi fitur <b>Backward Wald</b>:
    hanya variabel yang signifikan secara statistik yang masuk ke model setiap provinsi.
  </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("⚙️ Melatih semua model provinsi... (hanya sekali saat startup)"):
    all_models, df_train = pretrain_all_models()

st.success(f"✅ **{len(all_models)} model** berhasil dilatih dan siap digunakan (Nasional + per-Provinsi).")
st.divider()

# ── ALUR ─────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Alur Klasifikasi Hierarkis")
c1, a1, c2, a2, c3 = st.columns([2,0.3,2,0.3,2])
with c1:
    st.markdown("""<div class="flow-step"><h4>① Input Wilayah</h4>
    <p>Data sosial-ekonomi kota/kab: PDRB, UMP, sinyal, household, pengeluaran, TV digital.</p></div>""", unsafe_allow_html=True)
with a1:
    st.markdown("<div style='text-align:center;font-size:1.8rem;color:#1d6eef;padding-top:18px;'>→</div>", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="flow-step" style="border-color:rgba(151,45,180,0.4);background:rgba(151,45,180,0.07);">
    <h4 style="color:#c084fc;">② Model Strategi (Multinomial)</h4>
    <p>Klasifikasi ke <b>DTH</b>, <b>OTT</b>, atau <b>Hybrid</b> berdasarkan probabilitas tertinggi dari 3 kelas.</p></div>""", unsafe_allow_html=True)
with a2:
    st.markdown("<div style='text-align:center;font-size:1.8rem;color:#972db4;padding-top:18px;'>→</div>", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="flow-step" style="border-color:rgba(96,165,250,0.4);background:rgba(96,165,250,0.07);">
    <h4 style="color:#93c5fd;">③ Model Produk (Biner)</h4>
    <p>DTH → Satellite vs Nusantara<br>OTT → Seru vs 2nd Gen (+Seru jika 2nd Gen menang)<br>Hybrid → DTH & OTT digabung.</p></div>""", unsafe_allow_html=True)
st.divider()

# ── FILTER PROVINSI (VERSI PIXEL PERFECT) ─────────────────────────────────────

# Gunakan vertical_alignment="top" karena kita akan atur margin manual agar presisi
col_sel, col_badge = st.columns([2.5, 1], vertical_alignment="top")

with col_sel:
    list_prov_ui = ["Seluruh Indonesia"] + sorted(df_train["Provinsi"].unique())
    selected_prov = st.selectbox(
        "📍 Pilih Wilayah Evaluasi", 
        list_prov_ui, 
        key="prov_selector"
    )

with col_badge:
    in_cache = selected_prov in all_models
    badge_cls = "badge-ready" if in_cache else "badge-missing"
    badge_txt = "Model Tersedia" if in_cache else "Tidak Tersedia"
    icon = "✅" if in_cache else "❌"
    
    # margin-top: 28px adalah angka 'magic' untuk sejajar dengan input berlabel
    st.markdown(f"""
        <div style="
            display: flex; 
            align-items: center; 
            height: 45px; 
            margin-top: 28px;
            background: rgba(0, 212, 255, 0.05); 
            border: 1px solid rgba(0, 212, 255, 0.3); 
            border-radius: 12px; 
            padding: 0 15px;
            width: fit-content;
        ">
            <p style="
                margin: 0; 
                color: #00d4ff; 
                font-size: 0.85rem; 
                font-weight: 600; 
                display: flex; 
                align-items: center; 
                gap: 12px;
                white-space: nowrap;
            ">
                <span style="display: flex; align-items: center; gap: 5px;">
                    🧠 Backward Wald
                </span>
                <span class="status-badge {badge_cls}" style="
                    padding: 3px 10px; 
                    font-size: 0.75rem;
                    border-radius: 8px;
                    display: inline-block;
                    line-height: 1;
                ">
                    {icon} {badge_txt}
                </span>
            </p>
        </div>
    """, unsafe_allow_html=True)

if selected_prov not in all_models:
    st.error(f"Model untuk **{selected_prov}** tidak tersedia (data terlalu sedikit atau hanya satu kelas).")
    st.stop()

assets        = all_models[selected_prov]
selected_vars = assets['selected_vars']
df_eval       = df_train if selected_prov=="Seluruh Indonesia" else df_train[df_train["Provinsi"]==selected_prov]
X             = df_eval[selected_vars]
y             = df_eval["label_kategori"]
num_classes   = len(y.unique())
X_scaled      = assets['scaler_strategi'].transform(X)
y_pred        = assets['model_strategi'].predict(X_scaled)
acc           = accuracy_score(y, y_pred)
present_ids   = sorted(y.unique())
labels_str    = " · ".join([label_map[i] for i in present_ids if i in label_map])

# ── METRICS ───────────────────────────────────────────────────────────────────
st.write("")
for col, lbl, val, sub in zip(
    st.columns(4),
    ["🎯 Akurasi Model","📊 Data Evaluasi","🏷️ Kelas Strategi","🔬 Fitur Aktif"],
    [f"{acc:.1%}", f"{len(df_eval)}", f"{num_classes}", f"{len(selected_vars)}"],
    [f"Ketepatan prediksi di {selected_prov}", f"Kab/Kota yang dievaluasi", labels_str, "Terpilih via Backward Wald"]
):
    with col:
        st.markdown(f"""<div class="metric-card">
          <span class="metric-label">{lbl}</span>
          <p class="metric-value">{val}</p>
          <span class="metric-sub">{sub}</span></div>""", unsafe_allow_html=True)

st.write("")
acc_nar = (f"🟢 Akurasi **{acc:.1%}** — Model sangat handal. Rekomendasi dapat dipercaya dengan confidence tinggi." if acc>=0.90
           else f"🟡 Akurasi **{acc:.1%}** — Model cukup baik. Disarankan dikombinasikan dengan validasi lapangan." if acc>=0.75
           else f"🔴 Akurasi **{acc:.1%}** — Perlu perhatian. Data wilayah ini mungkin terlalu sedikit atau pola belum konsisten.")
st.markdown(f'<div class="insight-box"><p>📌 <b>Interpretasi Bisnis:</b> {acc_nar}</p></div>', unsafe_allow_html=True)
st.divider()

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Confusion Matrix & Performa Kelas","🔬 Variabel Penentu (Koefisien & Odds Ratio)","📐 Persamaan Model"])

with tab1:
    st.markdown("### 📊 Confusion Matrix")
    st.markdown("""<div class="insight-box" style="margin-bottom:18px;"><p>
    <b>Cara membaca:</b> Baris = kondisi <i>aktual</i> di lapangan, Kolom = hasil <i>prediksi</i> model.<br>
    ✅ Angka di <b>diagonal</b> = prediksi <b>benar</b>. ❌ Angka di <b>luar diagonal</b> = <b>kesalahan klasifikasi</b>.<br>
    Angka dalam kurung (%) = proporsi dari total aktual kelas tersebut (recall visual).
    </p></div>""", unsafe_allow_html=True)

    labels_cm = [label_map[i] for i in sorted(y.unique())]
    cm        = confusion_matrix(y, y_pred)
    cm_pct    = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot     = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.0f}%)" for j in range(len(labels_cm))] for i in range(len(labels_cm))]
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels_cm,
        y=labels_cm,
        text=annot,
        texttemplate="%{text}",
    colorscale=[
        [0, "#0f172a"],
        [0.3, "#1d4ed8"],
        [0.6, "#3b82f6"],
        [1, "#93c5fd"]
    ],   
        zmin=0,
        zmax=cm.max(),

        xgap=3,
        ygap=3,
        
        textfont={"color": "white", "size": 13},

        showscale=False
    ))

    fig_cm.update_layout(
        xaxis_title="Prediksi Model",
        yaxis_title="Kondisi Aktual",
    
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    
        font=dict(color="#f8fafc", size=13),

        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),

        height=360,
        margin=dict(l=20, r=20, t=10, b=20)
    )
    
    col_cm, col_cm_info = st.columns([3,2])
    with col_cm:
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_cm_info:
        st.markdown("#### 🏷️ Performa per Strategi")
        report = classification_report(y, y_pred, output_dict=True)
        for cls_id in present_ids:
            cls_key = str(float(cls_id))
            cls_name= label_map[cls_id]
            clr     = label_color[cls_name]
            if cls_key in report:
                r = report[cls_key]
                prec, rec, f1 = r['precision'], r['recall'], r['f1-score']
                rec_nar = ("Jarang melewatkan wilayah ini." if rec>=0.85
                           else "Kadang melewatkan sebagian wilayah." if rec>=0.65
                           else "Sering melewatkan — perlu perhatian.")
                st.markdown(f"""<div style="background:rgba(30,41,59,0.55);border-left:4px solid {clr};
                    border-radius:10px;padding:11px 14px;margin-bottom:10px;">
                  <b style="color:{clr};font-size:0.97rem;">{cls_name}</b><br>
                  <span style="color:#94a3b8;font-size:0.8rem;">Precision {prec:.0%} · Recall {rec:.0%} · F1 {f1:.0%}</span><br>
                  <span style="color:#cbd5e1;font-size:0.83rem;">{rec_nar}</span></div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📋 Tabel Performa Lengkap")
    with st.expander("📖 Panduan membaca tabel (klik untuk buka)"):
        st.markdown("""<div class="insight-box"><p>
        <b style="color:#00d4ff;">Precision</b> — Dari semua yang diprediksi DTH, berapa % yang benar DTH? Tinggi = jarang salah jarang salah klasifikasi.<br><br>
        <b style="color:#00d4ff;">Recall</b> — Dari semua yang sebenarnya DTH, berapa % yang berhasil dideteksi? Tinggi = tidak banyak melewatkan.<br><br>
        <b style="color:#00d4ff;">F1-Score</b> — Rata-rata harmonis Precision & Recall. Gunakan ini jika jumlah data antar kelas tidak merata.<br><br>
        <b style="color:#00d4ff;">Support</b> — Jumlah data aktual per kelas. Jika kurang dari 10, metrik kurang representatif.
        </p></div>""", unsafe_allow_html=True)
    report_df = pd.DataFrame(report).transpose()
    new_index = {str(float(k)): v for k, v in label_map.items()}
    report_df.index = [new_index.get(x,x) for x in report_df.index]
    st.dataframe(
        report_df.style
        .background_gradient(cmap='Blues',subset=['precision','recall','f1-score'],low=0.2,high=0.8)
        .format("{:.3f}",subset=['precision','recall','f1-score'])
        .format("{:.0f}",subset=['support']),
        use_container_width=True
    )

with tab2:
    st.markdown("### 🔬 Variabel yang Menentukan Strategi Wilayah")
    st.markdown("""<div class="insight-box" style="margin-bottom:18px;"><p>
    <b>Koefisien positif (biru)</b> → variabel ini menaikkan peluang dipilih strategi tersebut.<br>
    <b>Koefisien negatif (merah)</b> → variabel ini menurunkan peluang.<br>
    Variabel tidak muncul = tidak signifikan secara statistik, sudah dieliminasi Backward Wald.<br>
    <b>Odds Ratio (OR)</b> = exp(koefisien). OR 2.5 artinya kenaikan 1 SD variabel membuat peluang 2.5x lebih tinggi.
    </p></div>""", unsafe_allow_html=True)

    classes = assets['model_strategi'].classes_
    if len(classes) > 2:
        ref_label       = label_map[classes[0]]
        target_options  = [label_map[c] for c in classes[1:]]
        selected_target = st.selectbox("Tampilkan pengaruh variabel terhadap peluang strategi:", target_options,
                                       help=f"Koefisien relatif terhadap referensi: {ref_label}")
        target_idx  = [i for i,c in enumerate(classes) if label_map[c]==selected_target][0]
        coef_values = assets['model_strategi'].coef_[target_idx]
    else:
        selected_target = label_map[classes[1]]
        ref_label       = label_map[classes[0]]
        st.info(f"Model biner: **{selected_target}** vs **{ref_label}**")
        coef_values = assets['model_strategi'].coef_[0]

    short_names = {v: FEATURE_BISNIS[v][1] if v in FEATURE_BISNIS else v for v in selected_vars}
    coef_df = pd.DataFrame({
        "Variabel": selected_vars, "Label": [short_names[v] for v in selected_vars],
        "Koefisien": coef_values, "Odds Ratio": np.exp(coef_values)
    }).sort_values("Koefisien", ascending=True)

    fig_coef = px.bar(coef_df, x="Koefisien", y="Label", orientation="h",
                      color="Koefisien", color_continuous_scale="RdBu", template="plotly_dark",
                      hover_data={"Odds Ratio":":.3f","Koefisien":":.4f","Label":False})
    fig_coef.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font=dict(color="#94a3b8"), height=360, coloraxis_showscale=False,
                           margin=dict(l=10,r=20,t=10,b=10), yaxis_title="")
    fig_coef.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("#### 💡 Interpretasi Bisnis")
    top_pos = coef_df[coef_df["Koefisien"]>0].nlargest(3,"Koefisien")
    top_neg = coef_df[coef_df["Koefisien"]<0].nsmallest(3,"Koefisien")
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown(f"**📈 Pendorong utama → {selected_target}:**")
        if top_pos.empty: st.caption("Tidak ada pendorong positif signifikan.")
        for _, row in top_pos.iterrows():
            icon,nama,desc = FEATURE_BISNIS.get(row["Variabel"],("📌",row["Variabel"],""))
            or_val = row["Odds Ratio"]
            st.markdown(f"""<div style="background:rgba(29,110,239,0.09);border:1px solid rgba(29,110,239,0.28);
                border-radius:10px;padding:12px 14px;margin-bottom:10px;">
              <b style="color:#60a5fa;">{icon} {nama}</b><br>
              <span style="color:#94a3b8;font-size:0.8rem;">OR = <b style="color:#f8fafc;">{or_val:.2f}x</b>
                — kenaikan 1 SD membuat peluang {selected_target} jadi {or_val:.2f}x lebih tinggi.</span><br>
              <span style="color:#cbd5e1;font-size:0.82rem;">{desc}</span></div>""", unsafe_allow_html=True)
    with col_neg:
        st.markdown(f"**📉 Penghambat strategi {selected_target}:**")
        if top_neg.empty: st.caption("Tidak ada penghambat signifikan.")
        for _, row in top_neg.iterrows():
            icon,nama,desc = FEATURE_BISNIS.get(row["Variabel"],("📌",row["Variabel"],""))
            or_val = row["Odds Ratio"]
            st.markdown(f"""<div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.28);
                border-radius:10px;padding:12px 14px;margin-bottom:10px;">
              <b style="color:#f87171;">{icon} {nama}</b><br>
              <span style="color:#94a3b8;font-size:0.8rem;">OR = <b style="color:#f8fafc;">{or_val:.2f}x</b>
                — wilayah dengan {nama.lower()} tinggi cenderung bukan {selected_target}.</span><br>
              <span style="color:#cbd5e1;font-size:0.82rem;">{desc}</span></div>""", unsafe_allow_html=True)

    with st.expander("📊 Tabel Lengkap Koefisien & Odds Ratio"):
        st.markdown("""<div class="insight-box"><p>
        <b style="color:#00d4ff;">Odds Ratio</b> = exp(koefisien). OR > 1 meningkatkan peluang, OR < 1 menurunkan.
        Contoh OR 2.5: kenaikan 1 satuan variabel membuat peluang 2.5x lebih besar.</p></div>""", unsafe_allow_html=True)
        st.dataframe(
            coef_df[["Label","Variabel","Koefisien","Odds Ratio"]].sort_values("Koefisien",ascending=False)
            .style.background_gradient(cmap="RdBu",subset=["Koefisien"])
            .format({"Koefisien":"{:.4f}","Odds Ratio":"{:.3f}"}),
            use_container_width=True
        )

with tab3:
    st.markdown("### 📐 Persamaan Matematis Model")
    st.markdown("""<div class="insight-box" style="margin-bottom:18px;"><p>
    Persamaan ini adalah rumus yang digunakan model untuk menghitung probabilitas setiap strategi.<br>
    Nilai variabel wilayah (sudah dinormalisasi / z-score) dimasukkan ke rumus ini dan menghasilkan peluang 0-100%.<br>
    Strategi dengan peluang tertinggi yang menjadi rekomendasi akhir.
    </p></div>""", unsafe_allow_html=True)

    var_latex = {COL_HH:"HH", COL_SINYAL:"Signal", COL_UMP:"UMP",
                 COL_PDRB:"PDRB", COL_EXP:"Exp", COL_TV:"TV"}
    def lvar(v): return var_latex.get(v, v[:8].replace(" ",""))

    intercepts = assets['model_strategi'].intercept_
    coefs_mat  = assets['model_strategi'].coef_
    classes    = assets['model_strategi'].classes_

    if len(classes) == 2:
        lbl_target  = label_map[classes[1]]
        linear_part = f"{intercepts[0]:.3f}"
        for var,c in zip(selected_vars, coefs_mat[0]):
            sign = "+" if c>=0 else "-"
            linear_part += f" {sign} {abs(c):.3f}\\cdot {lvar(var)}"
        st.latex(r'g(x) = ' + linear_part)
        st.latex(r'\pi_{\text{'+lbl_target+r'}}(x)=\frac{e^{g(x)}}{1+e^{g(x)}}')
        st.info(f"Model biner — satu persamaan untuk strategi **{lbl_target}**")
    else:
        ref_label   = label_map[classes[0]]
        g_functions = []
        st.info(f"Kategori Referensi (pi_0): **{ref_label}**")
        for i, cls in enumerate(classes):
            if i==0: continue
            lp = f"{intercepts[i]:.3f}"
            for var,c in zip(selected_vars, coefs_mat[i]):
                sign = "+" if c>=0 else "-"
                lp  += f" {sign} {abs(c):.3f}\\cdot {lvar(var)}"
            g_functions.append((label_map[cls], lp))
        denom = "1 + " + " + ".join([f"e^{{g_{j+1}(x)}}" for j in range(len(g_functions))])
        st.latex(r"\pi_0(x) = \frac{1}{" + denom + r"}")
        for i,(cls_name,g_x) in enumerate(g_functions):
            idx = i+1
            st.latex(r"g_"+str(idx)+r"(x) = "+g_x)
            st.latex(r"\pi_"+str(idx)+r"(x) = \frac{e^{g_"+str(idx)+r"(x)}}{"+denom+r"}")
            st.caption(f"pi_{idx} = Peluang strategi **{cls_name}**")

    st.markdown("#### 📖 Keterangan Variabel")
    leg_cols = st.columns(3)
    for i,var in enumerate(selected_vars):
        icon,nama,_ = FEATURE_BISNIS.get(var,("📌",var,""))
        with leg_cols[i%3]:
            st.markdown(f"""<div style="background:rgba(30,41,59,0.5);border-radius:8px;padding:8px 12px;margin-bottom:8px;">
              <span style="color:#00d4ff;font-weight:700;">{lvar(var)}</span>
              <span style="color:#94a3b8;font-size:0.85rem;"> = {icon} {nama}</span></div>""", unsafe_allow_html=True)

    eliminated = [v for v in feature_cols if v not in selected_vars]
    if eliminated:
        el_names = ", ".join([FEATURE_BISNIS.get(v,("","",v))[1] for v in eliminated])
        st.markdown(f"""<div class="warn-box"><p>
        ⚠️ <b>Variabel dieliminasi Backward Wald</b> (tidak signifikan di wilayah ini): {el_names}.<br>
        Dihapus agar model tidak terganggu noise — tidak berarti variabel ini tidak penting secara bisnis,
        hanya tidak cukup membedakan strategi di wilayah {selected_prov} berdasarkan data historis.
        </p></div>""", unsafe_allow_html=True)

st.divider()

# ── STATUS SEMUA MODEL ────────────────────────────────────────────────────────
st.subheader("📦 Status Model Semua Provinsi")
st.markdown("Ringkasan model yang dilatih otomatis. **Ref DTH/OTT** = apakah model sub-produk pakai data lokal atau fallback nasional.")

status_rows = []
for prov in ["Seluruh Indonesia"] + sorted(df_train["Provinsi"].unique()):
    if prov in all_models:
        pk   = all_models[prov]
        df_p = df_train if prov=="Seluruh Indonesia" else df_train[df_train["Provinsi"]==prov]
        n_data = len(df_p[df_p["label_kategori"].notna()])
        try:
            y_p   = df_p["label_kategori"]
            X_ps  = pk['scaler_strategi'].transform(df_p[pk['selected_vars']])
            acc_p = f"{accuracy_score(y_p, pk['model_strategi'].predict(X_ps)):.1%}"
        except:
            acc_p = "-"
        status_rows.append({"Provinsi":prov,"Status":"✅ Siap","Data":n_data,
                             "Fitur Aktif":len(pk['selected_vars']),"Akurasi":acc_p,
                             "Ref DTH":pk.get('dth_source','').title(),
                             "Ref OTT":pk.get('ott_source','').title()})
    else:
        status_rows.append({"Provinsi":prov,"Status":"⚠️ Dilewati",
                             "Data":"-","Fitur Aktif":"-","Akurasi":"-","Ref DTH":"-","Ref OTT":"-"})

df_status = pd.DataFrame(status_rows)
st.dataframe(df_status, use_container_width=True, height=340)
ready_count = len([r for r in status_rows if "✅" in r["Status"]])
st.caption(f"Total **{ready_count} model aktif** dari {len(status_rows)} wilayah. Refresh halaman = latih ulang otomatis.")
