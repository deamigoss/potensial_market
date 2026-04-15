<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
from utils.preprocessing import preprocessing_pipeline
from components.sidebar import sidebar

st.set_page_config(page_title="Prediction Tool", layout="wide", page_icon="🔮")
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
.badge-dth{background:rgba(0,212,255,0.15);color:#00d4ff;border:1px solid rgba(0,212,255,0.4);border-radius:8px;padding:3px 12px;font-size:0.82rem;font-weight:700;display:inline-block;}
.badge-ott{background:rgba(151,45,180,0.15);color:#c084fc;border:1px solid rgba(151,45,180,0.4);border-radius:8px;padding:3px 12px;font-size:0.82rem;font-weight:700;display:inline-block;}
.badge-hybrid{background:rgba(29,110,239,0.15);color:#60a5fa;border:1px solid rgba(29,110,239,0.4);border-radius:8px;padding:3px 12px;font-size:0.82rem;font-weight:700;display:inline-block;}
.model-badge{background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.3);border-radius:12px;padding:12px 20px;}
.model-badge p{color:#00d4ff;font-weight:600;margin:0;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ======================
# IMPORT MODEL CACHE DARI MODEL PERFORMANCE
# Model sudah di-pretrain di halaman Model Performance via st.cache_resource
# Kita import fungsi yang sama agar mengakses cache yang sama
# ======================
from utils.preprocessing import preprocessing_pipeline, get_training_data
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

COL_HH     = "Jumlah Household"
COL_SINYAL = "Jumlah Desa/Kelurahan Yang Menerima Sinyal 5G/4G/LTE (Persen)"
COL_UMP    = "UMP (Rupiah)"
COL_PDRB   = "PDRB (Ribu Rp)"
COL_EXP    = "Rata-Rata Pengeluaran per Kapita Bukan Makanan (Rupiah)"
COL_TV     = "Tercover TV Digital?"
feature_cols = [COL_HH, COL_SINYAL, COL_UMP, COL_PDRB, COL_EXP, COL_TV]
label_map    = {1:"DTH", 2:"OTT", 3:"Hybrid"}

def backward_wald(X, y, threshold=0.05):
    X_const   = sm.add_constant(X)
    variables = list(X_const.columns)
    while True:
        mdl     = sm.Logit(y, X_const[variables]).fit(disp=0)
        pvalues = mdl.pvalues
        max_p   = pvalues.max()
        if max_p > threshold:
            remove_var = pvalues.idxmax()
            if remove_var == "const": break
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
        if df_scope.empty or len(df_scope["label_kategori"].unique()) < 2: continue
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
        all_models[prov_name] = build_model_pack(df_scope, df_train, vars_sel, sc, mdl, prov_name)
    return all_models, df_train

# ======================
# PREDIKSI HIERARKIS SATU BARIS
# ======================
def predict_row(row_data, assets):
    """
    Alur:
    1. Prediksi strategi: DTH / OTT / Hybrid (pilih prob tertinggi)
    2. DTH  -> bandingkan prob Satellite vs Nusantara -> pilih tertinggi
    3. OTT  -> bandingkan prob Seru vs 2nd Gen
               -> 2nd Gen menang: catat "2nd Gen + Seru" (bundling wajib)
               -> Seru menang  : catat "Seru"
    4. Hybrid -> jalankan (2) DAN (3), gabungkan
    """
    sv   = assets['selected_vars']
    X_row = pd.DataFrame([row_data])[sv]

    # Strategi
    X_st       = assets['scaler_strategi'].transform(X_row)
    st_probs   = assets['model_strategi'].predict_proba(X_st)[0]
    st_pred    = assets['model_strategi'].predict(X_st)[0]
    strategi   = assets['label_map_strategi'][st_pred]
    confidence = f"{np.max(st_probs)*100:.1f}%"

    # DTH sub-produk
    prod_dth = "-"
    prob_dth_detail = "-"
    if assets.get('model_produk_dth') is not None:
        try:
            X_d      = assets['scaler_dth'].transform(X_row)
            dth_prob = assets['model_produk_dth'].predict_proba(X_d)[0]
            dth_cls  = assets['model_produk_dth'].classes_
            winner   = int(dth_cls[np.argmax(dth_prob)])
            prod_dth = assets['label_map_dth'][winner]
            prob_dth_detail = f"Satellite {dth_prob[0]*100:.0f}% / Nusantara {dth_prob[1]*100:.0f}%"
        except: pass

    # OTT sub-produk
    prod_ott = "-"
    prob_ott_detail = "-"
    if assets.get('model_produk_ott') is not None:
        try:
            X_o      = assets['scaler_ott'].transform(X_row)
            ott_prob = assets['model_produk_ott'].predict_proba(X_o)[0]
            ott_cls  = assets['model_produk_ott'].classes_
            winner   = int(ott_cls[np.argmax(ott_prob)])
            ott_raw  = assets['label_map_ott'][winner]
            # Business rule: 2nd Gen SELALU bundel Seru
            prod_ott = "2nd Gen + Seru" if ott_raw=="2nd Gen" else "Seru"
            prob_ott_detail = f"Seru {ott_prob[0]*100:.0f}% / 2nd Gen {ott_prob[1]*100:.0f}%"
        except: pass

    # Rekomendasi akhir
    if strategi == "DTH":
        rekomendasi = prod_dth
    elif strategi == "OTT":
        rekomendasi = prod_ott
    elif strategi == "Hybrid":
        parts = [p for p in [prod_dth, prod_ott] if p!="-"]
        rekomendasi = " + ".join(parts) if parts else "-"
    else:
        rekomendasi = "-"

    return {"strategi": strategi, "prod_dth": prod_dth, "prod_ott": prod_ott,
            "rekomendasi": rekomendasi, "confidence": confidence,
            "prob_dth": prob_dth_detail, "prob_ott": prob_ott_detail}

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Memuat model..."):
    all_models, df_train_ref = pretrain_all_models()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🔮 Prediksi Strategi & Produk")
st.markdown("""
<div style="background:rgba(30,41,59,0.45);padding:22px 26px;border-radius:18px;
            border-left:5px solid #972db4;border:1px solid rgba(255,255,255,0.06);margin-bottom:8px;">
  <p style="margin:0 0 8px 0;font-size:1.1rem;font-weight:700;color:#f8fafc;">🔮 Bulk Prediction Berbasis Model Hierarkis</p>
  <p style="margin:0;font-size:0.93rem;color:#cbd5e1;line-height:1.75;">
    Upload file Excel berisi data wilayah target → sistem menentukan <b>Strategi</b> (DTH/OTT/Hybrid)
    lalu memilih <b>Produk terbaik</b> berdasarkan probabilitas logistik.<br>
    <span style="color:#00d4ff;">Model dipilih <b>dinamis per provinsi</b> — setiap provinsi menggunakan model yang dilatih dari data historis lokal.</span><br>
  </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── SECTION 1: DOWNLOAD TEMPLATE ─────────────────────────────────────────────
st.subheader("① Persiapan Data — Download Template")
df_raw_ref = pd.read_excel("data/dataset2k22-2k26.xlsx")
avail_cols = df_raw_ref.columns.tolist()
col_prov   = next((c for c in avail_cols if 'PROVINSI' in c.upper()), None)
col_kab    = next((c for c in avail_cols if 'KABUPATEN' in c.upper() or 'KOTA' in c.upper()), None)

# Template pakai fitur nasional (semua provinsi)
template_df = df_raw_ref[df_raw_ref['Tahun']==2022].groupby('Provinsi').head(1)
cols_tmpl   = []
if col_prov: cols_tmpl.append(col_prov)
if col_kab:  cols_tmpl.append(col_kab)
cols_tmpl  += [c for c in feature_cols if c in avail_cols]
cols_tmpl   = list(dict.fromkeys(cols_tmpl))
template_df = template_df[[c for c in cols_tmpl if c in template_df.columns]]

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
    template_df.to_excel(writer, index=False, sheet_name='Template_Prediksi')

col_info, col_btn = st.columns([2,1])
with col_info:
    st.markdown(f"""<div class="model-badge" style="margin-bottom:10px;">
      <p>📋 Template: {len(template_df)} baris contoh data 2022 · {len(cols_tmpl)} kolom · Semua provinsi</p></div>""",
      unsafe_allow_html=True)
with col_btn:
    st.download_button("📥 Download Template Excel", data=buf.getvalue(),
                       file_name="template_prediksi_transvision.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

st.markdown("""
<div style="background:rgba(151,45,180,0.08);border:1px solid rgba(151,45,180,0.3);border-radius:12px;padding:14px;margin-top:10px;">
  <b style="color:#c084fc;">📌 Petunjuk Pengisian:</b>
  <ul style="margin:8px 0 0 0;color:#e2e8f0;font-size:0.9rem;line-height:1.8;">
    <li>Isi nilai fitur sesuai data aktual wilayah target (proyeksi 2026 atau data terkini).</li>
    <li>Kolom <b>Provinsi</b> dan <b>Kota/Kab</b> wajib diisi — digunakan untuk memilih model yang tepat.</li>
    <li>Jangan mengubah nama kolom header.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── SECTION 2: UPLOAD & PREDICT ──────────────────────────────────────────────
st.subheader("② Upload File & Jalankan Prediksi")
uploaded_file = st.file_uploader("Pilih file Excel hasil pengisian template", type=["xlsx"],
                                  help="Format harus sesuai template yang sudah didownload.")

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    up_cols  = df_input.columns.tolist()
    up_col_prov = next((c for c in up_cols if 'PROVINSI' in c.upper()), None)
    up_col_kab  = next((c for c in up_cols if 'KABUPATEN' in c.upper() or 'KOTA' in c.upper()), None)

    missing = [c for c in feature_cols if c not in up_cols]
    if missing:
        st.error(f"❌ Kolom tidak ditemukan di file: {missing}")
        st.stop()

    with st.spinner("⏳ Menjalankan prediksi hierarkis..."):
        df_proc, _ = preprocessing_pipeline(df_input)
        hasil_list = []
        provinsi_list = df_proc[up_col_prov].unique() if up_col_prov else ["_all_"]

        for prov in provinsi_list:
            df_prov_rows = df_proc[df_proc[up_col_prov]==prov] if up_col_prov else df_proc

            # Pilih model: spesifik provinsi -> fallback nasional
            if prov in all_models:
                assets    = all_models[prov]
                model_src = f"Spesifik ({prov})"
            elif "Seluruh Indonesia" in all_models:
                assets    = all_models["Seluruh Indonesia"]
                model_src = "Fallback Nasional"
            else:
                st.warning(f"⚠️ Tidak ada model untuk {prov}, dilewati.")
                continue

            for idx, row in df_prov_rows.iterrows():
                # Pastikan semua fitur model ada
                sv = [v for v in assets['selected_vars'] if v in df_prov_rows.columns]
                if not sv: continue
                assets_used = {**assets, 'selected_vars': sv}
                result = predict_row(row, assets_used)
                row_dict = {c: row[c] for c in up_cols if c in df_proc.columns}
                row_dict.update({
                    "Strategi Terpilih" : result["strategi"],
                    "Produk DTH"        : result["prod_dth"],
                    "Prob DTH"          : result["prob_dth"],
                    "Produk OTT"        : result["prod_ott"],
                    "Prob OTT"          : result["prob_ott"],
                    "Rekomendasi Produk": result["rekomendasi"],
                    "Confidence"        : result["confidence"],
                    "Model Digunakan"   : model_src,
                })
                hasil_list.append(row_dict)

        df_result = pd.DataFrame(hasil_list)

    st.success(f"✅ Prediksi selesai — **{len(df_result)} wilayah** berhasil diproses.")
    st.divider()

    # ── RINGKASAN GLOBAL ────────────────────────────────────────────────────────
    st.subheader("📊 Ringkasan Hasil Prediksi")
    total_dth    = len(df_result[df_result["Strategi Terpilih"]=="DTH"])
    total_ott    = len(df_result[df_result["Strategi Terpilih"]=="OTT"])
    total_hybrid = len(df_result[df_result["Strategi Terpilih"]=="Hybrid"])

    for col, lbl, val, sub in zip(
        st.columns(4),
        ["🗂️ Total Wilayah","📡 Strategi DTH","📺 Strategi OTT","🔀 Strategi Hybrid"],
        [len(df_result), total_dth, total_ott, total_hybrid],
        ["Kab/Kota yang diproses","Cocok produk satelit","Cocok produk streaming","Cocok keduanya"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
              <span class="metric-label">{lbl}</span>
              <p class="metric-value">{val}</p>
              <span class="metric-sub">{sub}</span></div>""", unsafe_allow_html=True)

    st.write("")
    c_donut, c_bar = st.columns([1,1])
    with c_donut:
        dist_df = df_result["Strategi Terpilih"].value_counts().reset_index()
        dist_df.columns = ["Strategi","Jumlah"]
        fig_dist = px.pie(dist_df, values="Jumlah", names="Strategi", hole=0.55,
                          template="plotly_dark",
                          color="Strategi", color_discrete_map={"DTH":"#00d4ff","OTT":"#c084fc","Hybrid":"#60a5fa"},
                          title="Distribusi Strategi Seluruh Wilayah")
        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font=dict(color="#f8fafc"))
        st.plotly_chart(fig_dist, use_container_width=True)
    with c_bar:
        prod_dist = df_result["Rekomendasi Produk"].value_counts().reset_index()
        prod_dist.columns = ["Produk","Jumlah"]
        fig_prod = px.bar(prod_dist, x="Jumlah", y="Produk", orientation='h', text="Jumlah",
                          template="plotly_dark", color="Jumlah",
                          color_continuous_scale=["#972db4","#00d4ff"],
                          title="Distribusi Rekomendasi Produk")
        fig_prod.update_traces(textposition='outside')
        fig_prod.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               showlegend=False, margin=dict(l=10,r=30,t=40,b=10),
                               font=dict(color="#f8fafc"))
        st.plotly_chart(fig_prod, use_container_width=True)

    st.divider()

    # ── DETAIL PER PROVINSI ─────────────────────────────────────────────────────
    st.subheader("📍 Detail Hasil per Provinsi")
    prov_groups = sorted(df_result[up_col_prov].unique()) if up_col_prov else ["Semua Wilayah"]

    for prov in prov_groups:
        df_prov = df_result[df_result[up_col_prov]==prov] if up_col_prov else df_result
        top_strat  = df_prov["Strategi Terpilih"].mode().iloc[0] if not df_prov.empty else "-"
        badge_cls  = {"DTH":"badge-dth","OTT":"badge-ott","Hybrid":"badge-hybrid"}.get(top_strat,"badge-dth")
        model_used = df_prov["Model Digunakan"].iloc[0] if not df_prov.empty else "-"

        with st.expander(f"📍 {prov}  ({len(df_prov)} Kota/Kab)"):
            st.markdown(f"""
            <div style="margin-bottom:14px;">
              Strategi Dominan: <span class="{badge_cls}">{top_strat}</span>
              &nbsp;&nbsp;<span style="color:#64748b;font-size:0.82rem;">Model: {model_used}</span>
            </div>""", unsafe_allow_html=True)

            for col_m, lbl_m, val_m in zip(
                st.columns(3),
                ["📡 DTH","📺 OTT","🔀 Hybrid"],
                [len(df_prov[df_prov["Strategi Terpilih"]=="DTH"]),
                 len(df_prov[df_prov["Strategi Terpilih"]=="OTT"]),
                 len(df_prov[df_prov["Strategi Terpilih"]=="Hybrid"])]
            ):
                with col_m:
                    st.markdown(f"""<div class="metric-card">
                      <span class="metric-label">{lbl_m}</span>
                      <p class="metric-value">{val_m}</p></div>""", unsafe_allow_html=True)

            st.write("")
            show_cols = []
            if up_col_kab and up_col_kab in df_prov.columns: show_cols.append(up_col_kab)
            show_cols += ["Strategi Terpilih","Produk DTH","Prob DTH","Produk OTT","Prob OTT",
                          "Rekomendasi Produk","Confidence"]
            st.dataframe(df_prov[show_cols].reset_index(drop=True), use_container_width=True)

    st.divider()

    # ── DOWNLOAD ────────────────────────────────────────────────────────────────
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_result.to_excel(writer, index=False, sheet_name='Hasil_Prediksi')

    dl_col, _ = st.columns([1,2])
    with dl_col:
        st.download_button("🚀 Download Hasil Prediksi (Excel)", data=output.getvalue(),
                           file_name="hasil_prediksi_transvision.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

# ── SIDEBAR STATUS ─────────────────────────────────────────────────────────────
ready = len(all_models)
st.sidebar.markdown(f"""
<div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);
            border-radius:10px;padding:12px;font-size:0.88rem;color:#94a3b8;
            margin-top:14px;">
  <b style="color:#00d4ff;">📦 Status Model</b><br><br>
  💾 Model tersedia: <b style="color:#f8fafc;">{ready} wilayah</b><br>
  🔄 Sumber: <b style="color:#f8fafc;">Memory Cache</b><br>
</div>
=======
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
from utils.preprocessing import preprocessing_pipeline
from components.sidebar import sidebar

st.set_page_config(page_title="Prediction Tool", layout="wide", page_icon="🔮")
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
.badge-dth{background:rgba(0,212,255,0.15);color:#00d4ff;border:1px solid rgba(0,212,255,0.4);border-radius:8px;padding:3px 12px;font-size:0.82rem;font-weight:700;display:inline-block;}
.badge-ott{background:rgba(151,45,180,0.15);color:#c084fc;border:1px solid rgba(151,45,180,0.4);border-radius:8px;padding:3px 12px;font-size:0.82rem;font-weight:700;display:inline-block;}
.badge-hybrid{background:rgba(29,110,239,0.15);color:#60a5fa;border:1px solid rgba(29,110,239,0.4);border-radius:8px;padding:3px 12px;font-size:0.82rem;font-weight:700;display:inline-block;}
.model-badge{background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.3);border-radius:12px;padding:12px 20px;}
.model-badge p{color:#00d4ff;font-weight:600;margin:0;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ======================
# IMPORT MODEL CACHE DARI MODEL PERFORMANCE
# Model sudah di-pretrain di halaman Model Performance via st.cache_resource
# Kita import fungsi yang sama agar mengakses cache yang sama
# ======================
from utils.preprocessing import preprocessing_pipeline, get_training_data
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

COL_HH     = "Jumlah Household"
COL_SINYAL = "Jumlah Desa/Kelurahan Yang Menerima Sinyal 5G/4G/LTE (Persen)"
COL_UMP    = "UMP (Rupiah)"
COL_PDRB   = "PDRB (Ribu Rp)"
COL_EXP    = "Rata-Rata Pengeluaran per Kapita Bukan Makanan (Rupiah)"
COL_TV     = "Tercover TV Digital?"
feature_cols = [COL_HH, COL_SINYAL, COL_UMP, COL_PDRB, COL_EXP, COL_TV]
label_map    = {1:"DTH", 2:"OTT", 3:"Hybrid"}

def backward_wald(X, y, threshold=0.05):
    X_const   = sm.add_constant(X)
    variables = list(X_const.columns)
    while True:
        mdl     = sm.Logit(y, X_const[variables]).fit(disp=0)
        pvalues = mdl.pvalues
        max_p   = pvalues.max()
        if max_p > threshold:
            remove_var = pvalues.idxmax()
            if remove_var == "const": break
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
        if df_scope.empty or len(df_scope["label_kategori"].unique()) < 2: continue
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
        all_models[prov_name] = build_model_pack(df_scope, df_train, vars_sel, sc, mdl, prov_name)
    return all_models, df_train

# ======================
# PREDIKSI HIERARKIS SATU BARIS
# ======================
def predict_row(row_data, assets):
    """
    Alur:
    1. Prediksi strategi: DTH / OTT / Hybrid (pilih prob tertinggi)
    2. DTH  -> bandingkan prob Satellite vs Nusantara -> pilih tertinggi
    3. OTT  -> bandingkan prob Seru vs 2nd Gen
               -> 2nd Gen menang: catat "2nd Gen + Seru" (bundling wajib)
               -> Seru menang  : catat "Seru"
    4. Hybrid -> jalankan (2) DAN (3), gabungkan
    """
    sv   = assets['selected_vars']
    X_row = pd.DataFrame([row_data])[sv]

    # Strategi
    X_st       = assets['scaler_strategi'].transform(X_row)
    st_probs   = assets['model_strategi'].predict_proba(X_st)[0]
    st_pred    = assets['model_strategi'].predict(X_st)[0]
    strategi   = assets['label_map_strategi'][st_pred]
    confidence = f"{np.max(st_probs)*100:.1f}%"

    # DTH sub-produk
    prod_dth = "-"
    prob_dth_detail = "-"
    if assets.get('model_produk_dth') is not None:
        try:
            X_d      = assets['scaler_dth'].transform(X_row)
            dth_prob = assets['model_produk_dth'].predict_proba(X_d)[0]
            dth_cls  = assets['model_produk_dth'].classes_
            winner   = int(dth_cls[np.argmax(dth_prob)])
            prod_dth = assets['label_map_dth'][winner]
            prob_dth_detail = f"Satellite {dth_prob[0]*100:.0f}% / Nusantara {dth_prob[1]*100:.0f}%"
        except: pass

    # OTT sub-produk
    prod_ott = "-"
    prob_ott_detail = "-"
    if assets.get('model_produk_ott') is not None:
        try:
            X_o      = assets['scaler_ott'].transform(X_row)
            ott_prob = assets['model_produk_ott'].predict_proba(X_o)[0]
            ott_cls  = assets['model_produk_ott'].classes_
            winner   = int(ott_cls[np.argmax(ott_prob)])
            ott_raw  = assets['label_map_ott'][winner]
            # Business rule: 2nd Gen SELALU bundel Seru
            prod_ott = "2nd Gen + Seru" if ott_raw=="2nd Gen" else "Seru"
            prob_ott_detail = f"Seru {ott_prob[0]*100:.0f}% / 2nd Gen {ott_prob[1]*100:.0f}%"
        except: pass

    # Rekomendasi akhir
    if strategi == "DTH":
        rekomendasi = prod_dth
    elif strategi == "OTT":
        rekomendasi = prod_ott
    elif strategi == "Hybrid":
        parts = [p for p in [prod_dth, prod_ott] if p!="-"]
        rekomendasi = " + ".join(parts) if parts else "-"
    else:
        rekomendasi = "-"

    return {"strategi": strategi, "prod_dth": prod_dth, "prod_ott": prod_ott,
            "rekomendasi": rekomendasi, "confidence": confidence,
            "prob_dth": prob_dth_detail, "prob_ott": prob_ott_detail}

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Memuat model..."):
    all_models, df_train_ref = pretrain_all_models()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🔮 Prediksi Strategi & Produk")
st.markdown("""
<div style="background:rgba(30,41,59,0.45);padding:22px 26px;border-radius:18px;
            border-left:5px solid #972db4;border:1px solid rgba(255,255,255,0.06);margin-bottom:8px;">
  <p style="margin:0 0 8px 0;font-size:1.1rem;font-weight:700;color:#f8fafc;">🔮 Bulk Prediction Berbasis Model Hierarkis</p>
  <p style="margin:0;font-size:0.93rem;color:#cbd5e1;line-height:1.75;">
    Upload file Excel berisi data wilayah target → sistem menentukan <b>Strategi</b> (DTH/OTT/Hybrid)
    lalu memilih <b>Produk terbaik</b> berdasarkan probabilitas logistik.<br>
    <span style="color:#00d4ff;">Model dipilih <b>dinamis per provinsi</b> — setiap provinsi menggunakan model yang dilatih dari data historis lokal.</span><br>
  </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── SECTION 1: DOWNLOAD TEMPLATE ─────────────────────────────────────────────
st.subheader("① Persiapan Data — Download Template")
df_raw_ref = pd.read_excel("data/dataset2k22-2k26.xlsx")
avail_cols = df_raw_ref.columns.tolist()
col_prov   = next((c for c in avail_cols if 'PROVINSI' in c.upper()), None)
col_kab    = next((c for c in avail_cols if 'KABUPATEN' in c.upper() or 'KOTA' in c.upper()), None)

# Template pakai fitur nasional (semua provinsi)
template_df = df_raw_ref[df_raw_ref['Tahun']==2022].groupby('Provinsi').head(1)
cols_tmpl   = []
if col_prov: cols_tmpl.append(col_prov)
if col_kab:  cols_tmpl.append(col_kab)
cols_tmpl  += [c for c in feature_cols if c in avail_cols]
cols_tmpl   = list(dict.fromkeys(cols_tmpl))
template_df = template_df[[c for c in cols_tmpl if c in template_df.columns]]

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
    template_df.to_excel(writer, index=False, sheet_name='Template_Prediksi')

col_info, col_btn = st.columns([2,1])
with col_info:
    st.markdown(f"""<div class="model-badge" style="margin-bottom:10px;">
      <p>📋 Template: {len(template_df)} baris contoh data 2022 · {len(cols_tmpl)} kolom · Semua provinsi</p></div>""",
      unsafe_allow_html=True)
with col_btn:
    st.download_button("📥 Download Template Excel", data=buf.getvalue(),
                       file_name="template_prediksi_transvision.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

st.markdown("""
<div style="background:rgba(151,45,180,0.08);border:1px solid rgba(151,45,180,0.3);border-radius:12px;padding:14px;margin-top:10px;">
  <b style="color:#c084fc;">📌 Petunjuk Pengisian:</b>
  <ul style="margin:8px 0 0 0;color:#e2e8f0;font-size:0.9rem;line-height:1.8;">
    <li>Isi nilai fitur sesuai data aktual wilayah target (proyeksi 2026 atau data terkini).</li>
    <li>Kolom <b>Provinsi</b> dan <b>Kota/Kab</b> wajib diisi — digunakan untuk memilih model yang tepat.</li>
    <li>Jangan mengubah nama kolom header.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── SECTION 2: UPLOAD & PREDICT ──────────────────────────────────────────────
st.subheader("② Upload File & Jalankan Prediksi")
uploaded_file = st.file_uploader("Pilih file Excel hasil pengisian template", type=["xlsx"],
                                  help="Format harus sesuai template yang sudah didownload.")

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    up_cols  = df_input.columns.tolist()
    up_col_prov = next((c for c in up_cols if 'PROVINSI' in c.upper()), None)
    up_col_kab  = next((c for c in up_cols if 'KABUPATEN' in c.upper() or 'KOTA' in c.upper()), None)

    missing = [c for c in feature_cols if c not in up_cols]
    if missing:
        st.error(f"❌ Kolom tidak ditemukan di file: {missing}")
        st.stop()

    with st.spinner("⏳ Menjalankan prediksi hierarkis..."):
        df_proc, _ = preprocessing_pipeline(df_input)
        hasil_list = []
        provinsi_list = df_proc[up_col_prov].unique() if up_col_prov else ["_all_"]

        for prov in provinsi_list:
            df_prov_rows = df_proc[df_proc[up_col_prov]==prov] if up_col_prov else df_proc

            # Pilih model: spesifik provinsi -> fallback nasional
            if prov in all_models:
                assets    = all_models[prov]
                model_src = f"Spesifik ({prov})"
            elif "Seluruh Indonesia" in all_models:
                assets    = all_models["Seluruh Indonesia"]
                model_src = "Fallback Nasional"
            else:
                st.warning(f"⚠️ Tidak ada model untuk {prov}, dilewati.")
                continue

            for idx, row in df_prov_rows.iterrows():
                # Pastikan semua fitur model ada
                sv = [v for v in assets['selected_vars'] if v in df_prov_rows.columns]
                if not sv: continue
                assets_used = {**assets, 'selected_vars': sv}
                result = predict_row(row, assets_used)
                row_dict = {c: row[c] for c in up_cols if c in df_proc.columns}
                row_dict.update({
                    "Strategi Terpilih" : result["strategi"],
                    "Produk DTH"        : result["prod_dth"],
                    "Prob DTH"          : result["prob_dth"],
                    "Produk OTT"        : result["prod_ott"],
                    "Prob OTT"          : result["prob_ott"],
                    "Rekomendasi Produk": result["rekomendasi"],
                    "Confidence"        : result["confidence"],
                    "Model Digunakan"   : model_src,
                })
                hasil_list.append(row_dict)

        df_result = pd.DataFrame(hasil_list)

    st.success(f"✅ Prediksi selesai — **{len(df_result)} wilayah** berhasil diproses.")
    st.divider()

    # ── RINGKASAN GLOBAL ────────────────────────────────────────────────────────
    st.subheader("📊 Ringkasan Hasil Prediksi")
    total_dth    = len(df_result[df_result["Strategi Terpilih"]=="DTH"])
    total_ott    = len(df_result[df_result["Strategi Terpilih"]=="OTT"])
    total_hybrid = len(df_result[df_result["Strategi Terpilih"]=="Hybrid"])

    for col, lbl, val, sub in zip(
        st.columns(4),
        ["🗂️ Total Wilayah","📡 Strategi DTH","📺 Strategi OTT","🔀 Strategi Hybrid"],
        [len(df_result), total_dth, total_ott, total_hybrid],
        ["Kab/Kota yang diproses","Cocok produk satelit","Cocok produk streaming","Cocok keduanya"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
              <span class="metric-label">{lbl}</span>
              <p class="metric-value">{val}</p>
              <span class="metric-sub">{sub}</span></div>""", unsafe_allow_html=True)

    st.write("")
    c_donut, c_bar = st.columns([1,1])
    with c_donut:
        dist_df = df_result["Strategi Terpilih"].value_counts().reset_index()
        dist_df.columns = ["Strategi","Jumlah"]
        fig_dist = px.pie(dist_df, values="Jumlah", names="Strategi", hole=0.55,
                          template="plotly_dark",
                          color="Strategi", color_discrete_map={"DTH":"#00d4ff","OTT":"#c084fc","Hybrid":"#60a5fa"},
                          title="Distribusi Strategi Seluruh Wilayah")
        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font=dict(color="#f8fafc"))
        st.plotly_chart(fig_dist, use_container_width=True)
    with c_bar:
        prod_dist = df_result["Rekomendasi Produk"].value_counts().reset_index()
        prod_dist.columns = ["Produk","Jumlah"]
        fig_prod = px.bar(prod_dist, x="Jumlah", y="Produk", orientation='h', text="Jumlah",
                          template="plotly_dark", color="Jumlah",
                          color_continuous_scale=["#972db4","#00d4ff"],
                          title="Distribusi Rekomendasi Produk")
        fig_prod.update_traces(textposition='outside')
        fig_prod.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               showlegend=False, margin=dict(l=10,r=30,t=40,b=10),
                               font=dict(color="#f8fafc"))
        st.plotly_chart(fig_prod, use_container_width=True)

    st.divider()

    # ── DETAIL PER PROVINSI ─────────────────────────────────────────────────────
    st.subheader("📍 Detail Hasil per Provinsi")
    prov_groups = sorted(df_result[up_col_prov].unique()) if up_col_prov else ["Semua Wilayah"]

    for prov in prov_groups:
        df_prov = df_result[df_result[up_col_prov]==prov] if up_col_prov else df_result
        top_strat  = df_prov["Strategi Terpilih"].mode().iloc[0] if not df_prov.empty else "-"
        badge_cls  = {"DTH":"badge-dth","OTT":"badge-ott","Hybrid":"badge-hybrid"}.get(top_strat,"badge-dth")
        model_used = df_prov["Model Digunakan"].iloc[0] if not df_prov.empty else "-"

        with st.expander(f"📍 {prov}  ({len(df_prov)} Kota/Kab)"):
            st.markdown(f"""
            <div style="margin-bottom:14px;">
              Strategi Dominan: <span class="{badge_cls}">{top_strat}</span>
              &nbsp;&nbsp;<span style="color:#64748b;font-size:0.82rem;">Model: {model_used}</span>
            </div>""", unsafe_allow_html=True)

            for col_m, lbl_m, val_m in zip(
                st.columns(3),
                ["📡 DTH","📺 OTT","🔀 Hybrid"],
                [len(df_prov[df_prov["Strategi Terpilih"]=="DTH"]),
                 len(df_prov[df_prov["Strategi Terpilih"]=="OTT"]),
                 len(df_prov[df_prov["Strategi Terpilih"]=="Hybrid"])]
            ):
                with col_m:
                    st.markdown(f"""<div class="metric-card">
                      <span class="metric-label">{lbl_m}</span>
                      <p class="metric-value">{val_m}</p></div>""", unsafe_allow_html=True)

            st.write("")
            show_cols = []
            if up_col_kab and up_col_kab in df_prov.columns: show_cols.append(up_col_kab)
            show_cols += ["Strategi Terpilih","Produk DTH","Prob DTH","Produk OTT","Prob OTT",
                          "Rekomendasi Produk","Confidence"]
            st.dataframe(df_prov[show_cols].reset_index(drop=True), use_container_width=True)

    st.divider()

    # ── DOWNLOAD ────────────────────────────────────────────────────────────────
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_result.to_excel(writer, index=False, sheet_name='Hasil_Prediksi')

    dl_col, _ = st.columns([1,2])
    with dl_col:
        st.download_button("🚀 Download Hasil Prediksi (Excel)", data=output.getvalue(),
                           file_name="hasil_prediksi_transvision.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

# ── SIDEBAR STATUS ─────────────────────────────────────────────────────────────
ready = len(all_models)
st.sidebar.markdown(f"""
<div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);
            border-radius:10px;padding:12px;font-size:0.88rem;color:#94a3b8;
            margin-top:14px;">
  <b style="color:#00d4ff;">📦 Status Model</b><br><br>
  💾 Model tersedia: <b style="color:#f8fafc;">{ready} wilayah</b><br>
  🔄 Sumber: <b style="color:#f8fafc;">Memory Cache</b><br>
</div>
>>>>>>> b475ac26a41bac64202d5dd694929abe2b8af603
""", unsafe_allow_html=True)