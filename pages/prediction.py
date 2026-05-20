import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from utils.preprocessing import preprocessing_pipeline, get_training_data
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

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
COL_HH     = "Jumlah Household"
COL_SINYAL = "Jumlah Desa/Kelurahan Yang Menerima Sinyal 5G/4G/LTE (Persen)"
COL_UMP    = "UMP (Rupiah)"
COL_PDRB   = "PDRB (Ribu Rp)"
COL_EXP    = "Rata-Rata Pengeluaran per Kapita Bukan Makanan (Rupiah)"
COL_TV     = "Tercover TV Digital?"
feature_cols = [COL_HH, COL_SINYAL, COL_UMP, COL_PDRB, COL_EXP, COL_TV]
label_map    = {1: "DTH", 2: "OTT", 3: "Hybrid"}

# ══════════════════════════════════════════════════════════════════════════════
# STATSMODELS WRAPPER — sama persis dengan model_performance.py
# Wajib ada di sini karena prediction.py juga memanggil pretrain_all_models()
# ══════════════════════════════════════════════════════════════════════════════
class StatsmodelsWrapper:
    def __init__(self, sm_result, classes, is_multinomial=False):
        self._result   = sm_result
        self.classes_  = np.array(classes)
        self._is_multi = is_multinomial

        params = sm_result.params

        if is_multinomial:
            # MNLogit: params = DataFrame (n_vars+1, n_classes-1)
            if isinstance(params, pd.DataFrame):
                self.intercept_ = params.iloc[0, :].values
                self.coef_      = params.iloc[1:, :].values.T
                self._var_names = list(params.index[1:])
            else:
                params = np.array(params)
                if params.ndim == 1:
                    params = params.reshape(-1, 1)
                self.intercept_ = params[0, :]
                self.coef_      = params[1:, :].T
                self._var_names = []
        else:
            # Logit biner: params = Series [const, var1, var2, ...]
            if isinstance(params, pd.Series):
                self.intercept_ = np.array([params.iloc[0]])
                self.coef_      = np.array([params.iloc[1:].values])
                self._var_names = list(params.index[1:])
            else:
                params = np.array(params).ravel()
                self.intercept_ = np.array([params[0]])
                self.coef_      = np.array([params[1:]])
                self._var_names = []

    def _add_const(self, X):
        if isinstance(X, pd.DataFrame):
            if 'const' not in X.columns:
                return sm.add_constant(X, has_constant='add')
            return X
        X = np.array(X, dtype=float)
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def predict_proba(self, X):
        X_c = self._add_const(X)
        if self._is_multi:
            proba = self._result.predict(X_c)
            if isinstance(proba, pd.DataFrame):
                proba = proba.values
            return np.array(proba, dtype=float)
        else:
            p1 = np.array(self._result.predict(X_c), dtype=float)
            return np.column_stack([1 - p1, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        idx   = np.argmax(proba, axis=1)
        return self.classes_[idx]


# ── BACKWARD WALD ─────────────────────────────────────────────────────────────
def backward_wald(X_df, y_bin, threshold=0.05):
    X_c       = sm.add_constant(X_df, has_constant='add')
    variables = list(X_c.columns)
    while True:
        try:
            mdl     = sm.Logit(y_bin, X_c[variables]).fit(disp=0, method='bfgs', maxiter=200)
            pvalues = mdl.pvalues
            max_p   = pvalues.max()
            if max_p > threshold:
                rv = pvalues.idxmax()
                if rv == 'const':
                    break
                variables.remove(rv)
            else:
                break
        except Exception:
            break
    return [v for v in variables if v != 'const']


# ── BUILD MODEL PACK ──────────────────────────────────────────────────────────
def build_model_pack(df_scope, df_train_full, vars_strategi,
                     scaler_strategi, model_strategi, provinsi_name):
    def _fit_statsmodels(df_data, df_fallback, label_col, use_notna_col=None):
        if use_notna_col and use_notna_col in df_data.columns:
            d = df_data[df_data[label_col].notna() & df_data[use_notna_col].notna()]
        else:
            d = df_data[df_data[label_col].notna()]

        if len(d[label_col].unique()) < 2:
            if use_notna_col and use_notna_col in df_fallback.columns:
                d = df_fallback[df_fallback[label_col].notna() & df_fallback[use_notna_col].notna()]
            else:
                d = df_fallback[df_fallback[label_col].notna()]

        Xs = d[vars_strategi]
        ys = d[label_col].values

        # ── Backward Wald untuk sub-model DTH / OTT ─────────────────────
        # Sama seperti model strategi: hanya variabel signifikan yang masuk
        feat_valid_sub = [c for c in vars_strategi if Xs[c].std() > 1e-6]
        if not feat_valid_sub:
            feat_valid_sub = vars_strategi

        sc_pre    = StandardScaler()
        Xs_pre    = pd.DataFrame(sc_pre.fit_transform(Xs[feat_valid_sub]),
                                 columns=feat_valid_sub)
        y_b_bw    = (ys == ys.max()).astype(int) if len(np.unique(ys)) == 2 else ys
        vars_sub  = backward_wald(Xs_pre, y_b_bw)
        if not vars_sub:
            vars_sub = feat_valid_sub

        sc    = StandardScaler()
        Xs_sc = sc.fit_transform(Xs[vars_sub])
        Xs_df = pd.DataFrame(Xs_sc, columns=vars_sub)

        try:
            X_c    = sm.add_constant(Xs_df, has_constant='add')
            y_b    = (ys == ys.max()).astype(int) if len(np.unique(ys)) == 2 else ys
            sm_res = sm.Logit(y_b, X_c).fit(disp=0, method='bfgs', maxiter=300)
            classes = sorted(np.unique(ys))
            wrapper = StatsmodelsWrapper(sm_res, classes, is_multinomial=False)
            # Simpan vars_sub agar predict_row tahu kolom yang dipakai
            wrapper._sub_vars = vars_sub
            return wrapper, sc
        except Exception:
            clf = LogisticRegression(max_iter=2000, C=1e9)
            clf.fit(Xs_sc, ys)
            clf._sub_vars = vars_sub
            return clf, sc

    m_dth, sc_dth = _fit_statsmodels(df_scope, df_train_full, label_col="label_dth", use_notna_col=None)
    m_ott, sc_ott = _fit_statsmodels(df_scope, df_train_full, label_col="label_ott", use_notna_col="OTT")

    return {
        'model_strategi'    : model_strategi,
        'scaler_strategi'   : scaler_strategi,
        'selected_vars'     : vars_strategi,
        'label_map_strategi': label_map,
        'provinsi_terpilih' : provinsi_name,
        'model_produk_dth'  : m_dth,
        'scaler_dth'        : sc_dth,
        'label_map_dth'     : {0: "Nusantara", 1: "Satellite"},
        'model_produk_ott'  : m_ott,
        'scaler_ott'        : sc_ott,
        'label_map_ott'     : {0: "Seru", 1: "2nd Gen"},
    }


# ── PRE-TRAINING ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def pretrain_all_models():
    df_raw          = pd.read_excel("data/dataset2k22-2k26.xlsx")
    df_processed, _ = preprocessing_pipeline(df_raw.copy())
    df_train        = get_training_data(df_processed, None)
    all_models      = {}

    scopes = [("Seluruh Indonesia", df_train)] + [
        (p, df_train[df_train["Provinsi"] == p])
        for p in df_train["Provinsi"].dropna().unique()
    ]

    for prov_name, df_scope in scopes:
        if df_scope.empty or len(df_scope["label_kategori"].unique()) < 2:
            continue

        feat_valid = [c for c in feature_cols if df_scope[c].std() > 1e-6]
        if not feat_valid:
            continue

        y_scope    = df_scope["label_kategori"]
        n_classes  = sorted(y_scope.unique())
        ref_cls    = n_classes[0]
        target_cls = n_classes[-1]

        scaler   = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(df_scope[feat_valid]),
            columns=feat_valid, index=df_scope.index
        )

        y_bin    = (y_scope == target_cls).astype(int).values
        vars_sel = backward_wald(X_scaled, y_bin)
        if not vars_sel:
            vars_sel = feat_valid

        scaler_fin = StandardScaler()
        X_fin_arr  = scaler_fin.fit_transform(df_scope[vars_sel])
        X_fin      = pd.DataFrame(X_fin_arr, columns=vars_sel, index=df_scope.index)

        try:
            X_c = sm.add_constant(X_fin, has_constant='add')
            if len(n_classes) == 2:
                sm_res    = sm.Logit(y_bin, X_c).fit(disp=0, method='bfgs', maxiter=300)
                mdl_strat = StatsmodelsWrapper(sm_res, [ref_cls, target_cls], False)
            else:
                sm_res    = sm.MNLogit(y_scope.values, X_c).fit(disp=0, method='bfgs', maxiter=300)
                mdl_strat = StatsmodelsWrapper(sm_res, n_classes, True)
        except Exception:
            clf = LogisticRegression(solver='lbfgs', max_iter=2000, C=1e9, multi_class='auto')
            clf.fit(X_fin_arr, y_scope.values)
            mdl_strat = clf

        pack = build_model_pack(
            df_scope        = df_scope,
            df_train_full   = df_train,
            vars_strategi   = vars_sel,
            scaler_strategi = scaler_fin,
            model_strategi  = mdl_strat,
            provinsi_name   = prov_name
        )
        all_models[prov_name] = pack

    return all_models, df_train


def get_model_for_province(prov_name: str, all_models: dict):
    """
    Cari model yang paling sesuai untuk nama provinsi tertentu.
    Return: (assets_dict, label_sumber)

    Urutan prioritas:
    1. Match persis (setelah strip)
    2. Match case-insensitive
    3. Fallback → model Seluruh Indonesia
    """
    prov_clean = str(prov_name).strip()

    # 1. Match persis setelah strip
    if prov_clean in all_models:
        return all_models[prov_clean], f"Model Spesifik: {prov_clean}"

    # 2. Match case-insensitive
    prov_lower = prov_clean.lower()
    for key in all_models:
        if key.lower() == prov_lower:
            return all_models[key], f"Model Spesifik: {key}"

    # 3. Fallback nasional
    if "Seluruh Indonesia" in all_models:
        return all_models["Seluruh Indonesia"], "Fallback Nasional"

    return None, "Tidak Tersedia"


# ── PREDIKSI HIERARKIS SATU BARIS ─────────────────────────────────────────────
def predict_row(row_data, assets):
    """
    Alur hierarkis:
    1. Strategi: DTH / OTT / Hybrid
    2. DTH  → bandingkan prob Satellite vs Nusantara → pilih tertinggi
    3. OTT  → bandingkan prob Seru vs 2nd Gen
               → 2nd Gen menang: "2nd Gen + Seru" (business rule: selalu bundel)
               → Seru menang   : "Seru"
    4. Hybrid → jalankan (2) DAN (3), gabungkan
    """
    sv    = assets['selected_vars']
    avail = [v for v in sv if v in row_data.index and not pd.isna(row_data.get(v, np.nan))]
    if not avail:
        return {"strategi":"?", "prod_dth":"-", "prod_ott":"-",
                "rekomendasi":"-", "confidence":"0%",
                "prob_dth":"-", "prob_ott":"-"}

    X_row = pd.DataFrame([row_data])[avail]

    # 1. Prediksi strategi
    try:
        X_sc     = assets['scaler_strategi'].transform(X_row[[v for v in sv if v in avail]])
        X_sc_df  = pd.DataFrame(X_sc, columns=[v for v in sv if v in avail])
        st_probs = assets['model_strategi'].predict_proba(X_sc_df)[0]
        st_pred  = assets['model_strategi'].predict(X_sc_df)[0]
        strategi  = assets['label_map_strategi'].get(st_pred, str(st_pred))
        confidence = f"{np.max(st_probs)*100:.1f}%"
        # Prob per kelas untuk kolom Prob Strategi
        strat_classes = assets['model_strategi'].classes_
        prob_strat_parts = []
        for cls_, p_ in zip(strat_classes, st_probs):
            lbl_ = assets['label_map_strategi'].get(cls_, str(cls_))
            prob_strat_parts.append(f"{lbl_} {p_*100:.0f}%")
        prob_strategi_str = " / ".join(prob_strat_parts)
    except Exception:
        return {"strategi":"?","prod_dth":"-","prod_ott":"-",
                "rekomendasi":"-","confidence":"0%",
                "prob_dth":"-","prob_ott":"-","prob_strategi":"-"}

    # 2. Prediksi produk DTH
    prod_dth      = "-"
    prob_dth_str  = "-"
    try:
        # Pakai _sub_vars jika ada (Backward Wald sub-model), fallback ke sv
        m_dth     = assets['model_produk_dth']
        sv_dth    = getattr(m_dth, '_sub_vars', [v for v in sv if v in avail])
        sv_dth    = [v for v in sv_dth if v in avail]
        if not sv_dth:
            sv_dth = [v for v in sv if v in avail]
        X_dth    = assets['scaler_dth'].transform(pd.DataFrame([row_data])[sv_dth])
        dth_prob = m_dth.predict_proba(X_dth)[0]
        dth_cls  = m_dth.classes_
        winner   = int(dth_cls[np.argmax(dth_prob)])
        prod_dth = assets['label_map_dth'][winner]
        # Susun prob sesuai urutan kelas {0:Nusantara, 1:Satellite}
        lmap_dth = assets['label_map_dth']
        prob_parts_dth = [f"{lmap_dth[int(c)]} {p*100:.0f}%" 
                          for c,p in zip(dth_cls, dth_prob)]
        prob_dth_str = " / ".join(prob_parts_dth)
    except Exception:
        pass

    # 3. Prediksi produk OTT
    prod_ott      = "-"
    prob_ott_str  = "-"
    try:
        # Pakai _sub_vars jika ada (Backward Wald sub-model), fallback ke sv
        m_ott     = assets['model_produk_ott']
        sv_ott    = getattr(m_ott, '_sub_vars', [v for v in sv if v in avail])
        sv_ott    = [v for v in sv_ott if v in avail]
        if not sv_ott:
            sv_ott = [v for v in sv if v in avail]
        X_ott    = assets['scaler_ott'].transform(pd.DataFrame([row_data])[sv_ott])
        ott_prob = m_ott.predict_proba(X_ott)[0]
        ott_cls  = m_ott.classes_
        winner   = int(ott_cls[np.argmax(ott_prob)])
        ott_raw  = assets['label_map_ott'][winner]
        # Business rule: 2nd Gen selalu bundel Seru
        prod_ott = "2nd Gen + Seru" if ott_raw == "2nd Gen" else "Seru"
        # Susun prob sesuai urutan kelas {0:Seru, 1:2nd Gen}
        lmap_ott = assets['label_map_ott']
        prob_parts_ott = [f"{lmap_ott[int(c)]} {p*100:.0f}%"
                          for c,p in zip(ott_cls, ott_prob)]
        prob_ott_str = " / ".join(prob_parts_ott)
    except Exception:
        pass

    # 4. Rekomendasi akhir berdasarkan strategi
    if strategi == "DTH":
        rekomendasi = prod_dth
    elif strategi == "OTT":
        rekomendasi = prod_ott
    elif strategi == "Hybrid":
        parts       = [p for p in [prod_dth, prod_ott] if p != "-"]
        rekomendasi = " + ".join(parts) if parts else "-"
    else:
        rekomendasi = "-"

    return {
        "strategi"     : strategi,
        "prod_dth"     : prod_dth,
        "prod_ott"     : prod_ott,
        "rekomendasi"  : rekomendasi,
        "confidence"   : confidence,
        "prob_strategi": prob_strategi_str,
        "prob_dth"     : prob_dth_str,
        "prob_ott"     : prob_ott_str,
    }



# ── LOAD MODELS ───────────────────────────────────────────────────────────────
with st.spinner("⚙️ Memuat model..."):
    all_models, df_train_ref = pretrain_all_models()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🔮 Prediksi Strategi & Produk")
st.markdown("""
<div style="background:rgba(30,41,59,0.45);padding:22px 26px;border-radius:18px;
            border-left:5px solid #972db4;border:1px solid rgba(255,255,255,0.06);margin-bottom:8px;">
  <p style="margin:0 0 8px 0;font-size:1.1rem;font-weight:700;color:#f8fafc;">
    🔮 Bulk Prediction Berbasis Model Hierarkis</p>
  <p style="margin:0;font-size:0.93rem;color:#cbd5e1;line-height:1.75;">
    Upload data wilayah target → sistem menentukan <b>Strategi</b> (DTH/OTT/Hybrid)
    lalu memilih <b>Produk terbaik</b> berdasarkan probabilitas logistik.<br>
    <span style="color:#00d4ff;">
      Model dipilih <b>dinamis per provinsi</b> — pakai model lokal jika tersedia,
      fallback nasional hanya jika provinsi tidak ada di data historis.
    </span><br>
  </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── SECTION 1: DOWNLOAD TEMPLATE ─────────────────────────────────────────────
st.subheader("① Download Template")

df_raw_ref = pd.read_excel("data/dataset2k22-2k26.xlsx")
avail_cols = df_raw_ref.columns.tolist()
col_prov   = next((c for c in avail_cols if 'PROVINSI' in c.upper()), None)
col_kab    = next((c for c in avail_cols if 'KABUPATEN' in c.upper() or 'KOTA' in c.upper()), None)

# Ambil semua provinsi (1 baris per provinsi dari tahun 2022)
template_df = df_raw_ref[df_raw_ref['Tahun'] == 2022].groupby('Provinsi').head(1) \
              if 'Tahun' in df_raw_ref.columns else df_raw_ref.head(10)

cols_tmpl = []
if col_prov: cols_tmpl.append(col_prov)
if col_kab:  cols_tmpl.append(col_kab)
cols_tmpl += [c for c in feature_cols if c in avail_cols]
cols_tmpl  = list(dict.fromkeys(cols_tmpl))
template_df = template_df[[c for c in cols_tmpl if c in template_df.columns]]

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
    template_df.to_excel(writer, index=False, sheet_name='Template_Prediksi')

col_info, col_btn = st.columns([2, 1])
with col_info:
    st.markdown(f"""<div class="model-badge">
      <p>📋 Template: {len(template_df)} baris · {len(cols_tmpl)} kolom ·
      Isi nilai fitur sesuai data aktual wilayah target</p></div>""",
    unsafe_allow_html=True)
with col_btn:
    st.download_button(
        "📥 Download Template Excel", data=buf.getvalue(),
        file_name="template_prediksi_transvision.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.markdown("""
<div style="background:rgba(151,45,180,0.08);border:1px solid rgba(151,45,180,0.3);
            border-radius:12px;padding:14px;margin-top:10px;margin-bottom:24px;"> <b style="color:#c084fc;">📌 Petunjuk:</b>
  <ul style="margin:8px 0 0 0;color:#e2e8f0;font-size:0.9rem;line-height:1.8;">
    <li>Kolom <b>Provinsi</b> wajib diisi persis seperti nama di data historis
        (contoh: "Jawa Timur", bukan "JAWA TIMUR" atau "jawa timur").</li>
    <li>Kolom <b>Tercover TV Digital?</b>: isi <b>1</b> (Ya) atau <b>0</b> (Tidak).</li>
    <li>Jangan ubah nama kolom header.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# Tampilkan daftar provinsi yang modelnya tersedia
with st.expander("📋 Lihat daftar provinsi yang punya model spesifik"):
    prov_available = sorted([k for k in all_models.keys() if k != "Seluruh Indonesia"])
    df_prov_list   = pd.DataFrame({
        "Provinsi"   : prov_available,
        "Status"     : ["✅ Model Spesifik"] * len(prov_available),
        "Fitur Aktif": [len(all_models[p]['selected_vars']) for p in prov_available],
    })
    st.dataframe(df_prov_list, use_container_width=True, hide_index=True)
    st.caption(f"Total {len(prov_available)} provinsi. "
               f"Provinsi di luar daftar ini akan memakai model 'Seluruh Indonesia'.")

st.divider()

# ── SECTION 2: UPLOAD & PREDICT ───────────────────────────────────────────────
st.subheader("② Upload File & Prediksi")

uploaded_file = st.file_uploader(
    "Pilih file Excel hasil pengisian template", type=["xlsx"],
    help="Format harus sesuai template yang didownload."
)

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    up_cols  = df_input.columns.tolist()

    # Deteksi kolom provinsi dan kab/kota secara fleksibel
    up_col_prov = next((c for c in up_cols if 'PROVINSI' in c.upper()), None)
    up_col_kab  = next((c for c in up_cols if 'KABUPATEN' in c.upper()
                        or 'KOTA' in c.upper()), None)

    # Validasi kolom fitur minimal ada
    missing_feat = [c for c in feature_cols if c not in up_cols]
    if missing_feat:
        st.error(f"❌ Kolom tidak ditemukan di file: {missing_feat}")
        st.stop()

    # Preprocessing
    df_proc, _ = preprocessing_pipeline(df_input.copy())

    # ── DEBUG INFO: tampilkan nama provinsi yang terdeteksi ──────────────────
    if up_col_prov:
        detected_provs = df_proc[up_col_prov].unique().tolist()
        model_match = []
        for p in detected_provs:
            _, src = get_model_for_province(p, all_models)
            model_match.append(f"**{p}** → {src}")
        st.info("🔍 Deteksi model per provinsi:\n" + "\n".join(model_match))

    with st.spinner("⏳ Menjalankan prediksi hierarkis..."):
        hasil_list = []

        if up_col_prov:
            # Kelompokkan per provinsi agar model yang dipakai tepat
            provinsi_unik = df_proc[up_col_prov].unique()
        else:
            provinsi_unik = ["_semua_"]
            st.warning("⚠️ Kolom Provinsi tidak ditemukan. Memakai model Seluruh Indonesia.")

        for prov_val in provinsi_unik:
            # Filter baris untuk provinsi ini
            if up_col_prov and prov_val != "_semua_":
                df_prov_rows = df_proc[df_proc[up_col_prov] == prov_val]
            else:
                df_prov_rows = df_proc

            # Cari model dengan matching yang robust (FIX UTAMA)
            assets, model_src = get_model_for_province(prov_val, all_models)
            if assets is None:
                st.warning(f"⚠️ Tidak ada model untuk '{prov_val}', baris dilewati.")
                continue

            for _, row in df_prov_rows.iterrows():
                result   = predict_row(row, assets)
                row_dict = {c: row[c] for c in up_cols if c in df_proc.columns}
                row_dict.update({
                    "Strategi Terpilih" : result["strategi"],
                    "Prob Strategi"     : result.get("prob_strategi", "-"),
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

    st.success(f"✅ Prediksi selesai — **{len(df_result)} wilayah** diproses.")
    st.divider()

    # ── RINGKASAN ─────────────────────────────────────────────────────────────
    st.subheader("📊 Ringkasan Hasil")

    total_dth    = len(df_result[df_result["Strategi Terpilih"] == "DTH"])
    total_ott    = len(df_result[df_result["Strategi Terpilih"] == "OTT"])
    total_hybrid = len(df_result[df_result["Strategi Terpilih"] == "Hybrid"])

    for col, lbl, val, sub in zip(
        st.columns(4),
        ["🗂️ Total Wilayah","📡 Strategi DTH","📺 Strategi OTT","🔀 Strategi Hybrid"],
        [len(df_result), total_dth, total_ott, total_hybrid],
        ["Kab/Kota diproses","Satellite / Nusantara",
         "Seru / 2nd Gen+Seru","Kombinasi DTH & OTT"]
    ):
        with col:
            st.markdown(f"""<div class="metric-card">
              <span class="metric-label">{lbl}</span>
              <p class="metric-value">{val}</p>
              <span class="metric-sub">{sub}</span></div>""", unsafe_allow_html=True)

    st.write("")

    c_donut, c_bar = st.columns([1, 1])
    with c_donut:
        dist_df = df_result["Strategi Terpilih"].value_counts().reset_index()
        dist_df.columns = ["Strategi","Jumlah"]
        fig_dist = px.pie(dist_df, values="Jumlah", names="Strategi", hole=0.55,
                          template="plotly_dark",
                          color="Strategi",
                          color_discrete_map={"DTH":"#00d4ff","OTT":"#c084fc","Hybrid":"#60a5fa"},
                          title="Distribusi Strategi")
        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               font=dict(color="#f8fafc"))
        st.plotly_chart(fig_dist, use_container_width=True)
    with c_bar:
        prod_dist = df_result["Rekomendasi Produk"].value_counts().reset_index()
        prod_dist.columns = ["Produk","Jumlah"]
        fig_prod = px.bar(prod_dist, x="Jumlah", y="Produk", orientation='h',
                          text="Jumlah", template="plotly_dark",
                          color="Jumlah", color_continuous_scale=["#972db4","#00d4ff"],
                          title="Distribusi Rekomendasi Produk")
        fig_prod.update_traces(textposition='outside')
        fig_prod.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)',
                               showlegend=False,
                               margin=dict(l=10,r=30,t=40,b=10),
                               font=dict(color="#f8fafc"))
        st.plotly_chart(fig_prod, use_container_width=True)

    st.divider()

    # ── DETAIL PER PROVINSI ───────────────────────────────────────────────────
    st.subheader("📍 Detail Hasil per Provinsi")

    prov_groups = (sorted(df_result[up_col_prov].unique())
                   if up_col_prov and up_col_prov in df_result.columns
                   else ["Semua Wilayah"])

    for prov in prov_groups:
        if up_col_prov and up_col_prov in df_result.columns:
            df_prov = df_result[df_result[up_col_prov] == prov]
        else:
            df_prov = df_result

        top_strat  = df_prov["Strategi Terpilih"].mode().iloc[0] if not df_prov.empty else "-"
        badge_cls  = {"DTH":"badge-dth","OTT":"badge-ott",
                      "Hybrid":"badge-hybrid"}.get(top_strat,"badge-dth")
        model_used = df_prov["Model Digunakan"].iloc[0] if not df_prov.empty else "-"

        with st.expander(f"📍 {prov}  ({len(df_prov)} Kota/Kab)"):
            st.markdown(f"""
            <div style="margin-bottom:14px;">
              Strategi Dominan: <span class="{badge_cls}">{top_strat}</span>
              &nbsp;&nbsp;<span style="color:#64748b;font-size:0.82rem;">
                Model: {model_used}</span>
            </div>""", unsafe_allow_html=True)

            for c_col, lbl_m, val_m in zip(
                st.columns(3),
                ["📡 DTH","📺 OTT","🔀 Hybrid"],
                [len(df_prov[df_prov["Strategi Terpilih"]=="DTH"]),
                 len(df_prov[df_prov["Strategi Terpilih"]=="OTT"]),
                 len(df_prov[df_prov["Strategi Terpilih"]=="Hybrid"])]
            ):
                with c_col:
                    st.markdown(f"""<div class="metric-card">
                      <span class="metric-label">{lbl_m}</span>
                      <p class="metric-value">{val_m}</p></div>""",
                    unsafe_allow_html=True)

            st.write("")
            show_cols = []
            if up_col_kab and up_col_kab in df_prov.columns:
                show_cols.append(up_col_kab)
            show_cols += ["Strategi Terpilih","Prob Strategi","Produk DTH","Prob DTH",
                          "Produk OTT","Prob OTT","Rekomendasi Produk","Confidence"]
            show_cols = [c for c in show_cols if c in df_prov.columns]
            st.dataframe(df_prov[show_cols].reset_index(drop=True),
                         use_container_width=True)

    st.divider()

    # ── DOWNLOAD HASIL ────────────────────────────────────────────────────────
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_result.to_excel(writer, index=False, sheet_name='Hasil_Prediksi')

    dl_col, _ = st.columns([1, 2])
    with dl_col:
        st.download_button(
            "🚀 Download Hasil Prediksi (Excel)",
            data=output.getvalue(),
            file_name="hasil_prediksi_transvision.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ── SIDEBAR STATUS ─────────────────────────────────────────────────────────────
ready = len([k for k in all_models if k != "Seluruh Indonesia"])
st.sidebar.markdown(f"""
<div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);
            border-radius:10px;padding:16px;font-size:0.88rem;color:#94a3b8; 
            margin-top:10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <b style="color:#00d4ff; font-size:1rem; display:block; margin-bottom:12px;">📦 Status Model</b>
  <div style="line-height:1.8;">
      🗺️ Model spesifik: <b style="color:#f8fafc;">{ready} provinsi</b><br>
      🌍 Fallback: <b style="color:#f8fafc;">Seluruh Indonesia</b><br>
      ✅ Siap prediksi
  </div>
</div>
""", unsafe_allow_html=True)
