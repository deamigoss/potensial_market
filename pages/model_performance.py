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
.stExpander{background:rgba(30,41,59,0.4)!important;border:1px solid rgba(0,212,255,0.2)!important;border-radius:12px!important;}
.equation-box{background:rgba(15,23,42,0.7);border:1px solid rgba(0,212,255,0.3);border-radius:16px;padding:24px 28px;margin:12px 0;overflow-x:auto;}
.eq-label{color:#94a3b8;font-size:0.82rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;display:block;}
.eq-formula{color:#e2e8f0;font-size:1.05rem;line-height:2.2;font-family:'Courier New', monospace;}
.eq-coef-pos{color:#00d4ff;font-weight:700;}
.eq-coef-neg{color:#f87171;font-weight:700;}
.eq-var{color:#c084fc;}
.eq-op{color:#64748b;margin:0 4px;}
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
label_color  = {"DTH": "#00d4ff", "OTT": "#c084fc", "Hybrid": "#60a5fa"}
FEATURE_BISNIS = {
    COL_HH    : ("🏠","Jumlah Rumah Tangga","Proxy ukuran pasar."),
    COL_SINYAL: ("📶","Penetrasi Sinyal 4G/5G","Infrastruktur internet kuat mendorong OTT."),
    COL_UMP   : ("💰","Upah Minimum Provinsi","Mencerminkan daya beli wilayah."),
    COL_PDRB  : ("📈","PDRB per Kapita","Kemakmuran ekonomi wilayah."),
    COL_EXP   : ("🛒","Pengeluaran Non-Makanan","Proxy kesediaan membayar hiburan."),
    COL_TV    : ("📺","Cakupan TV Digital","Kematangan infrastruktur media."),
}
SHORT_NAMES = {
    COL_HH    : "HH",
    COL_SINYAL: "Signal",
    COL_UMP   : "UMP",
    COL_PDRB  : "PDRB",
    COL_EXP   : "Exp",
    COL_TV    : "TV",
}

# ══════════════════════════════════════════════════════════════════════════════
# STATSMODELS WRAPPER 
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: ambil koefisien langsung dari sm_result.params
# ══════════════════════════════════════════════════════════════════════════════
def _get_coef_safe(model_asset, var_names):
    """
    Return (intercept: float, coefs: list, names: list) dari sm_result.params.
    Fallback ke wrapper.coef_ jika _result tidak ada.
    Untuk MNLogit return list of (intercept, coefs, names, class_col).
    """
    try:
        params = model_asset._result.params
        if isinstance(params, pd.Series):
            intercept = float(params['const'])
            coefs     = [float(params[nm]) for nm in var_names if nm in params.index]
            names     = [nm for nm in var_names if nm in params.index]
            return intercept, coefs, names
        elif isinstance(params, pd.DataFrame):
            results = []
            for col in params.columns:
                intercept = float(params.loc['const', col])
                coefs     = [float(params.loc[nm, col]) for nm in var_names if nm in params.index]
                names     = [nm for nm in var_names if nm in params.index]
                results.append((intercept, coefs, names, col))
            return results
    except Exception:
        pass

    # Fallback wrapper.coef_
    try:
        coef   = model_asset.coef_
        if hasattr(coef, 'values'): coef = coef.values
        if coef.ndim == 2: coef = coef[0]
        interc = model_asset.intercept_
        interc = float(interc[0]) if hasattr(interc, '__len__') else float(interc)
        return interc, list(coef), var_names
    except Exception:
        return 0.0, [], var_names


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: persamaan & probabilitas HTML
# ══════════════════════════════════════════════════════════════════════════════
def _build_equation_html(intercept, coefs, var_names, lhs_label="g(x)"):
    def _fmt_coef(v):
        cls  = "eq-coef-pos" if v >= 0 else "eq-coef-neg"
        sign = "+" if v >= 0 else "−"
        return f'<span class="{cls}">{sign}{abs(v):.3f}</span>'

    def _fmt_var(name):
        short = SHORT_NAMES.get(name, name.split("(")[0].strip()[:15])
        return f'<span class="eq-var"> · {short}</span>'

    b0_cls  = "eq-coef-pos" if intercept >= 0 else "eq-coef-neg"
    b0_sign = "+" if intercept >= 0 else "−"
    parts   = [f'<span class="{b0_cls}">{b0_sign}{abs(intercept):.3f}</span>']
    for c, nm in zip(coefs, var_names):
        parts.append(f'{_fmt_coef(c)}{_fmt_var(nm)}')

    eq_body = ' '.join(parts)
    return f"""
    <div class="equation-box">
      <span class="eq-label">📐 {lhs_label}</span>
      <div class="eq-formula">
        <span style="color:#60a5fa;font-weight:700;">{lhs_label}</span>
        <span class="eq-op"> = </span>
        {eq_body}
      </div>
    </div>"""


def _build_prob_equation_html(lhs_label, numerator_label, extra_label=""):
    return f"""
    <div class="equation-box" style="text-align:center;">
      <span class="eq-label">📊 Probabilitas {extra_label}</span>
      <div class="eq-formula" style="font-size:1.1rem;">
        <span style="color:#60a5fa;font-weight:700;">{lhs_label}</span>
        <span class="eq-op"> = </span>
        <span>e<sup style="color:#00d4ff;">{numerator_label}</sup>
        &nbsp;/&nbsp;
        (1 + e<sup style="color:#00d4ff;">{numerator_label}</sup>)</span>
      </div>
    </div>"""


def _build_multinomial_prob_html(n_classes, class_names):
    if n_classes == 2:
        return ""
    denom = "1 + e<sup>g₁(x)</sup> + e<sup>g₂(x)</sup>" if n_classes == 3 else "Σ denominator"
    html  = f"""
    <div class="equation-box" style="text-align:center;">
      <span class="eq-label">📊 Probabilitas Kelas Referensi ({class_names[0]})</span>
      <div class="eq-formula" style="font-size:1.05rem;">
        π₀(x) = 1 / ({denom})
      </div>
    </div>"""
    for i in range(1, n_classes):
        html += f"""
    <div class="equation-box" style="text-align:center;margin-top:8px;">
      <span class="eq-label">📊 Probabilitas {class_names[i]}</span>
      <div class="eq-formula" style="font-size:1.05rem;">
        π{i}(x) = e<sup>g{i}(x)</sup> / ({denom})
      </div>
    </div>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: safe extract statistik koefisien dari sm_result (fix KeyError MNLogit)
# ══════════════════════════════════════════════════════════════════════════════
def _extract_sm_stats(sm_result, var_names, is_multinomial=False):
    rows   = []
    params = sm_result.params
    try:
        pvalues = sm_result.pvalues
        bse     = sm_result.bse
    except Exception:
        pvalues = None
        bse     = None

    if isinstance(params, pd.DataFrame):
        # MNLogit — params adalah DataFrame
        for col in params.columns:
            for nm in params.index:
                if nm == "const":
                    continue
                b    = params.loc[nm, col]
                se   = bse.loc[nm, col]     if (bse     is not None and isinstance(bse, pd.DataFrame))     else np.nan
                pv   = pvalues.loc[nm, col] if (pvalues is not None and isinstance(pvalues, pd.DataFrame)) else np.nan
                OR   = np.exp(b)
                wald = (b / se) ** 2 if (not np.isnan(se) and se != 0) else np.nan
                rows.append({
                    "Kelas"     : str(col),
                    "Variabel"  : nm,
                    "β (Koef.)" : f"{b:.4f}",
                    "Std. Error": f"{se:.4f}" if not np.isnan(se) else "—",
                    "Wald χ²"   : f"{wald:.3f}" if not np.isnan(wald) else "—",
                    "p-value"   : f"{pv:.4f}"   if not np.isnan(pv)   else "—",
                    "Odds Ratio": f"{OR:.4f}",
                    "Sig."      : "✅" if (not np.isnan(pv) and pv < 0.05) else ("⚠️" if (not np.isnan(pv) and pv < 0.10) else "—"),
                })
    elif isinstance(params, pd.Series):
        # Logit biner — params adalah Series
        for nm in params.index:
            if nm == "const":
                continue
            b    = params[nm]
            se   = bse[nm]     if bse     is not None else np.nan
            pv   = pvalues[nm] if pvalues is not None else np.nan
            OR   = np.exp(b)
            wald = (b / se) ** 2 if (not np.isnan(se) and se != 0) else np.nan
            rows.append({
                "Variabel"  : nm,
                "β (Koef.)" : f"{b:.4f}",
                "Std. Error": f"{se:.4f}" if not np.isnan(se) else "—",
                "Wald χ²"   : f"{wald:.3f}" if not np.isnan(wald) else "—",
                "p-value"   : f"{pv:.4f}"   if not np.isnan(pv)   else "—",
                "Odds Ratio": f"{OR:.4f}",
                "Sig."      : "✅" if (not np.isnan(pv) and pv < 0.05) else ("⚠️" if (not np.isnan(pv) and pv < 0.10) else "—"),
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: confusion matrix & classification report
# ══════════════════════════════════════════════════════════════════════════════
def _confusion_fig(y_true, y_pred_arr, lbl_map):
    classes = sorted(set(list(y_true) + list(y_pred_arr)))
    names   = [lbl_map.get(c, str(c)) for c in classes]
    cm      = confusion_matrix(y_true, y_pred_arr, labels=classes)
    fig = px.imshow(
        cm, text_auto=True, x=names, y=names,
        color_continuous_scale=[[0, "rgba(30,41,59,0.4)"], [1, "#1d6eef"]],
        labels=dict(x="Prediksi", y="Aktual"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"), margin=dict(t=30, b=10, l=10, r=10),
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(size=13, color="#e2e8f0")),
        yaxis=dict(tickfont=dict(size=13, color="#e2e8f0")),
    )
    fig.update_traces(textfont_size=16)
    return fig


def _report_df(y_true, y_pred_arr, lbl_map):
    classes = sorted(set(list(y_true) + list(y_pred_arr)))
    names   = [lbl_map.get(c, str(c)) for c in classes]
    rpt     = classification_report(y_true, y_pred_arr, labels=classes,
                                    target_names=names, output_dict=True, zero_division=0)
    rows = []
    for nm in names:
        r = rpt.get(nm, {})
        rows.append({
            "Kelas"    : nm,
            "Precision": f"{r.get('precision', 0):.2f}",
            "Recall"   : f"{r.get('recall', 0):.2f}",
            "F1-Score" : f"{r.get('f1-score', 0):.2f}",
            "Support"  : int(r.get('support', 0)),
        })
    return pd.DataFrame(rows)


def _coef_bar(model_asset, vars_list, title=""):
    try:
        result = _get_coef_safe(model_asset, vars_list)
        if isinstance(result, list):
            # MNLogit — ambil kelas pertama
            _, coefs, names, _ = result[0]
        else:
            _, coefs, names = result
        df_coef = pd.DataFrame({"Variabel": names, "Koefisien": coefs})
        df_coef["Warna"] = df_coef["Koefisien"].apply(lambda v: "#00d4ff" if v > 0 else "#f87171")
        df_coef = df_coef.sort_values("Koefisien")
        fig = go.Figure(go.Bar(
            x=df_coef["Koefisien"], y=df_coef["Variabel"],
            orientation="h",
            marker_color=df_coef["Warna"],
            text=df_coef["Koefisien"].round(3),
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=12),
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(color="#94a3b8", size=13)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            margin=dict(t=40, b=10, l=10, r=40),
            xaxis=dict(gridcolor="rgba(255,255,255,0.07)",
                       zerolinecolor="rgba(255,255,255,0.2)",
                       tickfont=dict(color="#94a3b8")),
            yaxis=dict(tickfont=dict(color="#e2e8f0", size=11)),
            height=max(250, 60 * len(names)),
        )
        return fig
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: statistik koefisien block (reusable)
# ══════════════════════════════════════════════════════════════════════════════
def _render_stat_table(model_asset, valid_vars, is_multinomial=False, key_suffix=""):
    try:
        sm_res = model_asset._result
        rows   = _extract_sm_stats(sm_res, valid_vars, is_multinomial)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except AttributeError:
        try:
            coef = model_asset.coef_
            if coef.ndim == 2: coef = coef[0]
            rows = [{"Variabel": v, "β (Koef.)": f"{c:.4f}", "Odds Ratio": f"{np.exp(c):.4f}"}
                    for v, c in zip(valid_vars, coef)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("ℹ️ p-value tidak tersedia (model fallback sklearn).")
        except Exception:
            st.info("Detail statistik tidak tersedia.")

    with st.expander("📖 Panduan Membaca Tabel Koefisien"):
        st.markdown("""
| Kolom | Arti |
|-------|------|
| **β (Koef.)** | Arah & besar pengaruh. Positif = meningkatkan peluang kelas target |
| **Std. Error** | Ketidakpastian estimasi koefisien |
| **Wald χ²** | (β/SE)² — statistik uji signifikansi |
| **p-value** | < 0.05 = signifikan ✅, < 0.10 = marginally significant ⚠️ |
| **Odds Ratio** | exp(β) — perubahan odds per 1 unit variabel (terstandarisasi) |
""")


# ══════════════════════════════════════════════════════════════════════════════
# SUB-MODEL RENDER — dipakai di Tab DTH & Tab OTT
# ══════════════════════════════════════════════════════════════════════════════
def _render_submodel_section(
    model_asset, scaler_asset, label_col, lbl_map,
    df_scope, vars_list, lhs_label,
    use_notna_col=None, warn_sep=False, key_suffix=""
):
    # Filter data
    if use_notna_col and use_notna_col in df_scope.columns:
        d = df_scope[df_scope[label_col].notna() & df_scope[use_notna_col].notna()]
    else:
        d = df_scope[df_scope[label_col].notna()]

    if d.empty or len(d[label_col].unique()) < 2:
        st.warning("⚠️ Data tidak cukup untuk evaluasi sub-model ini.")
        return

    valid_vars = [v for v in vars_list if v in d.columns]
    X_sub      = pd.DataFrame(
        scaler_asset.transform(d[valid_vars]),
        columns=valid_vars, index=d.index
    )
    y_sub      = d[label_col]
    y_sub_pred = model_asset.predict(X_sub)
    acc_sub    = accuracy_score(y_sub, y_sub_pred)

    # ── Metrik (langsung di bawah judul) ────────────────────────────────
    kelas_str = " vs ".join([lbl_map.get(k, str(k)) for k in sorted(lbl_map)])
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="metric-card">
          <span class="metric-label">🎯 Akurasi</span>
          <p class="metric-value">{acc_sub:.1%}</p>
          <span class="metric-sub">Ketepatan klasifikasi</span></div>""",
          unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
          <span class="metric-label">📊 Data Uji</span>
          <p class="metric-value">{len(d)}</p>
          <span class="metric-sub">Baris yang dievaluasi</span></div>""",
          unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
          <span class="metric-label">🏷️ Kelas</span>
          <p class="metric-value" style="font-size:1.3rem;">{kelas_str}</p>
          <span class="metric-sub">Label sub-produk</span></div>""",
          unsafe_allow_html=True)

    st.write("")

    # ── Persamaan g(x) ──────────────────────────────────────────────────
    result = _get_coef_safe(model_asset, valid_vars)
    if isinstance(result, tuple) and len(result) == 3:
        intercept, coefs, names = result
        st.markdown(
            _build_equation_html(intercept, coefs, names, lhs_label=lhs_label),
            unsafe_allow_html=True
        )
    else:
        st.info("Persamaan tidak tersedia.")

    # ── Rumus probabilitas ──────────────────────────────────────────────
    st.markdown(
        _build_prob_equation_html("π(x)", lhs_label, f"({kelas_str})"),
        unsafe_allow_html=True
    )

    # ── Bar chart (hidden) ──────────────────────────────────────────────
    with st.expander("📊 Lihat Bar Chart Koefisien"):
        fig = _coef_bar(model_asset, valid_vars, f"Koefisien — {lhs_label}")
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"bar_{key_suffix}")

    if warn_sep:
        st.markdown("""<div class="warn-box"><p>
⚠️ <b>Complete Separation:</b> Koefisien DTH besar karena fitur dapat memisahkan
kelas dengan sempurna. Ini valid secara statistik dan konsisten dengan output SPSS.
</p></div>""", unsafe_allow_html=True)

    st.write("")

    # ── Statistik koefisien ─────────────────────────────────────────────
    st.markdown("#### 🔬 Statistik Koefisien")
    _render_stat_table(model_asset, valid_vars, is_multinomial=False, key_suffix=key_suffix)

    # ── Confusion Matrix & Report ───────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 Confusion Matrix")
    cf, rf = st.columns([1.1, 1])
    with cf:
        st.markdown("##### 🔲 Confusion Matrix")
        st.plotly_chart(
            _confusion_fig(y_sub, y_sub_pred, lbl_map),
            use_container_width=True, key=f"cm_{key_suffix}"
        )
    with rf:
        st.markdown("##### 📋 Classification Report")
        st.dataframe(_report_df(y_sub, y_sub_pred, lbl_map),
                     use_container_width=True, hide_index=True)
        with st.expander("📖 Panduan Membaca Tabel"):
            st.markdown("""
| Metrik | Arti |
|--------|------|
| **Precision** | Dari prediksi kelas ini, berapa % yang benar? |
| **Recall** | Dari aktual kelas ini, berapa % yang tertangkap? |
| **F1-Score** | Rata-rata harmonis Precision & Recall |
| **Support** | Jumlah data aktual kelas ini |
""")


# ══════════════════════════════════════════════════════════════════════════════
# RENDER UTAMA
# ══════════════════════════════════════════════════════════════════════════════
st.title("⚙️ Model Performance & Evaluasi")

st.markdown("""
<div style="background:rgba(30,41,59,0.45);padding:22px 26px;border-radius:18px;
            border-left:5px solid #1d6eef;border:1px solid rgba(255,255,255,0.06);margin-bottom:8px;">
  <p style="margin:0 0 8px 0;font-size:1.1rem;font-weight:700;color:#f8fafc;">🧠 Sistem Pre-Training Otomatis</p>
  <p style="margin:0;font-size:0.93rem;color:#cbd5e1;line-height:1.75;">
    Model dilatih <b>otomatis saat aplikasi dibuka</b>.<br>
    Menggunakan <b>Regresi Logistik</b> dengan seleksi fitur via <b>Backward Wald</b>.
  </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("⚙️ Melatih semua model provinsi..."):
    all_models, df_train = pretrain_all_models()

st.success(f"✅ **{len(all_models)} model** berhasil dilatih (Nasional + per-Provinsi).")
st.divider()

# ── ALUR ──────────────────────────────────────────────────────────────────────
st.subheader("🗺️ Alur Klasifikasi Hierarkis")
c1, a1, c2, a2, c3 = st.columns([2, 0.3, 2, 0.3, 2])
with c1:
    st.markdown("""<div class="flow-step"><h4>① Input Wilayah</h4>
    <p>Data sosial-ekonomi kota/kab: PDRB, UMP, sinyal, household, pengeluaran, TV digital.</p>
    </div>""", unsafe_allow_html=True)
with a1:
    st.markdown("<div style='text-align:center;font-size:1.8rem;color:#1d6eef;padding-top:18px;'>→</div>",
                unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="flow-step" style="border-color:rgba(151,45,180,0.4);background:rgba(151,45,180,0.07);">
    <h4 style="color:#c084fc;">② Model Strategi</h4>
    <p>Regresi Logistik + Backward Wald.<br>Koefisien konsisten dengan output SPSS.</p>
    </div>""", unsafe_allow_html=True)
with a2:
    st.markdown("<div style='text-align:center;font-size:1.8rem;color:#972db4;padding-top:18px;'>→</div>",
                unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="flow-step" style="border-color:rgba(96,165,250,0.4);background:rgba(96,165,250,0.07);">
    <h4 style="color:#93c5fd;">③ Model Produk (Biner)</h4>
    <p>DTH → Satellite vs Nusantara<br>OTT → Seru vs 2nd Gen<br>Hybrid → keduanya digabung.</p>
    </div>""", unsafe_allow_html=True)
st.divider()

# ── FILTER PROVINSI ───────────────────────────────────────────────────────────
col_sel, col_badge = st.columns([2.5, 1], vertical_alignment="center")
with col_sel:
    list_prov_ui  = ["Seluruh Indonesia"] + sorted(df_train["Provinsi"].unique())
    selected_prov = st.selectbox("📍 Pilih Wilayah Evaluasi", list_prov_ui, key="prov_selector")
with col_badge:
    in_cache  = selected_prov in all_models
    badge_cls = "badge-ready" if in_cache else "badge-missing"
    badge_txt = "✅ Model Tersedia" if in_cache else "❌ Tidak Tersedia"
    st.markdown(f"""
    <div style="background:rgba(30,41,59,0.6);backdrop-filter:blur(15px);
                border:1px solid rgba(0,212,255,0.25);border-radius:14px;
                padding:10px 16px;display:flex;flex-direction:column;gap:4px;">
      <span style="color:#94a3b8;font-size:0.75rem;font-weight:600;
                   text-transform:uppercase;letter-spacing:1px;">Metode</span>
      <span style="color:#00d4ff;font-size:0.95rem;font-weight:700;">📐 Regresi Logistik</span>
      <span class="status-badge {badge_cls}" style="font-size:0.8rem;margin-top:2px;
             align-self:flex-start;">{badge_txt}</span>
    </div>""", unsafe_allow_html=True)

if selected_prov not in all_models:
    st.error(f"Model untuk **{selected_prov}** tidak tersedia.")
    st.stop()

assets        = all_models[selected_prov]
selected_vars = assets['selected_vars']
df_eval       = df_train if selected_prov == "Seluruh Indonesia" \
                else df_train[df_train["Provinsi"] == selected_prov]

feat_eval = [c for c in selected_vars if c in df_eval.columns]
X_scaled  = pd.DataFrame(
    assets['scaler_strategi'].transform(df_eval[feat_eval]),
    columns=feat_eval, index=df_eval.index
)
y           = df_eval["label_kategori"]
num_classes = len(y.unique())
y_pred      = assets['model_strategi'].predict(X_scaled)
acc         = accuracy_score(y, y_pred)
present_ids = sorted(y.unique())
labels_str  = " · ".join([label_map[i] for i in present_ids if i in label_map])

# ── METRICS ───────────────────────────────────────────────────────────────────
st.write("")
for col, lbl, val, sub in zip(
    st.columns(4),
    ["🎯 Akurasi Model", "📊 Data Evaluasi", "🏷️ Kelas Strategi", "🔬 Fitur Aktif"],
    [f"{acc:.1%}", f"{len(df_eval)}", f"{num_classes}", f"{len(selected_vars)}"],
    [f"Ketepatan prediksi di {selected_prov}", "Kab/Kota yang dievaluasi",
     labels_str, "Terpilih via Backward Wald"]
):
    with col:
        st.markdown(f"""<div class="metric-card">
          <span class="metric-label">{lbl}</span>
          <p class="metric-value">{val}</p>
          <span class="metric-sub">{sub}</span></div>""", unsafe_allow_html=True)

st.write("")
acc_nar = (f"🟢 Akurasi **{acc:.1%}** — Model sangat handal." if acc >= 0.90
           else f"🟡 Akurasi **{acc:.1%}** — Cukup baik, disarankan validasi lapangan." if acc >= 0.75
           else f"🔴 Akurasi **{acc:.1%}** — Perlu perhatian, data mungkin terlalu sedikit.")
st.markdown(f'<div class="insight-box"><p>📌 <b>Interpretasi Bisnis:</b> {acc_nar}</p></div>',
            unsafe_allow_html=True)
st.divider()

# ── STATUS SEMUA MODEL ────────────────────────────────────────────────────────
st.subheader("📦 Status Model Semua Provinsi")
status_rows = []
for prov in ["Seluruh Indonesia"] + sorted(df_train["Provinsi"].unique()):
    if prov in all_models:
        pk     = all_models[prov]
        df_p   = df_train if prov == "Seluruh Indonesia" \
                 else df_train[df_train["Provinsi"] == prov]
        n_data = len(df_p[df_p["label_kategori"].notna()])
        try:
            sv   = pk['selected_vars']
            fv   = [c for c in sv if c in df_p.columns]
            Xp   = pd.DataFrame(
                pk['scaler_strategi'].transform(df_p[fv]),
                columns=fv, index=df_p.index
            )
            yp    = pk['model_strategi'].predict(Xp)
            acc_p = f"{accuracy_score(df_p['label_kategori'], yp):.1%}"
        except Exception:
            acc_p = "-"
        status_rows.append({
            "Provinsi"   : prov,
            "Status"     : "✅ Siap",
            "Data"       : n_data,
            "Fitur Aktif": len(pk['selected_vars']),
            "Akurasi"    : acc_p,
        })
    else:
        status_rows.append({
            "Provinsi": prov, "Status": "⚠️ Dilewati",
            "Data": "-", "Fitur Aktif": "-", "Akurasi": "-"
        })

st.dataframe(pd.DataFrame(status_rows), use_container_width=True, height=340)
ready_count = len([r for r in status_rows if "✅" in r["Status"]])
st.caption(f"Total **{ready_count} model aktif** dari {len(status_rows)} wilayah.")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Model Strategi",
    "📡 Model DTH",
    "📺 Model OTT",
])

n_classes   = len(present_ids)
class_names = [label_map.get(c, str(c)) for c in present_ids]

df_scope_eval = (df_train if selected_prov == "Seluruh Indonesia"
                 else df_train[df_train["Provinsi"] == selected_prov])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — MODEL STRATEGI
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### 📐 Persamaan Regresi Logistik — Model Strategi")

    # ── Metrik ringkasan (Akurasi, Data Uji, Kelas) ─────────────────────────
    kelas_str_strat = " / ".join(class_names)
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(f"""<div class="metric-card">
          <span class="metric-label">🎯 Akurasi</span>
          <p class="metric-value">{acc:.1%}</p>
          <span class="metric-sub">Ketepatan klasifikasi</span></div>""",
          unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""<div class="metric-card">
          <span class="metric-label">📊 Data Uji</span>
          <p class="metric-value">{len(df_eval)}</p>
          <span class="metric-sub">Baris yang dievaluasi</span></div>""",
          unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""<div class="metric-card">
          <span class="metric-label">🏷️ Kelas</span>
          <p class="metric-value" style="font-size:1.3rem;">{kelas_str_strat}</p>
          <span class="metric-sub">Label strategi</span></div>""",
          unsafe_allow_html=True)
    st.write("")

    try:
        result = _get_coef_safe(assets['model_strategi'], selected_vars)

        if n_classes == 2:
            # Binary Logit
            if isinstance(result, tuple) and len(result) == 3:
                intercept, coefs, names = result
            else:
                intercept, coefs, names = result[0][0], result[0][1], result[0][2]
            lbl = f"g(x) — Peluang {class_names[-1]}"
            st.markdown(_build_equation_html(intercept, coefs, names, lhs_label=lbl),
                        unsafe_allow_html=True)
            st.markdown(_build_prob_equation_html("π(x)", "g(x)", "(Probabilitas kelas target)"),
                        unsafe_allow_html=True)
        else:
            # Multinomial
            if isinstance(result, list):
                for i, (intercept, coefs, names, col) in enumerate(result):
                    lbl = f"g{i+1}(x) — Peluang {class_names[i+1] if i+1 < len(class_names) else str(col)}"
                    st.markdown(_build_equation_html(intercept, coefs, names, lhs_label=lbl),
                                unsafe_allow_html=True)
            st.markdown(_build_multinomial_prob_html(n_classes, class_names),
                        unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Persamaan tidak dapat ditampilkan: {e}")

    with st.expander("📊 Lihat Bar Chart Koefisien — Model Strategi"):
        fig_coef = _coef_bar(assets['model_strategi'], selected_vars, "Koefisien — Model Strategi")
        if fig_coef:
            st.plotly_chart(fig_coef, use_container_width=True, key="bar_strategi")

    st.divider()

    # Statistik Koefisien
    st.markdown("#### 🔬 Statistik Koefisien (Backward Wald)")
    _render_stat_table(assets['model_strategi'], selected_vars,
                       is_multinomial=(n_classes > 2), key_suffix="strategi")

    st.divider()

    # Confusion Matrix
    st.markdown("#### 📊 Confusion Matrix — Model Strategi")
    colA, colB = st.columns([1.1, 1])
    with colA:
        st.plotly_chart(
            _confusion_fig(y, y_pred, label_map),
            use_container_width=True, key="cm_strategi"
        )
    with colB:
        st.markdown("##### 📋 Classification Report")
        st.dataframe(_report_df(y, y_pred, label_map),
                     use_container_width=True, hide_index=True)
        with st.expander("📖 Panduan Membaca Tabel"):
            st.markdown("""
| Metrik | Arti |
|--------|------|
| **Precision** | Dari semua prediksi kelas ini, berapa % yang benar? |
| **Recall** | Dari semua aktual kelas ini, berapa % yang tertangkap? |
| **F1-Score** | Rata-rata harmonis Precision & Recall |
| **Support** | Jumlah data aktual kelas ini |
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL DTH
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 📡 Model DTH — Satellite vs Nusantara")
    vars_dth = getattr(assets['model_produk_dth'], '_sub_vars',
                       assets['selected_vars'])
    _render_submodel_section(
        model_asset   = assets['model_produk_dth'],
        scaler_asset  = assets['scaler_dth'],
        label_col     = "label_dth",
        lbl_map       = assets['label_map_dth'],
        df_scope      = df_scope_eval,
        vars_list     = vars_dth,
        lhs_label     = "g_DTH(x)",
        use_notna_col = None,
        warn_sep      = True,
        key_suffix    = f"dth_{selected_prov}",
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL OTT
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 📺 Model OTT — Seru vs 2nd Gen")
    vars_ott = getattr(assets['model_produk_ott'], '_sub_vars',
                       assets['selected_vars'])
    _render_submodel_section(
        model_asset   = assets['model_produk_ott'],
        scaler_asset  = assets['scaler_ott'],
        label_col     = "label_ott",
        lbl_map       = assets['label_map_ott'],
        df_scope      = df_scope_eval,
        vars_list     = vars_ott,
        lhs_label     = "g_OTT(x)",
        use_notna_col = "OTT",
        warn_sep      = False,
        key_suffix    = f"ott_{selected_prov}",
    )

st.divider()
