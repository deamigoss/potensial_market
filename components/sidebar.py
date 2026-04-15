<<<<<<< HEAD
import streamlit as st

def sidebar():
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"] { display: none; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 60%, #0a1628 100%);
        border-right: 1px solid rgba(0,212,255,0.1);
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] {
        border-radius: 10px;
        transition: all 0.3s ease;
        margin-bottom: 2px;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"]:hover {
        background: rgba(0,212,255,0.08) !important;
        border-left: 3px solid #00d4ff;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] p {
        color: #cbd5e1 !important;
        font-weight: 500;
        font-size: 0.92rem;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"]:hover p {
        color: #00d4ff !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.07);
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.image("assets/transvision.jpeg", use_container_width=True)

        st.markdown("""
        <div style="padding:4px 8px 2px 8px;">
          <p style="color:#64748b;font-size:0.72rem;font-weight:700;
                    letter-spacing:1.5px;text-transform:uppercase;margin:0;">
            NAVIGASI
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.page_link("app.py",                     label="🏠  Home")
        st.page_link("pages/eda.py",               label="📊  Exploratory Data Analysis")
        st.page_link("pages/model_performance.py", label="🤖  Model Performance")
        st.page_link("pages/prediction.py",        label="🔮  Prediction")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
                    border-radius:12px;padding:12px 14px;margin-bottom:10px;">
          <p style="color:#64748b;font-size:0.7rem;font-weight:700;letter-spacing:1.2px;
                    text-transform:uppercase;margin:0 0 8px 0;">INFO PROYEK</p>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span>📊</span>
            <span style="color:#94a3b8;font-size:0.82rem;">Dataset: <b style="color:#e2e8f0;">2022–2025</b></span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span>🤖</span>
            <span style="color:#94a3b8;font-size:0.82rem;">Model: <b style="color:#e2e8f0;">Logistic Regression</b></span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span>📡</span>
            <span style="color:#94a3b8;font-size:0.82rem;">Target: <b style="color:#e2e8f0;">4 Produk Transvision</b></span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span>🗓️</span>
            <span style="color:#94a3b8;font-size:0.82rem;">Proyeksi: <b style="color:#e2e8f0;">2026</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(29,110,239,0.12),rgba(151,45,180,0.12));
                    border:1px solid rgba(151,45,180,0.25);border-radius:12px;
                    padding:12px 14px;text-align:center;">
          <p style="margin:0 0 3px 0;color:#64748b;font-size:0.68rem;
                    font-weight:700;letter-spacing:1.2px;text-transform:uppercase;">DIBUAT OLEH</p>
          <p style="margin:0 0 2px 0;font-size:0.9rem;font-weight:700;
                    background:linear-gradient(90deg,#00d4ff,#972db4);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Intern Business Statistics
          </p>
          <p style="margin:0;color:#94a3b8;font-size:0.78rem;">
            Institut Teknologi Sepuluh Nopember · 2026
          </p>
        </div>
=======
import streamlit as st

def sidebar():
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"] { display: none; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 60%, #0a1628 100%);
        border-right: 1px solid rgba(0,212,255,0.1);
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] {
        border-radius: 10px;
        transition: all 0.3s ease;
        margin-bottom: 2px;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"]:hover {
        background: rgba(0,212,255,0.08) !important;
        border-left: 3px solid #00d4ff;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"] p {
        color: #cbd5e1 !important;
        font-weight: 500;
        font-size: 0.92rem;
    }
    [data-testid="stSidebar"] [data-testid="stPageLink"]:hover p {
        color: #00d4ff !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.07);
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.image("assets/transvision.jpeg", use_container_width=True)

        st.markdown("""
        <div style="padding:4px 8px 2px 8px;">
          <p style="color:#64748b;font-size:0.72rem;font-weight:700;
                    letter-spacing:1.5px;text-transform:uppercase;margin:0;">
            NAVIGASI
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.page_link("app.py",                     label="🏠  Home")
        st.page_link("pages/eda.py",               label="📊  Exploratory Data Analysis")
        st.page_link("pages/model_performance.py", label="🤖  Model Performance")
        st.page_link("pages/prediction.py",        label="🔮  Prediction")

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
                    border-radius:12px;padding:12px 14px;margin-bottom:10px;">
          <p style="color:#64748b;font-size:0.7rem;font-weight:700;letter-spacing:1.2px;
                    text-transform:uppercase;margin:0 0 8px 0;">INFO PROYEK</p>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span>📊</span>
            <span style="color:#94a3b8;font-size:0.82rem;">Dataset: <b style="color:#e2e8f0;">2022–2025</b></span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span>🤖</span>
            <span style="color:#94a3b8;font-size:0.82rem;">Model: <b style="color:#e2e8f0;">Logistic Regression</b></span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span>📡</span>
            <span style="color:#94a3b8;font-size:0.82rem;">Target: <b style="color:#e2e8f0;">4 Produk Transvision</b></span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span>🗓️</span>
            <span style="color:#94a3b8;font-size:0.82rem;">Proyeksi: <b style="color:#e2e8f0;">2026</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(29,110,239,0.12),rgba(151,45,180,0.12));
                    border:1px solid rgba(151,45,180,0.25);border-radius:12px;
                    padding:12px 14px;text-align:center;">
          <p style="margin:0 0 3px 0;color:#64748b;font-size:0.68rem;
                    font-weight:700;letter-spacing:1.2px;text-transform:uppercase;">DIBUAT OLEH</p>
          <p style="margin:0 0 2px 0;font-size:0.9rem;font-weight:700;
                    background:linear-gradient(90deg,#00d4ff,#972db4);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Intern Business Statistics
          </p>
          <p style="margin:0;color:#94a3b8;font-size:0.78rem;">
            Institut Teknologi Sepuluh Nopember · 2026
          </p>
        </div>
>>>>>>> b475ac26a41bac64202d5dd694929abe2b8af603
        """, unsafe_allow_html=True)