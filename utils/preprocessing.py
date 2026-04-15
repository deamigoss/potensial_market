<<<<<<< HEAD
import pandas as pd

# ======================
# NORMALISASI KOLOM
# ======================
COL_RENAME = {
    "JUMLAH HOUSEHOLD"                                           : "Jumlah Household",
    "% JUMLAH DESA/KELURAHAN YANG MENERIMA SINYAL 5G/4G/LTE"   : "Jumlah Desa/Kelurahan Yang Menerima Sinyal 5G/4G/LTE (Persen)",
    "UMP"                                                        : "UMP (Rupiah)",
    "PDRB"                                                       : "PDRB (Ribu Rp)",
    "Rata-Rata Pengeluaran per Kapita BUKAN MANAKAN"            : "Rata-Rata Pengeluaran per Kapita Bukan Makanan (Rupiah)",
}

FEATURE_COLS = [
    "Jumlah Household",
    "Jumlah Desa/Kelurahan Yang Menerima Sinyal 5G/4G/LTE (Persen)",
    "UMP (Rupiah)",
    "PDRB (Ribu Rp)",
    "Rata-Rata Pengeluaran per Kapita Bukan Makanan (Rupiah)",
    "Tercover TV Digital?",
]

# ======================
# NORMALISASI PRODUK
# ======================
def normalize_produk(df):
    if "DTH/ MASS" in df.columns:
        df["DTH/ MASS"] = df["DTH/ MASS"].astype(str).str.lower().str.strip()
    if "OTT" in df.columns:
        df["OTT"] = df["OTT"].astype(str).str.lower().str.strip()
        df["OTT"] = df["OTT"].replace({
            "2nd gen, seru": "2nd gen",
            "seru, 2nd gen": "2nd gen",
        })
    return df

# ======================
# CLEAN NUMERIC
# ======================
def clean_numeric_columns(df):
    # Kandidat nama kolom (sebelum DAN sesudah rename)
    candidates = list(COL_RENAME.keys()) + list(COL_RENAME.values()) + [
        "Jumlah Household",
        "Jumlah Desa/Kelurahan Yang Menerima Sinyal 5G/4G/LTE (Persen)",
        "UMP (Rupiah)", "PDRB (Ribu Rp)",
        "Rata-Rata Pengeluaran per Kapita Bukan Makanan (Rupiah)",
    ]
    for col in candidates:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ======================
# ENCODE TV DIGITAL
# ======================
def encode_tv_digital(df):
    if "Tercover TV Digital?" in df.columns:
        df["Tercover TV Digital?"] = df["Tercover TV Digital?"].map({"Ya": 1, "Tidak": 0})
        df["Tercover TV Digital?"] = pd.to_numeric(df["Tercover TV Digital?"], errors="coerce")
    return df

# ======================
# LABEL BUILDERS
# ======================
def create_label_kategori(row):
    dth = str(row.get("DTH/ MASS","")).lower().strip()
    ott = str(row.get("OTT","")).lower().strip()
    has_dth = dth in ["satellite", "nusantara"]
    has_ott = ott in ["seru", "2nd gen"]
    no_dth  = dth in ["nan", "", "none", "-"]
    no_ott  = ott in ["nan", "", "none", "-"]
    if has_dth and no_ott : return 1   # DTH Only
    if has_ott and no_dth : return 2   # OTT Only
    if has_dth and has_ott: return 3   # Hybrid
    return None

def create_label_dth(row):
    dth = str(row.get("DTH/ MASS","")).lower().strip()
    if dth == "satellite" : return 0
    if dth == "nusantara" : return 1
    return None

def create_label_ott(row):
    ott = str(row.get("OTT","")).lower().strip()
    if ott == "seru"   : return 0
    if ott == "2nd gen": return 1
    return None

# ======================
# PIPELINE UTAMA
# ======================
def preprocessing_pipeline(df):
    # 1. Rename kolom pendek jika ada
    df = df.rename(columns={k:v for k,v in COL_RENAME.items() if k in df.columns})
    # 2. Bersihkan produk & angka
    df = normalize_produk(df)
    df = clean_numeric_columns(df)
    df = encode_tv_digital(df)
    # 3. Buat label (hanya jika kolom asli ada — aman untuk data prediksi baru)
    if "DTH/ MASS" in df.columns and "OTT" in df.columns:
        df["label_kategori"] = df.apply(create_label_kategori, axis=1)
        df["label_dth"]      = df.apply(create_label_dth, axis=1)
        df["label_ott"]      = df.apply(create_label_ott, axis=1)
    # 4. Isi NaN pada kolom fitur saja (bukan label)
    non_label = [c for c in df.columns if c not in ["label_kategori","label_dth","label_ott"]]
    df[non_label] = df[non_label].fillna(0)
    return df, ["Satellite", "Nusantara", "Seru", "2nd Gen"]

# ======================
# AMBIL DATA TRAINING
# ======================
def get_training_data(df, produk_cols=None):
    if "label_kategori" not in df.columns:
        return df
=======
import pandas as pd

# ======================
# NORMALISASI KOLOM
# ======================
COL_RENAME = {
    "JUMLAH HOUSEHOLD"                                           : "Jumlah Household",
    "% JUMLAH DESA/KELURAHAN YANG MENERIMA SINYAL 5G/4G/LTE"   : "Jumlah Desa/Kelurahan Yang Menerima Sinyal 5G/4G/LTE (Persen)",
    "UMP"                                                        : "UMP (Rupiah)",
    "PDRB"                                                       : "PDRB (Ribu Rp)",
    "Rata-Rata Pengeluaran per Kapita BUKAN MANAKAN"            : "Rata-Rata Pengeluaran per Kapita Bukan Makanan (Rupiah)",
}

FEATURE_COLS = [
    "Jumlah Household",
    "Jumlah Desa/Kelurahan Yang Menerima Sinyal 5G/4G/LTE (Persen)",
    "UMP (Rupiah)",
    "PDRB (Ribu Rp)",
    "Rata-Rata Pengeluaran per Kapita Bukan Makanan (Rupiah)",
    "Tercover TV Digital?",
]

# ======================
# NORMALISASI PRODUK
# ======================
def normalize_produk(df):
    if "DTH/ MASS" in df.columns:
        df["DTH/ MASS"] = df["DTH/ MASS"].astype(str).str.lower().str.strip()
    if "OTT" in df.columns:
        df["OTT"] = df["OTT"].astype(str).str.lower().str.strip()
        df["OTT"] = df["OTT"].replace({
            "2nd gen, seru": "2nd gen",
            "seru, 2nd gen": "2nd gen",
        })
    return df

# ======================
# CLEAN NUMERIC
# ======================
def clean_numeric_columns(df):
    # Kandidat nama kolom (sebelum DAN sesudah rename)
    candidates = list(COL_RENAME.keys()) + list(COL_RENAME.values()) + [
        "Jumlah Household",
        "Jumlah Desa/Kelurahan Yang Menerima Sinyal 5G/4G/LTE (Persen)",
        "UMP (Rupiah)", "PDRB (Ribu Rp)",
        "Rata-Rata Pengeluaran per Kapita Bukan Makanan (Rupiah)",
    ]
    for col in candidates:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ======================
# ENCODE TV DIGITAL
# ======================
def encode_tv_digital(df):
    if "Tercover TV Digital?" in df.columns:
        df["Tercover TV Digital?"] = df["Tercover TV Digital?"].map({"Ya": 1, "Tidak": 0})
        df["Tercover TV Digital?"] = pd.to_numeric(df["Tercover TV Digital?"], errors="coerce")
    return df

# ======================
# LABEL BUILDERS
# ======================
def create_label_kategori(row):
    dth = str(row.get("DTH/ MASS","")).lower().strip()
    ott = str(row.get("OTT","")).lower().strip()
    has_dth = dth in ["satellite", "nusantara"]
    has_ott = ott in ["seru", "2nd gen"]
    no_dth  = dth in ["nan", "", "none", "-"]
    no_ott  = ott in ["nan", "", "none", "-"]
    if has_dth and no_ott : return 1   # DTH Only
    if has_ott and no_dth : return 2   # OTT Only
    if has_dth and has_ott: return 3   # Hybrid
    return None

def create_label_dth(row):
    dth = str(row.get("DTH/ MASS","")).lower().strip()
    if dth == "satellite" : return 0
    if dth == "nusantara" : return 1
    return None

def create_label_ott(row):
    ott = str(row.get("OTT","")).lower().strip()
    if ott == "seru"   : return 0
    if ott == "2nd gen": return 1
    return None

# ======================
# PIPELINE UTAMA
# ======================
def preprocessing_pipeline(df):
    # 1. Rename kolom pendek jika ada
    df = df.rename(columns={k:v for k,v in COL_RENAME.items() if k in df.columns})
    # 2. Bersihkan produk & angka
    df = normalize_produk(df)
    df = clean_numeric_columns(df)
    df = encode_tv_digital(df)
    # 3. Buat label (hanya jika kolom asli ada — aman untuk data prediksi baru)
    if "DTH/ MASS" in df.columns and "OTT" in df.columns:
        df["label_kategori"] = df.apply(create_label_kategori, axis=1)
        df["label_dth"]      = df.apply(create_label_dth, axis=1)
        df["label_ott"]      = df.apply(create_label_ott, axis=1)
    # 4. Isi NaN pada kolom fitur saja (bukan label)
    non_label = [c for c in df.columns if c not in ["label_kategori","label_dth","label_ott"]]
    df[non_label] = df[non_label].fillna(0)
    return df, ["Satellite", "Nusantara", "Seru", "2nd Gen"]

# ======================
# AMBIL DATA TRAINING
# ======================
def get_training_data(df, produk_cols=None):
    if "label_kategori" not in df.columns:
        return df
>>>>>>> b475ac26a41bac64202d5dd694929abe2b8af603
    return df[df["label_kategori"].notna() & df["label_kategori"].isin([1,2,3])].copy()