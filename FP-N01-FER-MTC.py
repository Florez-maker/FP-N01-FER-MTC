import warnings
warnings.filterwarnings("ignore")

import io
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# ════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════

st.set_page_config(
    page_title="Calculadora de Fertilización — FarmPrecision",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#1b60a7",
    "success": "#2ca02c",
    "danger":  "#d62728",
    "warning": "#F1C40F",
    "info":    "#17becf",
}

# ════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0A3D62 0%, #1A6B3C 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.9rem; }
    .kpi-card {
        background: white; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #1b60a7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .kpi-label { font-size: 0.72rem; font-weight: 600; color: #7A8899;
                 text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1C2B3A; line-height: 1.1; }
    .kpi-sub   { font-size: 0.72rem; color: #7A8899; margin-top: 2px; }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #0A3D62;
        border-bottom: 2px solid #1A6B3C;
        padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem;
    }
    [data-testid="stSidebar"] { background: #F0F4F8; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600; font-size: 0.85rem;
    }
    .upload-zone {
        border: 2px dashed #1b60a7; border-radius: 10px;
        padding: 2rem; text-align: center; background: #f0f7ff;
    }
    .formula-badge {
        background: linear-gradient(135deg, #0A3D62 0%, #1A6B3C 100%);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        margin: 0.25rem 0.25rem 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# 1. CONSTANTES AGRONÓMICAS
# ════════════════════════════════════════════════════

# ── Rangos foliares PP — Palma Productiva (edad > EDAD_INMADURA_MAX) ──
# Faixa Ótima (Suficiência): macros en % (input g/kg ÷ 10), micros ppm → %
FOLIAR_RANGES = {
    "N":  {"min": 2.50,   "max": 2.80,   "unidad": "g/kg"},   # 2,50–2,80 %
    "P":  {"min": 0.16,   "max": 0.19,   "unidad": "g/kg"},   # 0,16–0,19 %
    "K":  {"min": 1.25,   "max": 1.45,   "unidad": "g/kg"},   # 1,25–1,45 %
    "Ca": {"min": 0.55,   "max": 0.75,   "unidad": "g/kg"},   # 0,55–0,75 %
    "Mg": {"min": 0.25,   "max": 0.40,   "unidad": "g/kg"},   # 0,25–0,40 %
    "S":  {"min": 0.20,   "max": 0.25,   "unidad": "g/kg"},   # 0,20–0,25 %
    "B":  {"min": 0.0020, "max": 0.0045, "unidad": "ppm"},    # 20–45 ppm
    "Cu": {"min": 0.0005, "max": 0.0008, "unidad": "ppm"},    # 5–8 ppm
    "Fe": {"min": 0.0080, "max": 0.0160, "unidad": "ppm"},    # 80–160 ppm
    "Mn": {"min": 0.0060, "max": 0.0200, "unidad": "ppm"},    # 60–200 ppm
    "Zn": {"min": 0.0015, "max": 0.0040, "unidad": "ppm"},    # 15–40 ppm
}

# ── Rangos foliares PNP — Palma No Productiva / inmadura (edad ≤ EDAD_INMADURA_MAX) ──
FOLIAR_RANGES_PNP = {
    "N":  {"min": 2.80,   "max": 3.20,   "unidad": "g/kg"},   # 2,80–3,20 %
    "P":  {"min": 0.17,   "max": 0.22,   "unidad": "g/kg"},   # 0,17–0,22 %
    "K":  {"min": 1.30,   "max": 1.60,   "unidad": "g/kg"},   # 1,30–1,60 %
    "Ca": {"min": 0.60,   "max": 0.80,   "unidad": "g/kg"},   # 0,60–0,80 %
    "Mg": {"min": 0.24,   "max": 0.40,   "unidad": "g/kg"},   # 0,24–0,40 %
    "S":  {"min": 0.22,   "max": 0.30,   "unidad": "g/kg"},   # 0,22–0,30 %
    "B":  {"min": 0.0020, "max": 0.0040, "unidad": "ppm"},    # 20–40 ppm
    "Cu": {"min": 0.0005, "max": 0.0008, "unidad": "ppm"},    # 5–8 ppm
    "Fe": {"min": 0.0080, "max": 0.0160, "unidad": "ppm"},    # 80–160 ppm
    "Mn": {"min": 0.0060, "max": 0.0200, "unidad": "ppm"},    # 60–200 ppm
    "Zn": {"min": 0.0015, "max": 0.0040, "unidad": "ppm"},    # 15–40 ppm
}

EXPORT_COEF = {
    "Cu": 0.004, "Fe": 0.03, "Mn": 0.025, "Zn": 0.01,
}

EFICIENCIA = {
    "Cu": 1.00, "Fe": 1.00, "Mn": 1.00, "Zn": 1.00,
}

DELTA = {
    "N": 0.1, "P": 0.1, "K": 0.50, "Ca": 0.01, "Mg": 0.01,
    "S": 0.01, "B": 0.0001, "Cu": 0.0001, "Fe": 0.0001, "Mn": 0.0001, "Zn": 0.0001,
}

G_PLANTA = {
    "N": 70, "P": 84, "K": 70, "Ca": 105, "Mg": 56,
    "S": 42, "B": 2.1, "Cu": 1.4, "Fe": 7.0, "Mn": 3.5, "Zn": 2.1,
}

N_PALMAS = 143

MACRO_ELEMS = {"N", "P", "K", "Ca", "Mg", "S"}
MICRO_ELEMS = {"B", "Cu", "Fe", "Mn", "Zn"}

ELEMENTOS_OFICIALES = {"N", "P", "K", "Ca", "Mg", "S", "B"}

ELEMENTOS_CALCULO = ["N", "P", "K", "Ca", "Mg", "S", "B", "Cu", "Fe", "Mn", "Zn"]

DB_COEF_GUINEENSIS = {
    "N":  (5.8305,   -0.1165),
    "P":  (4.3580,   -0.1892),
    "K":  (10.4364,  -0.1356),
    "Ca": (3.4953,   -0.3007),
    "Mg": (4.3580,   -0.1892),
    "S":  (1.043632, -0.013383),
    "B":  (0.013141, -0.001582),
}

DB_COEF_HIBRIDO = {
    "N":  (6.3014,   0.2818),
    "P":  (4.7183,  -0.0361),
    "K":  (11.3221, -0.3537),
    "Ca": (3.7766,  -0.1794),
    "Mg": (4.7183,  -0.0361),
    "S":  (1.130900, -0.003850),
    "B":  (0.014119, -0.000262),
}

PROD_COEF_GUINEENSIS = (-0.1283, 4.0247, -3.9629)
PROD_COEF_HIBRIDO = (-0.1450, 4.5496, -4.4798)
PROD_MAX_MAYOR_26 = 14.0
PROD_DEFAULT = 20.0

FUENTES_KALINI = {
    "N":  {"fuente": "Urea",         "aporte": 0.45},
    "P":  {"fuente": "SPT",          "aporte": 0.46},
    "K":  {"fuente": "KCl",          "aporte": 0.60},
    "Ca": {"fuente": "Cal_dolomita", "aporte": 0.35},
    "Mg": {"fuente": "Kieserita",    "aporte": 0.25, "aporte_secundario": {"S": 0.20}},
    "B":  {"fuente": "Granubor",     "aporte": 0.15},
    "S":  {"fuente": "Sulfato",      "aporte": 0.20},
}

FUENTES_EXCLUIDAS_FORMULA = {"cal_dolomita"}

ELEMENTOS_DESCOMPOSICION = ["N", "P", "K", "Ca", "Mg", "S", "B"]

DENSIDAD_TEORICA_HA = 143
# PNP (Palma No Productiva / inmadura): edad <= 3 según el modelo AGROPALMA
EDAD_INMADURA_MAX   = 3

N_PALMAS = DENSIDAD_TEORICA_HA
DENSIDAD_PALMAS_HA = DENSIDAD_TEORICA_HA

NIVELES_OPTIMOS_INMADURO = {}
NIVELES_INMADURO_CARGADOS = False

PRIORIDAD_IMPUTACION_FOLIAR = [
    ("edad_departamento", ["_imp_edad", "_imp_departamento"]),
    ("edad_material",     ["_imp_edad", "_imp_material"]),
    ("edad_finca",        ["_imp_edad", "_imp_finca"]),
    ("edad",              ["_imp_edad"]),
]

ETIQUETAS_FOLIAR = {
    "medido":            "Medido",
    "edad_departamento": "Edad + Departamento",
    "edad_material":     "Edad + Material (G/H)",
    "edad_finca":        "Edad + Finca",
    "edad":              "Edad",
    "edad_cercana":      "Edad más cercana",
    "sin_dato":          "Sin dato",
}

# ════════════════════════════════════════════════════
# 2. NORMALIZACIÓN Y CARGA — ROBUSTA
# ════════════════════════════════════════════════════


def normalize_col(col: str) -> str:
    import unicodedata
    col = unicodedata.normalize("NFKD", str(col)).encode("ascii", "ignore").decode()
    col = str(col).strip().lower()
    col = re.sub(r"[%/\-\+\.\(\)\[\]{}]", "_", col)
    col = re.sub(r"[^a-z0-9_]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col

def normalize_for_ui(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    area_col = _safe_find(df, ["area", "ha", "hectareas", "superficie"])
    if area_col and area_col in df.columns:
        area_s = df[area_col].apply(lambda x: to_float_safe(x, np.nan))
    else:
        area_s = pd.Series(np.nan, index=df.index)

    if "densidad_real" in df.columns:
        dens_s = pd.to_numeric(df["densidad_real"], errors="coerce").fillna(DENSIDAD_TEORICA_HA)
    else:
        dens_s = pd.Series(DENSIDAD_TEORICA_HA, index=df.index)

    for e in ["N", "P", "K", "Ca", "Mg", "B", "S"]:
        if e == "S":
            cand_ha = find_col(df.columns, [f"nec_{e.lower()}_kg_ha", f"da_{e.lower()}_kg_ha",
                                            f"nec_{e} (kg/ha)", f"da_{e} (kg/ha)"])
            cand_lote = find_col(df.columns, [f"nec_{e.lower()}_kg_lote", f"da_{e.lower()}_kg_lote",
                                              f"{e.lower()}_lote", f"da_{e}_lote"])
            cand_gpalma = find_col(df.columns, [f"nec_{e.lower()}_g_palma", f"da_{e.lower()}_g_palma",
                                                f"{e.lower()}_g_palma", f"da_{e}/palma (g)"])
        else:
            cand_ha = find_col(df.columns, [f"da_{e.lower()}_kg_ha", f"da_{e} (kg/ha)", f"da_{e.lower()} (kg/ha)"])
            cand_lote = find_col(df.columns, [f"da_{e.lower()}_kg_lote", f"da_{e}_lote (kg)", f"da_{e}_lote"])
            cand_gpalma = find_col(df.columns, [f"da_{e.lower()}_g_palma", f"da_{e}/palma (g)"])

        if cand_ha and cand_ha in df.columns:
            df[f"Rec_da_{e}_kgha"] = pd.to_numeric(df[cand_ha], errors="coerce")

        if cand_lote and cand_lote in df.columns:
            df[f"Rec_da_{e}_kglote"] = pd.to_numeric(df[cand_lote], errors="coerce")
        elif f"Rec_da_{e}_kgha" in df.columns and area_col in df.columns:
            df[f"Rec_da_{e}_kglote"] = kg_lote_desde_kgha(df[f"Rec_da_{e}_kgha"], area_s, dens_s)

        if cand_gpalma and cand_gpalma in df.columns:
            df[f"Rec_da_{e}_gpalma"] = pd.to_numeric(df[cand_gpalma], errors="coerce")
        elif f"Rec_da_{e}_kgha" in df.columns:
            df[f"Rec_da_{e}_gpalma"] = _to_g_palma_from_kgha(df[f"Rec_da_{e}_kgha"])
        elif f"Rec_da_{e}_kglote" in df.columns and area_col in df.columns:
            kgha_impl = (pd.to_numeric(df[f"Rec_da_{e}_kglote"], errors="coerce") / area_s) \
                        .replace([np.inf, -np.inf], np.nan)
            df[f"Rec_da_{e}_gpalma"] = _to_g_palma_from_kgha(kgha_impl)

        if f"Rec_da_{e}_kgha" in df.columns:
            df[f"Rec_da_{e}_kgpalma"] = kg_a_kg_palma(df[f"Rec_da_{e}_kgha"])
        elif f"Rec_da_{e}_gpalma" in df.columns:
            df[f"Rec_da_{e}_kgpalma"] = pd.to_numeric(df[f"Rec_da_{e}_gpalma"], errors="coerce") / 1000.0

    gpal_cols = [c for c in df.columns if c.startswith("Rec_da_") and c.endswith("_gpalma")]
    if gpal_cols:
        df["Rec_total_n_gpalma"] = df[gpal_cols].sum(axis=1, min_count=1)
        df["Rec_total_n_kgpalma"] = pd.to_numeric(df["Rec_total_n_gpalma"], errors="coerce") / 1000.0

    fuente_map_keys = {
        "urea":      ["fuente_urea_kg_ha", "urea_kg_ha", "urea (kg/ha)"],
        "spt":       ["fuente_spt_kg_ha", "spt_kg_ha", "spt (kg/ha)"],
        "kcl":       ["fuente_kcl_kg_ha", "kcl_kg_ha", "kcl (kg/ha)"],
        "kieserita": ["fuente_kieserita_kg_ha", "kieserita_kg_ha", "kieserita (kg/ha)"],
        "granubor":  ["fuente_granubor_kg_ha", "granubor_kg_ha", "granubor (kg/ha)"],
        "sulfato":   ["fuente_sulfato_kg_ha", "sulfato_kg_ha", "sulfato (kg/ha)"],
    }
    for key, candidates in fuente_map_keys.items():
        found = None
        for cand in candidates:
            c = find_col(df.columns, [cand])
            if c:
                found = c
                break
        if not found:
            df[f"Rec_{key}_kgha"] = np.nan
            df[f"Rec_{key}_kglote"] = np.nan
            df[f"Rec_{key}_gpalma"] = np.nan
            df[f"Rec_{key}_kgpalma"] = np.nan
            continue

        df[f"Rec_{key}_kgha"] = pd.to_numeric(df[found], errors="coerce")
        df[f"Rec_{key}_gpalma"] = _to_g_palma_from_kgha(df[f"Rec_{key}_kgha"])
        df[f"Rec_{key}_kgpalma"] = kg_a_kg_palma(df[f"Rec_{key}_kgha"])

        cand_lote_motor = find_col(df.columns, [f"fuente_{key}_kg_lote"])
        if cand_lote_motor:
            df[f"Rec_{key}_kglote"] = pd.to_numeric(df[cand_lote_motor], errors="coerce")
        elif area_col in df.columns:
            df[f"Rec_{key}_kglote"] = kg_lote_desde_kgha(df[f"Rec_{key}_kgha"], area_s, dens_s)
        else:
            cand_lote = find_col(df.columns,
                                 [found.replace(" (kg/ha)", "_lote (kg)"),
                                  found.replace("_kg_ha", "_kg_lote")])
            df[f"Rec_{key}_kglote"] = (
                pd.to_numeric(df[cand_lote], errors="coerce") if cand_lote else np.nan
            )

    for old, new in [("Rec_spt_kgha", "Rec_SPT_kgha"),
                     ("Rec_spt_kglote", "Rec_SPT_kglote"),
                     ("Rec_spt_gpalma", "Rec_SPT_gpalma"),
                     ("Rec_spt_kgpalma", "Rec_SPT_kgpalma")]:
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    caldol = _find_caldol_columns(df.columns)
    if caldol["kgha"]:
        df["Rec_caldol_kgha"] = pd.to_numeric(df[caldol["kgha"]], errors="coerce")
    if caldol["kglote"]:
        df["Rec_caldol_kglote"] = pd.to_numeric(df[caldol["kglote"]], errors="coerce")
    elif "Rec_caldol_kgha" in df.columns and area_col in df.columns:
        df["Rec_caldol_kglote"] = kg_lote_desde_kgha(df["Rec_caldol_kgha"], area_s, dens_s)
    if caldol["gpalma"]:
        df["Rec_caldol_gpalma"] = pd.to_numeric(df[caldol["gpalma"]], errors="coerce")
    elif "Rec_caldol_kgha" in df.columns:
        df["Rec_caldol_gpalma"] = _to_g_palma_from_kgha(df["Rec_caldol_kgha"])
    elif "Rec_caldol_kglote" in df.columns and area_col in df.columns:
        kgha_impl = (pd.to_numeric(df["Rec_caldol_kglote"], errors="coerce") / area_s) \
                    .replace([np.inf, -np.inf], np.nan)
        df["Rec_caldol_gpalma"] = _to_g_palma_from_kgha(kgha_impl)
    if "Rec_caldol_kgha" in df.columns:
        df["Rec_caldol_kgpalma"] = kg_a_kg_palma(df["Rec_caldol_kgha"])
    elif "Rec_caldol_gpalma" in df.columns:
        df["Rec_caldol_kgpalma"] = pd.to_numeric(df["Rec_caldol_gpalma"], errors="coerce") / 1000.0
    if "Rec_caldol_kglote" in df.columns:
        df["Rec_caldol_tonlote"] = (
            pd.to_numeric(df["Rec_caldol_kglote"], errors="coerce") / 1000.0
        ).replace([np.inf, -np.inf], np.nan)
    df["flag_caldol_detectada"] = any(v is not None for v in caldol.values())

    fuentes_g_existentes = [c for c in
                            ["Rec_urea_gpalma", "Rec_SPT_gpalma", "Rec_kcl_gpalma",
                             "Rec_kieserita_gpalma", "Rec_granubor_gpalma", "Rec_sulfato_gpalma"]
                            if c in df.columns]
    if fuentes_g_existentes:
        df["Rec_total_f_gpalma"] = (
            df[fuentes_g_existentes].apply(pd.to_numeric, errors="coerce")
            .sum(axis=1, min_count=1)
        )
        df["Rec_total_f_kgpalma"] = pd.to_numeric(df["Rec_total_f_gpalma"], errors="coerce") / 1000.0

    if "Formula_Kalini" in df.columns and "Formula (N-P-K-MgO-B)" not in df.columns:
        df["Formula (N-P-K-MgO-B)"] = df["Formula_Kalini"]

    df = _round_cols(df, decimals_default=2, decimals_gpalma=4)

    return df


def to_float_safe(valor, default=np.nan):
    if valor is None:
        return default
    try:
        if pd.isna(valor):
            return default
    except Exception:
        pass
    if isinstance(valor, str):
        v = valor.strip()
        if v == "" or v.lower() in {"nan", "none", "null", "-", "—"}:
            return default
        if v.startswith("="):
            return default
        v = v.replace("\xa0", "").replace(" ", "")
        if "," in v and "." in v:
            if v.rfind(",") > v.rfind("."):
                v = v.replace(".", "").replace(",", ".")
            else:
                v = v.replace(",", "")
        elif "," in v:
            v = v.replace(",", ".")
        try:
            return float(v)
        except Exception:
            return default
    try:
        return float(valor)
    except Exception:
        return default


FRANJAS_EDAD = [
    "0–3 (inmaduro)", "4–8", "9–13", "14–18",
    "19–23", "24–28", "29+", "Sin edad",
]


def franja_edad(valor, sin_edad="Sin edad"):
    edad = to_float_safe(valor, np.nan)
    if pd.isna(edad):
        return sin_edad
    if edad <= 3:
        return "0–3 (inmaduro)"
    if edad <= 8:
        return "4–8"
    if edad <= 13:
        return "9–13"
    if edad <= 18:
        return "14–18"
    if edad <= 23:
        return "19–23"
    if edad <= 28:
        return "24–28"
    return "29+"


def find_col(df_cols, candidates):
    norm_map = {normalize_col(c): c for c in df_cols}
    for cand in candidates:
        nc = normalize_col(cand)
        if nc in norm_map:
            return norm_map[nc]
    return None

def calcular_densidad_real(df, area_col, n_palmas_col):
    if area_col and n_palmas_col and area_col in df.columns and n_palmas_col in df.columns:
        area = df[area_col].apply(lambda x: to_float_safe(x, np.nan))
        npal = df[n_palmas_col].apply(lambda x: to_float_safe(x, np.nan))
        with np.errstate(divide="ignore", invalid="ignore"):
            dens = (npal / area).replace([np.inf, -np.inf], np.nan)
        dens = dens.where((dens >= 50) & (dens <= 300))
    else:
        dens = pd.Series(np.nan, index=df.index)
    return dens.fillna(DENSIDAD_TEORICA_HA).round(2)


def kg_a_g_palma(kgha_s):
    s = pd.to_numeric(kgha_s, errors="coerce")
    return (s * 1000.0 / DENSIDAD_TEORICA_HA).replace([np.inf, -np.inf], np.nan)


def kg_a_kg_palma(kgha_s):
    s = pd.to_numeric(kgha_s, errors="coerce")
    return (s / DENSIDAD_TEORICA_HA).replace([np.inf, -np.inf], np.nan)


def kg_lote_desde_kgha(kgha_s, area_s, densidad_s):
    kgha = pd.to_numeric(kgha_s, errors="coerce")
    area = pd.to_numeric(area_s, errors="coerce")
    dens = pd.to_numeric(densidad_s, errors="coerce").fillna(DENSIDAD_TEORICA_HA)
    return (kgha * area * (dens / DENSIDAD_TEORICA_HA)).replace([np.inf, -np.inf], np.nan)


def _factor_densidad(df, index):
    if "densidad_real" in df.columns:
        d = pd.to_numeric(df["densidad_real"], errors="coerce").fillna(DENSIDAD_TEORICA_HA)
    else:
        d = pd.Series(DENSIDAD_TEORICA_HA, index=df.index)
    return (d / DENSIDAD_TEORICA_HA).reindex(index)


def _to_g_palma_from_kg_lote(kglote_s, n_palmas_s):
    kgl = pd.to_numeric(kglote_s, errors="coerce")
    npal = pd.to_numeric(n_palmas_s, errors="coerce")
    return (kgl / npal.replace(0, np.nan) * 1000.0).replace([np.inf, -np.inf], np.nan)

def cargar_dataset(file_bytes: bytes, file_name: str = "") -> pd.DataFrame:
    is_csv = isinstance(file_name, str) and file_name.lower().endswith(".csv")

    if is_csv:
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes), sep=None, engine="python",
                dtype=object, encoding="utf-8-sig",
            )
        except Exception:
            df = pd.read_csv(
                io.BytesIO(file_bytes), sep=",",
                dtype=object, encoding="latin1",
            )
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl", dtype=object)

    df = df.dropna(how="all").copy()
    df.columns = [normalize_col(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()

    rename_map = {}

    alias_generales = {
        "lote":        ["lote", "block", "bloque", "codigo_lote", "cod_lote", "id_lote", "lote_id"],
        "finca":       ["finca", "fazenda", "farm", "hacienda"],
        "departamento":["departamento", "depto", "zona", "region"],
        "manejo":      ["manejo", "Manejo"],
        "siembra":     ["siembra", "ano_siembra", "anio_siembra", "planting_year"],
        "edad":        ["edad", "age", "anos", "ano_planta", "idade", "idade_ano"],
        "variedad":    ["variedad", "variety", "cultivar", "tipo_variedad"],
        "material":    ["material", "material_genetico", "genetica", "origen_material"],
        "n_palmas":    ["n_palmas", "numero_palmas", "palmas", "plantas", "plantas_ha"],
        "area":        ["area", "ha", "hectareas", "superficie"],
        "densidad":    ["densidad", "density", "palmas_ha"],
        "ton_ha":      ["ton_ha", "ton/ha", "ton ha", "t_ha", "rff", "cff",
                        "produtividade", "productividad", "produccion_ha",
                        "estimativa_cff_t_ha"],
        "0g_1h":       ["0g_1h", "0G_1H", "og_1h", "g_h", "tipo_material",
                        "flag_0g_1h", "indicador_0g_1h"],
    }

    for target, aliases in alias_generales.items():
        col = find_col(df.columns, aliases)
        if col and normalize_col(col) != normalize_col(target):
            rename_map[col] = target

    foliar_aliases = {
        "N":  ["n_f", "n_fol", "fol_n", "foliar_n"],
        "P":  ["p_f", "p_fol", "fol_p", "foliar_p"],
        "K":  ["k_f", "k_fol", "fol_k", "foliar_k"],
        "Ca": ["ca_f", "ca_fol", "fol_ca", "foliar_ca"],
        "Mg": ["mg_f", "mg_fol", "fol_mg", "foliar_mg"],
        "S":  ["s_f", "s_fol", "fol_s", "foliar_s"],
        "B":  ["b_f", "b_fol", "fol_b", "foliar_b"],
        "Cu": ["cu_f", "cu_fol", "fol_cu", "foliar_cu"],
        "Fe": ["fe_f", "fe_fol", "fol_fe", "foliar_fe"],
        "Mn": ["mn_f", "mn_fol", "fol_mn", "foliar_mn"],
        "Zn": ["zn_f", "zn_fol", "fol_zn", "foliar_zn"],
    }

    for elem, aliases in foliar_aliases.items():
        target = f"fol_{elem.lower()}"
        if target in df.columns:
            continue
        col = find_col(df.columns, aliases)
        if col:
            rename_map[col] = target

    df = df.rename(columns=rename_map)

    text_cols = {"finca", "lote", "departamento", "variedad", "material", "manejo"}

    for col in df.columns:
        if col in text_cols:
            mask = df[col].notna()
            df[col] = df[col].astype(object)
            df.loc[mask, col] = (
                df.loc[mask, col]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({"NAN": np.nan, "NONE": np.nan, "NULL": np.nan, "": np.nan})
            )
        else:
            df[col] = df[col].apply(lambda x: to_float_safe(x, default=np.nan))

    return df

# ════════════════════════════════════════════════════
# 3. MOTOR DE CÁLCULO
# ════════════════════════════════════════════════════


def determinar_especie(variedad="", material="", indicador_0g_1h=np.nan):
    ind = to_float_safe(indicador_0g_1h, default=np.nan)
    if not pd.isna(ind):
        return "Guineensis" if int(ind) == 0 else "Hibrido_OxG"

    texto = f"{variedad} {material}".strip().lower()

    if any(x in texto for x in ["hibrido", "híbrido", "hybrid", "oxg", "clone", "clon"]):
        return "Hibrido_OxG"

    if any(x in texto for x in ["tenera", "dura", "pisifera", "guineensis", "deli", "nigeria", "ghana"]):
        return "Guineensis"

    return "No_identificado"


def get_flag_0g_1h(row, flag_col=None, variedad_col=None, material_col=None):

    if flag_col and flag_col in row.index:
        val = to_float_safe(row.get(flag_col, np.nan), default=np.nan)
        if not pd.isna(val):
            return 0 if int(val) == 0 else 1
    variedad = row.get(variedad_col, "") if variedad_col else ""
    material = row.get(material_col, "") if material_col else ""
    especie = determinar_especie(variedad, material)
    return 1 if especie == "Hibrido_OxG" else 0


def foliar_a_pct(valor, elem):

    if pd.isna(valor):
        return np.nan
    valor = to_float_safe(valor, default=np.nan)
    if pd.isna(valor):
        return np.nan
    if elem in MICRO_ELEMS and elem not in ELEMENTOS_OFICIALES:
        return valor / 10000.0
    return valor / 10.0

def get_foliar_ranges(edad=np.nan):
    """Rangos foliares según etapa del cultivo:
    PNP (Palma No Productiva, edad ≤ EDAD_INMADURA_MAX) o PP (Palma Productiva).
    Si el Excel trae una hoja 'niveles_optimos', esa tabla personalizada
    tiene prioridad para los lotes inmaduros."""
    e = to_float_safe(edad, default=np.nan)
    if (not pd.isna(e)) and e <= EDAD_INMADURA_MAX:
        return NIVELES_OPTIMOS_INMADURO if NIVELES_OPTIMOS_INMADURO else FOLIAR_RANGES_PNP
    return FOLIAR_RANGES


def cargar_niveles_optimos(file_bytes: bytes) -> dict:

    global NIVELES_OPTIMOS_INMADURO, NIVELES_INMADURO_CARGADOS
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
        hoja = next((h for h in xl.sheet_names if normalize_col(h) == "niveles_optimos"), None)
        if hoja is None:
            return {}
        raw = xl.parse(hoja)
        raw.columns = [normalize_col(c) for c in raw.columns]
        col_elem = find_col(raw.columns, ["elemento", "nutriente", "elem"])
        col_min  = find_col(raw.columns, ["minimo", "min", "minimo_foliar", "limite_inferior"])
        col_max  = find_col(raw.columns, ["maximo", "max", "maximo_foliar", "limite_superior"])
        col_und  = find_col(raw.columns, ["unidad", "unidades"])
        if not (col_elem and col_min and col_max):
            return {}
        out = {}
        for _, r in raw.iterrows():
            e = re.sub(r"[^A-Z]", "", str(r.get(col_elem, "")).strip().upper())
            if e in FOLIAR_RANGES:
                out[e] = {
                    "min": to_float_safe(r.get(col_min, np.nan)),
                    "max": to_float_safe(r.get(col_max, np.nan)),
                    "unidad": str(r.get(col_und, FOLIAR_RANGES[e]["unidad"])),
                }
        if out:
            NIVELES_OPTIMOS_INMADURO = out
            NIVELES_INMADURO_CARGADOS = True
        return out
    except Exception:
        return {}


def get_foliar_status(val_pct, elem, edad=np.nan):
    r = get_foliar_ranges(edad).get(elem)
    if not r or pd.isna(val_pct) or pd.isna(r["min"]) or pd.isna(r["max"]):
        return "sin_dato", "—"
    mn, mx = r["min"], r["max"]
    if val_pct < mn * 0.80:
        return "critico", "🔴 Muy Bajo"
    if val_pct < mn:
        return "bajo", "🟡 Bajo"
    if val_pct <= mx:
        return "optimo", "🟢 Óptimo"
    return "alto", "🔵 Alto"


def calc_fator_reajuste(val_pct, elem, edad=np.nan):
    r = get_foliar_ranges(edad).get(elem)
    if not r or pd.isna(val_pct) or pd.isna(r["min"]) or pd.isna(r["max"]):
        return 1.0
    mn, mx = r["min"], r["max"]
    if val_pct < mn * 0.80:
        return 1.4
    if val_pct < mn:
        return 1.2
    if val_pct > mx:
        return 0.5
    return 1.0


def calc_demanda_bruta(elem, rff, flag_0g_1h, edad=np.nan):
    """Demanda bruta (kg/ha) con límite inferior de cero.

    Para reproducir el archivo de referencia AGROPALMA, la ecuación de
    demanda se aplica con el RFF seleccionado para todas las edades,
    incluyendo lotes inmaduros cuando exista `ton/ha` real.
    """
    rff = to_float_safe(rff, default=np.nan)
    if pd.isna(rff):
        return 0.0
    if elem in ELEMENTOS_OFICIALES:
        coefs = DB_COEF_GUINEENSIS if int(flag_0g_1h) == 0 else DB_COEF_HIBRIDO
        a, b = coefs[elem]
        return max(0.0, a * rff + b)
    return max(0.0, EXPORT_COEF.get(elem, 0.0) * rff)


def calc_demanda_ajustada(elem, val_pct, demanda_bruta, edad=np.nan):
    """Corrección foliar usando el rango óptimo correspondiente a la EDAD del lote."""
    r = get_foliar_ranges(edad).get(elem)
    if (not r) or pd.isna(val_pct) or pd.isna(r.get("min")) or pd.isna(r.get("max")):
        return max(0.0, demanda_bruta)
    mn, mx = r["min"], r["max"]
    delta = DELTA.get(elem, np.nan)
    g_planta = G_PLANTA.get(elem, np.nan)
    if pd.isna(delta) or pd.isna(g_planta) or delta <= 0:
        return max(0.0, demanda_bruta)
    if val_pct < mn:
        correccion = ((mn - val_pct) / delta) * g_planta * DENSIDAD_TEORICA_HA / 1000
        return max(0.0, demanda_bruta + correccion)
    if val_pct > mx:
        return max(0.0, demanda_bruta * 0.5)
    return max(0.0, demanda_bruta)

def imputar_foliares(df: pd.DataFrame, elementos=None) -> pd.DataFrame:
    """
    Determina análisis foliares faltantes por prioridad (promedio por grupo,
    calculado SOLO con datos medidos → no propaga imputaciones):

        1. misma edad + mismo departamento
        2. misma edad + mismo material (G = Guineensis / H = Híbrido OxG)
        3. misma edad + misma finca
        4. misma edad
        5. edad más cercana con datos medidos

    Salidas por elemento:
        fol_pct_{e}       valor foliar en % (medido o imputado)
        fol_fuente_{e}    nivel de imputación aplicado
    Salida consolidada:
        Foliar            cómo se determinó el dato foliar del lote
                          (el nivel MENOS específico usado entre sus elementos)
    """
    df = df.copy()
    if elementos is None:
        elementos = ELEMENTOS_CALCULO

    edad_col  = find_col(df.columns, ["edad", "age", "anos", "ano_planta", "idade"])
    dep_col   = find_col(df.columns, ["departamento", "depto", "zona", "region"])
    finca_col = find_col(df.columns, ["finca", "farm", "fazenda", "hacienda"])

    def _txt(col):
        if col and col in df.columns:
            return (df[col].astype(str).str.strip().str.upper()
                    .replace({"NAN": np.nan, "NONE": np.nan, "NULL": np.nan, "": np.nan}))
        return pd.Series(np.nan, index=df.index, dtype=object)

    edad_s = (df[edad_col].apply(lambda x: to_float_safe(x, np.nan))
              if edad_col else pd.Series(np.nan, index=df.index))
    especie_s = (df["especie"].astype(str) if "especie" in df.columns
                 else pd.Series("No_identificado", index=df.index))

    df["_imp_edad"]         = edad_s.round(0)
    df["_imp_departamento"] = _txt(dep_col)
    df["_imp_finca"]        = _txt(finca_col)
    df["_imp_material"]     = especie_s.map({"Guineensis": "G", "Hibrido_OxG": "H"}).fillna("ND")

    for elem in elementos:
        fol_col = find_col(df.columns, [f"fol_{elem.lower()}", f"{elem}_f", f"{elem.lower()}_f",
                                        f"{elem}_fol", f"{elem.lower()}_fol", f"foliar_{elem.lower()}"])
        pct_col = f"fol_pct_{elem.lower()}"
        fte_col = f"fol_fuente_{elem.lower()}"

        medido = df[fol_col].apply(lambda x: foliar_a_pct(x, elem)) if fol_col \
            else pd.Series(np.nan, index=df.index)

        imputada = medido.copy()
        fuente = pd.Series(np.where(imputada.notna(), "medido", "sin_dato"), index=df.index)

        def _medias_grupo(cols):
            clave = df[cols].astype(str).apply(lambda r: "|".join(r.values), axis=1)
            clave = clave.where(df[cols].notna().all(axis=1))
            tmp = pd.DataFrame({"k": clave, "v": medido}).dropna(subset=["k"])
            medias = tmp.dropna(subset=["v"]).groupby("k")["v"].mean()
            return clave.map(medias)

        for nivel, cols in PRIORIDAD_IMPUTACION_FOLIAR:
            pend = imputada.isna()
            if not pend.any():
                break
            sug = _medias_grupo(cols)
            usar = pend & sug.notna()
            if usar.any():
                imputada.loc[usar] = sug.loc[usar]
                fuente.loc[usar] = nivel

        pend = imputada.isna()
        if pend.any():
            tmp = pd.DataFrame({"edad": df["_imp_edad"], "v": medido}).dropna()
            if len(tmp):
                m_edad = tmp.groupby("edad")["v"].mean()
                if len(m_edad):
                    edades = m_edad.index.to_numpy(dtype=float)

                    def _cercana(e):
                        if pd.isna(e):
                            return np.nan
                        j = int(np.abs(edades - float(e)).argmin())
                        return float(m_edad.iloc[j])

                    sug = df["_imp_edad"].apply(_cercana)
                    usar = pend & sug.notna()
                    if usar.any():
                        imputada.loc[usar] = sug.loc[usar]
                        fuente.loc[usar] = "edad_cercana"

        df[pct_col] = imputada.round(8)
        df[fte_col] = fuente

    jerarquia = ["medido"] + [n for n, _ in PRIORIDAD_IMPUTACION_FOLIAR] + ["edad_cercana", "sin_dato"]
    rango = {n: i for i, n in enumerate(jerarquia)}
    fte_cols = [f"fol_fuente_{e.lower()}" for e in elementos if f"fol_fuente_{e.lower()}" in df.columns]

    def _foliar_row(vals):
        niveles = [v for v in vals if v in rango]
        if not niveles:
            return ETIQUETAS_FOLIAR["sin_dato"]
        peor = max(niveles, key=lambda v: rango[v])
        return ETIQUETAS_FOLIAR.get(peor, peor)

    if fte_cols:
        df["Foliar"] = df[fte_cols].astype(str).apply(_foliar_row, axis=1)

    df = df.drop(columns=[c for c in ["_imp_edad", "_imp_departamento", "_imp_finca", "_imp_material"]
                          if c in df.columns])
    return df


def calc_recomendacion_final(demanda_ajustada, elem):
    if elem in ELEMENTOS_OFICIALES:
        return max(0.0, demanda_ajustada)
    ef = to_float_safe(EFICIENCIA.get(elem, 1.0), default=1.0)
    if pd.isna(ef) or ef <= 0:
        return max(0.0, demanda_ajustada)
    return max(0.0, demanda_ajustada / ef)


def calcular_produccion_modelada(edad, especie="No_identificado"):
    edad = to_float_safe(edad, default=np.nan)
    if pd.isna(edad):
        return np.nan

    especie_txt = str(especie).strip().lower()
    if especie_txt == "guineensis":
        a, b, c = PROD_COEF_GUINEENSIS
    elif especie_txt in {"hibrido_oxg", "hibrido", "híbrido", "clone", "clon"}:
        a, b, c = PROD_COEF_HIBRIDO
    else:
        return np.nan

    if edad > 26:
        return round(PROD_MAX_MAYOR_26, 3)

    prod = a * (edad ** 2) + b * edad + c
    prod = max(0.0, prod)

    return round(prod, 3)


def obtener_rff_calculo(row, rff_col=None, edad_col=None, especie="No_identificado"):
    edad = np.nan
    if edad_col and edad_col in row.index:
        edad = to_float_safe(row.get(edad_col, np.nan), default=np.nan)

    # ── Coincidencia con el modelo AGROPALMA: el dato REAL de ton/ha manda; ──
    # la curva por edad solo aplica cuando no hay dato real.
    if rff_col and rff_col in row.index:
        rff_real = to_float_safe(row.get(rff_col, np.nan), default=np.nan)
        if not pd.isna(rff_real) and rff_real > 0:
            if not pd.isna(edad) and edad > 26:
                return round(min(rff_real, PROD_MAX_MAYOR_26), 3), "dato_real_mayor_26"
            return round(rff_real, 3), "dato_real"

    if not pd.isna(edad):
        rff_modelado = calcular_produccion_modelada(edad, especie=especie)
        if not pd.isna(rff_modelado) and rff_modelado > 0:
            if not pd.isna(edad) and edad > 26:
                return rff_modelado, "techo_mayor_26"
            return rff_modelado, "curva_variedad_edad"

    return PROD_DEFAULT, "respaldo"


def descomponer_fuentes_kalini(df, area_serie, densidad_serie):
    """
    Descompone cada nutriente en su fuente comercial usando la
    RECOMENDACIÓN FINAL (nec_*). Para N, P, K, Ca, Mg, S y B:
    nec_* = da_* (eficiencia 1.0, ELEMENTOS_OFICIALES). Si nec_* no
    existe para un elemento, cae a da_* como respaldo (compatibilidad).

    Reglas de descomposición (tabla de fuentes oficial):
        Kieserita (Mg, 25 %) aporta S secundario (20 % de su dosis);
        el Sulfato cubre el saldo de S con su aporte del 20 %:
        Sulfato = max(0, nec_S − Kieserita × 0.20) / 0.20
    """
    df = df.copy()
    ha_cols, lote_cols, gpalma_cols = [], [], []
    aporte_secundario_acum = {}
    orden = ["N", "P", "K", "Ca", "Mg", "B", "S"]

    for elem in orden:
        meta = FUENTES_KALINI.get(elem)
        if not meta:
            continue
        fuente_nombre = meta["fuente"]
        apor = to_float_safe(meta["aporte"], default=np.nan)

        # ── FIX QA: base = recomendación final (post-eficiencia), no la demanda ajustada ──
        rec_ha_col = f"nec_{elem.lower()}_kg_ha"
        da_ha_col = f"da_{elem.lower()}_kg_ha"
        base_col = rec_ha_col if rec_ha_col in df.columns else da_ha_col
        if base_col not in df.columns or pd.isna(apor) or apor <= 0:
            continue

        nombre_key = fuente_nombre.lower()

        if elem in aporte_secundario_acum:
            aporte_secundario = aporte_secundario_acum[elem]
            base_efectivo = (df[base_col] - aporte_secundario).clip(lower=0)
            # ── Trazabilidad QA del sulfato ──
            if elem == "S":
                df["S_necesidad_final_kg_ha"] = df[base_col].round(6)
                df["S_aportado_kieserita_kg_ha"] = aporte_secundario.round(6)
                df["S_saldo_para_sulfato_kg_ha"] = base_efectivo.round(6)
        else:
            base_efectivo = df[base_col]
            if elem == "S":
                df["S_necesidad_final_kg_ha"] = df[base_col].round(6)
                df["S_aportado_kieserita_kg_ha"] = 0.0
                df["S_saldo_para_sulfato_kg_ha"] = base_efectivo.round(6)

        ha_c = f"fuente_{nombre_key}_kg_ha"
        lo_c = f"fuente_{nombre_key}_kg_lote"
        gp_c = f"fuente_{nombre_key}_g_palma"

        df[ha_c] = (base_efectivo / apor).round(6)

        # ── Trazabilidad QA: dosis final de sulfato ──
        if elem == "S":
            df["Sulfato_calculado_kg_ha"] = df[ha_c]

        df[lo_c] = kg_lote_desde_kgha(df[ha_c], area_serie, densidad_serie).round(6)
        df[gp_c] = kg_a_g_palma(df[ha_c]).round(6)

        ha_cols.append(ha_c)
        lote_cols.append(lo_c)
        gpalma_cols.append(gp_c)

        for sec_elem, sec_apor in meta.get("aporte_secundario", {}).items():
            sec_apor = to_float_safe(sec_apor, default=np.nan)
            if pd.isna(sec_apor) or sec_apor <= 0:
                continue
            aporte_kg_ha = df[ha_c] * sec_apor
            aporte_secundario_acum[sec_elem] = aporte_secundario_acum.get(sec_elem, 0.0) + aporte_kg_ha

    if gpalma_cols:
        df["total_fuentes_g_palma"] = df[gpalma_cols].sum(axis=1, min_count=1).round(6)
        df["total_fuentes_kg_palma"] = (df["total_fuentes_g_palma"] / 1000.0).round(6)
    else:
        df["total_fuentes_g_palma"] = np.nan
        df["total_fuentes_kg_palma"] = np.nan

    excluidas = {f["fuente"].lower() for f in FUENTES_KALINI.values()
                 if f["fuente"].lower() in FUENTES_EXCLUIDAS_FORMULA}
    compuesto_ha_cols = [c for c in ha_cols
                         if c.replace("fuente_", "").replace("_kg_ha", "") not in excluidas]

    if compuesto_ha_cols:
        df["total_compuesto_kg_ha"] = df[compuesto_ha_cols].sum(axis=1, min_count=1).round(6)
        compuesto_lote_cols = [c.replace("_kg_ha", "_kg_lote") for c in compuesto_ha_cols]
        df["total_compuesto_kg_lote"] = df[compuesto_lote_cols].sum(axis=1, min_count=1).round(6)
    else:
        df["total_compuesto_kg_ha"] = np.nan
        df["total_compuesto_kg_lote"] = np.nan

    def _valor_nutriente(r, e):
        """Recomendación final (nec_*) si existe; si no, demanda ajustada (da_*)."""
        c_nec = f"nec_{e.lower()}_kg_ha"
        c_da = f"da_{e.lower()}_kg_ha"
        if c_nec in r.index and not pd.isna(r.get(c_nec, np.nan)):
            return r[c_nec]
        return r.get(c_da, np.nan)

    def formula_row(r):
        tot_c = r.get("total_compuesto_kg_ha", np.nan)
        if pd.isna(tot_c) or tot_c <= 0:
            return np.nan

        def pct(e, dec=0):
            v = _valor_nutriente(r, e)
            if pd.isna(v):
                return 0
            return round(v / tot_c * 100, dec)

        return (f"{pct('N')}-{pct('P')}-{pct('K')}-{pct('Mg')} MgO-"
                f"{pct('S')} S-{pct('B', 2)} B")

    df["Formula_Kalini"] = df.apply(formula_row, axis=1)
    return df


def calculadora_fert(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    edad_col = find_col(df.columns, ["edad", "age", "anos", "ano_planta", "idade", "idade_ano"])
    rff_col = find_col(df.columns, ["ton_ha", "ton/ha", "ton ha", "t_ha", "rff", "cff",
                                    "produtividade", "productividad", "produccion_ha",
                                    "estimativa_cff_t_ha"])
    variedad_col = find_col(df.columns, ["variedad", "variety", "cultivar", "tipo_variedad"])
    material_col = find_col(df.columns, ["material", "material_genetico", "genetica", "origen_material"])
    flag_col = find_col(df.columns, ["0g_1h", "og_1h", "g_h", "flag_0g_1h",
                                     "tipo_material", "indicador_0g_1h"])
    area_col = find_col(df.columns, ["area", "ha", "hectareas", "superficie"])
    n_palmas_col = find_col(df.columns, ["n_palmas", "numero_palmas", "palmas", "plantas", "plantas_ha"])

    if n_palmas_col is None:
        dens_col = find_col(df.columns, ["densidad", "palmas_ha", "density", "plantas_ha"])
        area_col_tmp = find_col(df.columns, ["area", "ha", "hectareas", "superficie"])
        if dens_col and area_col_tmp:
            df["n_palmas_inferido"] = (
                df[area_col_tmp].apply(lambda x: to_float_safe(x, np.nan)) *
                df[dens_col].apply(lambda x: to_float_safe(x, np.nan))
            ).round(0)
            n_palmas_col = "n_palmas_inferido"

    rff_res, fuente_res, especie_res, flag_res, prod_obs_res, prod_mod_res = [], [], [], [], [], []

    for _, row in df.iterrows():
        variedad = row.get(variedad_col, "") if variedad_col is not None else ""
        material = row.get(material_col, "") if material_col is not None else ""
        flag = get_flag_0g_1h(row, flag_col=flag_col,
                              variedad_col=variedad_col, material_col=material_col)
        especie = determinar_especie(variedad=variedad, material=material, indicador_0g_1h=flag)
        edad_val = row.get(edad_col, np.nan) if edad_col is not None else np.nan
        prod_modelada = calcular_produccion_modelada(edad_val, especie=especie)
        prod_observada = row.get(rff_col, np.nan) if rff_col is not None else np.nan
        prod_observada = to_float_safe(prod_observada, default=np.nan)
        rff, fuente = obtener_rff_calculo(row, rff_col=rff_col, edad_col=edad_col, especie=especie)

        rff_res.append(round(rff, 3))
        fuente_res.append(fuente)
        especie_res.append(especie)
        flag_res.append(flag)
        prod_obs_res.append(round(prod_observada, 3) if not pd.isna(prod_observada) else np.nan)
        prod_mod_res.append(round(prod_modelada, 3) if not pd.isna(prod_modelada) else np.nan)

    df["ton_ha_observada"] = prod_obs_res
    df["ton_ha_modelada"] = prod_mod_res
    df["rff_calculo"] = rff_res
    df["fuente_rff"] = fuente_res
    df["especie"] = especie_res
    df["flag_0g_1h"] = flag_res

    if edad_col is not None:
        edad_serie = df[edad_col].apply(lambda x: to_float_safe(x, np.nan))
    else:
        edad_serie = pd.Series(np.nan, index=df.index)

    # ── QA: lotes sin edad se conservan, solo se marcan (no se eliminan) ──
    df["flag_edad_faltante"] = edad_serie.isna().astype(int)

    # La edad se conserva únicamente como atributo de trazabilidad.
    # No altera el RFF ni la demanda nutricional: el archivo AGROPALMA usa
    # el `ton/ha` real también para edades <= 3 cuando está disponible.
    df["flag_inmaduro"] = edad_serie.notna() & (edad_serie <= EDAD_INMADURA_MAX)

    df = imputar_foliares(df)

    if area_col is not None and area_col in df.columns:
        area_serie = df[area_col].apply(lambda x: to_float_safe(x, default=np.nan))
    else:
        area_serie = pd.Series(np.nan, index=df.index, dtype="float64")
    area_serie = area_serie.replace([np.inf, -np.inf, 0], np.nan)

    densidad_serie = calcular_densidad_real(df, area_col, n_palmas_col)
    df["densidad_real"] = densidad_serie
    df["n_palmas_lote"] = (area_serie * densidad_serie).round(0)

    # ── Motor por elemento (usa fol_pct_ ya imputado y rangos según edad) ──
    for elem in ELEMENTOS_CALCULO:
        fol_res, db_res, da_res, rec_res, fator_res, status_res = [], [], [], [], [], []

        for _, row in df.iterrows():
            rff = to_float_safe(row.get("rff_calculo", np.nan), default=np.nan)
            flag = to_float_safe(row.get("flag_0g_1h", 0), default=0)
            edad = to_float_safe(row.get(edad_col, np.nan), default=np.nan) if edad_col is not None else np.nan

            val_pct = to_float_safe(row.get(f"fol_pct_{elem.lower()}", np.nan), default=np.nan)

            demanda_bruta = calc_demanda_bruta(elem=elem, rff=rff, flag_0g_1h=flag, edad=edad)
            demanda_ajustada = calc_demanda_ajustada(elem=elem, val_pct=val_pct,
                                                     demanda_bruta=demanda_bruta, edad=edad)
            recomendacion_final = calc_recomendacion_final(demanda_ajustada=demanda_ajustada, elem=elem)
            fator = calc_fator_reajuste(val_pct=val_pct, elem=elem, edad=edad)
            status, _ = get_foliar_status(val_pct=val_pct, elem=elem, edad=edad)

            fol_res.append(round(val_pct, 8) if not pd.isna(val_pct) else np.nan)
            db_res.append(round(demanda_bruta, 6))
            da_res.append(round(demanda_ajustada, 6))
            rec_res.append(round(recomendacion_final, 6))
            fator_res.append(fator)
            status_res.append(status)

        df[f"fol_pct_{elem.lower()}"] = fol_res
        df[f"db_{elem.lower()}_kg_ha"] = db_res
        df[f"da_{elem.lower()}_kg_ha"] = da_res
        df[f"nec_{elem.lower()}_kg_ha"] = rec_res
        df[f"fator_{elem.lower()}"] = fator_res
        df[f"status_{elem.lower()}"] = status_res

    # ── Conversiones por lote y por planta (densidad real / teórica 143) ──
    for elem in ELEMENTOS_CALCULO:
        nec_ha_col = f"nec_{elem.lower()}_kg_ha"
        if nec_ha_col not in df.columns:
            continue
        df[f"nec_{elem.lower()}_kg_lote"] = kg_lote_desde_kgha(df[nec_ha_col], area_serie, densidad_serie).round(6)
        df[f"nec_{elem.lower()}_g_palma"] = kg_a_g_palma(df[nec_ha_col]).round(6)
        df[f"nec_{elem.lower()}_kg_palma"] = kg_a_kg_palma(df[nec_ha_col]).round(6)

    elementos_descom_ordenados = [e for e in ELEMENTOS_DESCOMPOSICION
                                  if f"da_{e.lower()}_kg_ha" in df.columns]
    g_palma_cols, kg_palma_cols = [], []

    for elem in elementos_descom_ordenados:
        da_ha_col = f"da_{elem.lower()}_kg_ha"
        df[f"da_{elem.lower()}_kg_lote"] = kg_lote_desde_kgha(df[da_ha_col], area_serie, densidad_serie).round(6)
        df[f"da_{elem.lower()}_g_palma"] = kg_a_g_palma(df[da_ha_col]).round(6)
        df[f"da_{elem.lower()}_kg_palma"] = kg_a_kg_palma(df[da_ha_col]).round(6)
        g_palma_cols.append(f"da_{elem.lower()}_g_palma")
        kg_palma_cols.append(f"da_{elem.lower()}_kg_palma")

    if g_palma_cols:
        df["total_g_palma"] = df[g_palma_cols].sum(axis=1, min_count=1).round(6)
        df["total_kg_palma"] = df[kg_palma_cols].sum(axis=1, min_count=1).round(6)
    else:
        df["total_g_palma"] = np.nan
        df["total_kg_palma"] = np.nan

    # ── Trazabilidad QA: DA de S (nec_S = da_S, sin eficiencia adicional) ──
    if "da_s_kg_ha" in df.columns:
        df["da_S_preEficiencia_kg_ha"] = df["da_s_kg_ha"]

    df = descomponer_fuentes_kalini(df, area_serie, densidad_serie)
    df = normalize_for_ui(df)

    return df

# ════════════════════════════════════════════════════
# 4. KPIs Y HELPERS
# ════════════════════════════════════════════════════


def kpi_card(col, label, value, sub="", icon="", color="#1b60a7"):
    col.markdown(f"""
    <div class="kpi-card" style="border-left-color:{color};">
        <div class="kpi-label">{icon} {label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def seccion_kpis(df: pd.DataFrame):
    edad_col  = find_col(df.columns, ["edad", "age"])
    rff_col   = find_col(df.columns, ["rff_calculo", "ton_ha_modelada", "ton_ha", "rff"])
    finca_col = find_col(df.columns, ["finca"])
    lote_col  = find_col(df.columns, ["lote"])

    cols = st.columns(5)
    kpi_card(cols[0], "Registros",   f"{len(df):,}",                                          icon="📋")
    kpi_card(cols[1], "Fincas",      str(df[finca_col].nunique()) if finca_col else "—",       icon="🏡")
    kpi_card(cols[2], "Lotes",       str(df[lote_col].nunique())  if lote_col  else "—",       icon="🌿")
    kpi_card(cols[3], "Edad media",  f"{df[edad_col].mean():.1f} años" if edad_col else "—",   icon="📅")
    kpi_card(cols[4], "RFF medio",   f"{df[rff_col].mean():.2f} t/ha" if rff_col else "—",    icon="📊", color="#1A6B3C")


def tab_info():
    st.markdown(
        '<div class="section-title">¿Cómo funciona la calculadora?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0A3D62 0%, #1A6B3C 100%);
            padding: 1.2rem 1.4rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.2rem;
        ">
            <div style="font-size: 1.25rem; font-weight: 700;">
                Motor de recomendación nutricional
            </div>
            <div style="font-size: 0.9rem; margin-top: 0.4rem; opacity: 0.92;">
                La calculadora estima la necesidad de cada nutriente a partir del
                rendimiento, el material genético y el estado foliar, y la convierte
                en dosis de producto físico (fuentes comerciales) usando el modelo
                de Kalini: una fuente por nutriente.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    pasos = [
        ("1", "Rendimiento y genética",
         "Se identifica la variedad o material y se selecciona la ecuación: "
         "Guineensis o Híbrido OxG. Para edad > 26 se usa el dato real con "
         "techo de 14 ton/ha.", "#1b60a7"),
        ("2", "Demanda bruta",
         "Se calcula la cantidad inicial de nutriente requerida según las "
         "toneladas de RFF por hectárea.", "#2ca02c"),
        ("3", "Corrección foliar",
         "La demanda se compara contra el rango foliar objetivo según la edad: "
         "rangos PNP para edad ≤ 3 (inmadura) y rangos PP para el resto. "
         "Si está bajo, se corrige; si está alto, se reduce.", "#f39c12"),
        ("4", "Fuentes de Kalini",
         "Cada nutriente se convierte en dosis de su fuente comercial (Urea, "
         "SPT, KCl, Kieserita, Granubor, Cal dolomita) y se arma el compuesto.",
         "#8e44ad"),
    ]

    cols = st.columns(4)

    for col, (numero, titulo, descripcion, color) in zip(cols, pasos):
        with col:
            st.markdown(
                f"""
                <div style="
                    min-height: 210px;
                    padding: 1rem;
                    border-radius: 10px;
                    background: #ffffff;
                    border-top: 5px solid {color};
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                ">
                    <div style="
                        display: inline-flex; align-items: center; justify-content: center;
                        width: 30px; height: 30px; border-radius: 50%;
                        background: {color}; color: white; font-weight: 700;
                        margin-bottom: 0.55rem;
                    ">{numero}</div>
                    <div style="color: #0A3D62; font-size: 0.98rem; font-weight: 700;
                                margin-bottom: 0.45rem;">{titulo}</div>
                    <div style="color: #52616B; font-size: 0.82rem; line-height: 1.45;">
                        {descripcion}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    col_formula, col_rules = st.columns([1.15, 1])

    with col_formula:
        st.markdown("#### Ecuaciones principales")

        st.markdown("**Demanda bruta**")
        st.latex(r"DB_e = a_e \times RFF + b_e")

        st.markdown("**Demanda ajustada**")
        st.latex(
            r"""
            DA_e =
            \begin{cases}
            DB_e + \left(\frac{Min_e-Foliar_e}{\Delta_e}\right)
            \times G_e \times \frac{143}{1000},
            & Foliar_e < Min_e \\[6pt]
            DB_e/2,
            & Foliar_e > Max_e \\[6pt]
            DB_e,
            & Min_e \le Foliar_e \le Max_e
            \end{cases}
            """
        )

        st.markdown("**Dosis de fuente**")
        st.latex(
            r"\text{Fuente}_e\ (\text{kg/ha}) = \frac{DA_e}{\text{Aporte}_e}"
        )

        st.markdown("**Fórmula compuesta**")
        st.latex(
            r"\text{Total} = \sum_{e \notin\{\text{Cal}\}} \text{Fuente}_e"
            r"\quad\Rightarrow\quad"
            r"\% e = \frac{DA_e}{\text{Total}} \times 100"
        )

    with col_rules:
        st.markdown("#### Reglas de decisión")

        st.markdown(
            """
            <div style="background: #F0F7FF; border-left: 4px solid #1b60a7;
                        padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.65rem;">
                <b>Foliar bajo</b><br>
                <span style="font-size: 0.85rem;">
                Se adiciona una corrección proporcional al déficit foliar.
                </span>
            </div>
            <div style="background: #F2FBF3; border-left: 4px solid #2ca02c;
                        padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.65rem;">
                <b>Foliar óptimo</b><br>
                <span style="font-size: 0.85rem;">
                La demanda ajustada conserva la demanda bruta.
                </span>
            </div>
            <div style="background: #FFF8E7; border-left: 4px solid #f39c12;
                        padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.65rem;">
                <b>Foliar alto</b><br>
                <span style="font-size: 0.85rem;">
                La demanda bruta se reduce al 50&nbsp;%.
                </span>
            </div>
            <div style="background: #F8F1FC; border-left: 4px solid #8e44ad;
                        padding: 0.8rem 1rem; border-radius: 6px;">
                <b>Modelo de  </b><br>
                <span style="font-size: 0.85rem;">
                Una fuente por nutriente. La cal dolomita se excluye del total
                del compuesto (es enmienda).
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="background: #FAFAFA; border: 1px solid #DDE3E8;
                    padding: 0.9rem 1rem; border-radius: 8px;
                    font-size: 0.88rem; color: #52616B;">
            <b>Flujo de unidades:</b>
            análisis foliar original → porcentaje foliar comparable →
            demanda bruta en kg/ha → demanda ajustada en kg/ha →
            recomendación final según eficiencia →
            <b>dosis de producto por fuente</b> en kg/ha, kg/lote y g/palma.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Modelo — Fuentes y aportes</div>',
        unsafe_allow_html=True
    )

    fuente_rows = []
    for elem, meta in FUENTES_KALINI.items():
        fuente_rows.append({
            "Nutriente": elem,
            "Fuente comercial": meta["fuente"].replace("_", " "),
            "Aporte (%)": f"{meta['aporte']*100:.0f} %",
            "Entra en fórmula compuesta": "No (enmienda)"
                if meta["fuente"] in FUENTES_EXCLUIDAS_FORMULA else "Sí",
        })

    st.dataframe(
        pd.DataFrame(fuente_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">Parámetros de referencia — Motor de cálculo</div>',
        unsafe_allow_html=True
    )

    ref_rows = []

    for elem in ELEMENTOS_CALCULO:
        r = FOLIAR_RANGES[elem]
        es_oficial = elem in ELEMENTOS_OFICIALES

        coef_g = DB_COEF_GUINEENSIS.get(elem)
        coef_h = DB_COEF_HIBRIDO.get(elem)

        if coef_g:
            ecuacion_g = f"{coef_g[0]}·t/ha {coef_g[1]:+.4f}"
        else:
            ecuacion_g = f"EXPORT×{EXPORT_COEF.get(elem, '—')}"

        if coef_h:
            ecuacion_h = f"{coef_h[0]}·t/ha {coef_h[1]:+.4f}"
        else:
            ecuacion_h = "—"

        ref_rows.append({
            "Nutriente": elem,
            "Mínimo foliar": r["min"],
            "Máximo foliar": r["max"],
            "Unidad": r["unidad"],
            "Ecuación Guineensis": ecuacion_g,
            "Ecuación Híbrido": ecuacion_h,
            "g/planta (Δ)": G_PLANTA.get(elem, "—"),
            "Delta": DELTA.get(elem, "—"),
            "Eficiencia": (
                "1.0 (da = rec)" if es_oficial else f"{EFICIENCIA.get(elem, 1.0)}"
            ),
        })

    st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)

    st.markdown("---")


def tab_resumen(df: pd.DataFrame):
    st.markdown('<div class="section-title">Resumen de resultados</div>', unsafe_allow_html=True)

    n_total = len(df)

    n_inferidos = 0
    pct = 0.0
    if "flag_inferido" in df.columns:
        try:
            n_inferidos = int(df["flag_inferido"].fillna(0).astype(int).sum())
        except Exception:
            n_inferidos = int(pd.to_numeric(df["flag_inferido"], errors="coerce").fillna(0).sum())
        pct = (n_inferidos / n_total * 100) if n_total > 0 else 0.0

    if "fuente_rff" in df.columns:
        n_mayor_26 = int(df["fuente_rff"].isin(["dato_real_mayor_26", "techo_mayor_26"]).sum())
        if n_mayor_26 > 0:
            st.info(
                f"📅 **{n_mayor_26} lotes** tienen edad > 26 años. La curva "
                f"modelada no es válida para esas edades, por lo que se usó el "
                f"dato real de `ton/ha` con techo de 14 t/ha (o 14 fijo como "
                f"respaldo). Revisa la columna `fuente_rff` en el export."
            )

    if "flag_edad_faltante" in df.columns:
        n_sin_edad = int(pd.to_numeric(df["flag_edad_faltante"], errors="coerce").fillna(0).sum())
        if n_sin_edad > 0:
            st.warning(
                f"⚠️ **{n_sin_edad} lotes** no tienen dato de edad. Se conservaron en el "
                f"dataset (no se eliminaron) usando RFF de respaldo. Revisa `flag_edad_faltante`."
            )

    elementos = []
    for cand in ELEMENTOS_CALCULO:
        fol_cand = find_col(df.columns, [f"fol_pct_{cand.lower()}", f"fol_pct_{cand.upper()}", f"fol_pct_{cand}"])
        if fol_cand:
            elementos.append(cand)
    if not elementos:
        st.info("No se detectaron columnas foliares en el dataset.")
        return

    rows = []
    for elem in elementos:
        fol_col = find_col(df.columns, [f"fol_pct_{elem.lower()}", f"fol_pct_{elem.upper()}", f"fol_pct_{elem}"])
        sta_col = find_col(df.columns, [f"status_{elem.lower()}", f"status_{elem.upper()}", f"estado_{elem.lower()}", f"estado_{elem.upper()}"])
        nec_col = find_col(df.columns, [f"Rec_da_{elem}_kgha", f"rec_da_{elem.lower()}_kgha", f"nec_{elem.lower()}_kg_ha", f"nec_{elem}_kg_ha"])
        r = FOLIAR_RANGES.get(elem, {"unidad": "%", "min": None, "max": None})

        if fol_col and fol_col in df.columns:
            vals = pd.to_numeric(df[fol_col], errors="coerce").dropna()
        else:
            vals = pd.Series(dtype=float)
        media = vals.mean() if len(vals) > 0 else np.nan

        try:
            status_label = get_foliar_status(media, elem)[1]
        except Exception:
            status_label = "—"

        nec_media = None
        if nec_col and nec_col in df.columns:
            nec_media = pd.to_numeric(df[nec_col], errors="coerce").mean()

        dist = {}
        if sta_col and sta_col in df.columns:
            dist_ser = df[sta_col].astype(str).str.lower().value_counts()
            dist = {
                "bajo": int(dist_ser.get("bajo", 0)),
                "critico": int(dist_ser.get("critico", 0)),
                "optimo": int(dist_ser.get("optimo", 0)),
                "alto": int(dist_ser.get("alto", 0)),
            }
        else:
            dist = {"bajo": 0, "critico": 0, "optimo": 0, "alto": 0}

        rows.append({
            "Nutriente": elem,
            "Unidad dataset": r.get("unidad", "%"),
            "Rango Óptimo (%)": f"{r.get('min','—')} – {r.get('max','—')}",
            "Media Foliar": f"{media:.3f}" if not pd.isna(media) else "—",
            "Estado": status_label,
            "Nec. media (kg/ha)": f"{nec_media:.2f}" if nec_media is not None and not pd.isna(nec_media) else "—",
            "Lotes Bajo/Crítico": dist["bajo"] + dist["critico"],
            "Lotes Óptimo": dist["optimo"],
            "Lotes Alto": dist["alto"],
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    rec_cols = [c for c in df.columns if c.startswith("Rec_da_") and c.endswith("_kgha")]
    lote_col = find_col(df.columns, ["lote", "Lote", "block", "id_lote"])

    st.markdown(
        '<div class="section-title">Mapa de Calor — Recomendación por Lote × Nutriente</div>',
        unsafe_allow_html=True
    )

    if rec_cols and lote_col:
        df_heat = df[[lote_col] + rec_cols].copy()
        df_heat = df_heat.set_index(lote_col)
        df_heat.columns = [re.sub(r"Rec_da_(.+?)_kgha", lambda m: m.group(1).upper(), c) for c in df_heat.columns]
        df_heat = df_heat.head(40)

        fig_h = px.imshow(
            df_heat,
            color_continuous_scale="YlOrRd",
            text_auto=".1f",
            labels=dict(x="Nutriente", y="Lote", color="kg/ha"),
            aspect="auto",
        )
        fig_h.update_layout(
            height=max(300, len(df_heat) * 18 + 80),
            margin=dict(t=10, l=0, r=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_h, use_container_width=True, key="resumen_heatmap")

        formula_col = find_col(df.columns, ["Formula_Kalini", "Formula (N-P-K-MgO-B)",
                                            "Formula (N-P-K-MgO-S-B)", "formula_kalini", "formula"])

        if formula_col and formula_col in df.columns:
            st.markdown(
                '<div class="section-title">Fórmulas compuestas más frecuentes</div>',
                unsafe_allow_html=True
            )
            top_formulas = df[formula_col].dropna().value_counts().head(8).reset_index()
            top_formulas.columns = ["Fórmula", "Lotes"]
            if len(top_formulas) > 0:
                cols = st.columns(min(len(top_formulas), 6))
                for col, (_, r) in zip(cols, top_formulas.iterrows()):
                    col.markdown(
                        f'<div class="formula-badge">{r["Fórmula"]}</div>'
                        f'<div style="font-size:0.75rem;color:#7A8899;margin-left:0.3rem;">'
                        f'{int(r["Lotes"])} lotes</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No hay fórmulas compuestas para mostrar.")

            edad_col_f = find_col(df.columns, ["edad", "age", "anos", "ano_planta", "idade"])
            if edad_col_f:
                st.markdown(
                    '<div class="section-title">Fórmulas compuestas por edad</div>',
                    unsafe_allow_html=True
                )
                st.caption(
                    "Variabilidad de fórmulas por edad, para apoyar la reducción y "
                    "compatibilización de formulaciones."
                )

                modo_edad = st.radio(
                    "Agrupar la edad por:",
                    options=["Franja etaria", "Edad exacta"],
                    horizontal=True,
                    key="res_form_edad_modo",
                )

                base = df[[edad_col_f, formula_col]].copy()
                base["_formula"] = base[formula_col].astype(str).replace({"nan": np.nan, "None": np.nan})
                base = base.dropna(subset=["_formula"])

                if base.empty:
                    st.info("No hay fórmulas compuestas calculadas para agrupar por edad.")
                else:
                    if modo_edad == "Franja etaria":
                        col_edad = "Franja de edad"
                        base[col_edad] = base[edad_col_f].apply(franja_edad)
                        orden_cat = pd.CategoricalDtype(FRANJAS_EDAD, ordered=True)
                        base[col_edad] = base[col_edad].astype(orden_cat)
                    else:
                        col_edad = "Edad (años)"
                        base[col_edad] = base[edad_col_f].apply(lambda x: to_float_safe(x, np.nan)).round(0)

                    tab_formula_edad = (
                        base.groupby([col_edad, "_formula"], observed=True)
                        .size()
                        .reset_index(name="Lotes")
                    )
                    tab_formula_edad["% del grupo"] = (
                        tab_formula_edad.groupby(col_edad, observed=True)["Lotes"]
                        .transform(lambda s: (s / s.sum() * 100).round(1))
                    )
                    tab_formula_edad = tab_formula_edad.sort_values(
                        [col_edad, "Lotes"], ascending=[True, False]
                    )
                    tab_formula_edad = tab_formula_edad.rename(columns={"_formula": "Fórmula"})

                    # Nº de fórmulas distintas por edad (para ver dónde compatibilizar)
                    n_formulas_por_edad = (
                        tab_formula_edad.groupby(col_edad, observed=True)["Fórmula"]
                        .nunique()
                        .reset_index(name="N° fórmulas distintas")
                    )

                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown("##### Detalle por edad")
                        st.dataframe(tab_formula_edad, use_container_width=True, hide_index=True)
                    with c2:
                        st.markdown("##### Variabilidad por edad")
                        st.dataframe(n_formulas_por_edad, use_container_width=True, hide_index=True)

                    modal = (
                        tab_formula_edad.sort_values("Lotes", ascending=False)
                        .drop_duplicates(subset=[col_edad])
                        .sort_values(col_edad)
                    )
                    st.caption("Fórmula predominante por grupo de edad:")
                    mcols = st.columns(min(len(modal), 5)) if len(modal) > 0 else []
                    for col, (_, r) in zip(mcols, modal.iterrows()):
                        col.markdown(
                            f'<div class="formula-badge">{r[col_edad]}: {r["Fórmula"]}</div>'
                            f'<div style="font-size:0.75rem;color:#7A8899;margin-left:0.3rem;">'
                            f'{int(r["Lotes"])} lotes ({r["% del grupo"]}%)</div>',
                            unsafe_allow_html=True
                        )

                    csv_edad = tab_formula_edad.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 Descargar fórmulas por edad (CSV)",
                        data=csv_edad,
                        file_name="formulas_por_edad.csv",
                        mime="text/csv",
                        key="dl_formulas_edad",
                    )

    if n_inferidos > 0:
        st.warning(
            f"⚠️ **{n_inferidos} de {n_total} lotes ({pct:.1f}%)** no tenían "
            f"el flag `0G_1H` en el dataset original y se les asumió "
            f"**Guineensis** por defecto (inferido por Variedad/Material). "
            f"Revisar con campo si corresponde."
        )

    st.markdown("---")


def tab_agrupaciones(df: pd.DataFrame):

    st.markdown('<div class="section-title">Agrupaciones — Estadísticas por grupo</div>', unsafe_allow_html=True)

    cat_candidates = []
    for c in ["finca", "departamento", "manejo", "material", "variedad", "edad", "lote",
              "especie", "Foliar"]:
        col = find_col(df.columns, [c])
        if col:
            cat_candidates.append(col)
    if not cat_candidates:
        st.info("No se detectaron columnas categóricas (finca, departamento, manejo, material, variedad, lote).")
        return

    tipo_var = st.radio(
        "Tipo de variable a explorar",
        options=["Fuentes", "Nutrientes", "Ambos"],
        index=2,
        horizontal=True,
        key="agr_tipo_var",
        help=(
            "Fuentes: Urea, SPT, KCl, Kieserita, Granubor, Sulfato, Cal dolomita. "
            "Nutrientes: N, P, K, Ca, Mg, B, S (demanda agronómica Rec_da_*). "
            "Ambos: lista completa."
        ),
    )

    unidad_opt = st.radio(
        "Tipo de unidad a listar",
        options=["kgpalma", "kgha", "kglote", "gpalma", "totales", "todas"],
        index=0,
        horizontal=True,
        key="agr_unidad_opt",
        help=(
            "'kgpalma' es la unidad base de la recomendación (kg/planta, densidad teórica 143). "
            "'totales' muestra únicamente las columnas agregadas (Rec_total_n_*, Rec_total_f_*)."
        ),
    )

    PAT_FUENTES  = r"Rec_(urea|SPT|kcl|kieserita|granubor|sulfato|caldol)_"
    PAT_NUTRI    = r"Rec_da_"
    PAT_TOTAL_N  = r"Rec_total_n_"
    PAT_TOTAL_F  = r"Rec_total_f_"

    def _es_fuente(c):   return bool(re.match(PAT_FUENTES, c, re.I))
    def _es_nutri(c):    return bool(re.match(PAT_NUTRI,   c, re.I))
    def _es_total_n(c):  return bool(re.match(PAT_TOTAL_N, c, re.I))
    def _es_total_f(c):  return bool(re.match(PAT_TOTAL_F, c, re.I))
    def _es_total(c):    return _es_total_n(c) or _es_total_f(c)

    def _match_unidad(col, unidad):
        cl = col.lower()
        if unidad == "kgha":
            return cl.endswith("_kgha") or "_kg_ha" in cl
        if unidad == "kgpalma":
            return cl.endswith("_kgpalma")
        if unidad == "kglote":
            return cl.endswith("_kglote") or "_kg_lote" in cl
        if unidad == "gpalma":
            return cl.endswith("_gpalma") or "_g_palma" in cl
        return True

    rec_cols = [c for c in df.columns if c.startswith("Rec_")]
    if not rec_cols:
        rec_cols = [
            c for c in df.columns
            if c.lower().startswith("da_") or c.lower().startswith("nec_") or c.lower().startswith("total_")
        ]

    candidate_num = []
    for c in rec_cols:
        if tipo_var == "Fuentes" and not (_es_fuente(c) or _es_total_f(c)):
            continue
        if tipo_var == "Nutrientes" and not (_es_nutri(c) or _es_total_n(c)):
            continue

        if unidad_opt == "totales":
            if tipo_var == "Fuentes" and not _es_total_f(c):
                continue
            if tipo_var == "Nutrientes" and not _es_total_n(c):
                continue
            if tipo_var == "Ambos" and not _es_total(c):
                continue
            candidate_num.append(c)
            continue

        if _es_total(c):
            continue

        if unidad_opt != "todas" and not _match_unidad(c, unidad_opt):
            continue
        candidate_num.append(c)

    if not candidate_num:
        for c in df.columns:
            cl = c.lower()
            if unidad_opt == "kgha" and (cl.endswith("_kgha") or cl.endswith("_kg_ha")):
                candidate_num.append(c)
            elif unidad_opt == "kgpalma" and cl.endswith("_kgpalma"):
                candidate_num.append(c)
            elif unidad_opt == "kglote" and (cl.endswith("_kglote") or cl.endswith("_kg_lote")):
                candidate_num.append(c)
            elif unidad_opt == "gpalma" and (cl.endswith("_gpalma") or "_g_palma" in cl):
                candidate_num.append(c)
            elif unidad_opt == "totales" and ("total_n" in cl or "total_f" in cl):
                candidate_num.append(c)
            elif unidad_opt == "todas":
                candidate_num.append(c)
        candidate_num = list(dict.fromkeys(candidate_num))

    if not candidate_num:
        st.info("No se encontraron columnas numéricas tras aplicar filtros. Elige 'todas' o revisa el dataset.")
        return

    edad_col_real = find_col(df.columns, ["edad"])

    with st.expander("Configuración de agrupación", expanded=True):
        agrup_col = st.selectbox(
            "Selecciona variable categórica para agrupar:",
            options=cat_candidates, index=0, key="agr_col",
        )

        opciones_multi = ["(ninguno)"] + [c for c in cat_candidates if c != agrup_col]
        multi_col = st.selectbox(
            "Agrupar adicionalmente por (opcional):",
            options=opciones_multi, index=0, key="agr_multi",
            help=(
                "Cruce tipo Edad × Departamento (acuerdo INCERES: demanda media por "
                "franja etaria dentro de cada departamento, para reducir el número de fórmulas)."
            ),
        )
        if multi_col not in opciones_multi:
            multi_col = "(ninguno)"

        grupo_cols_preview = [c for c in [agrup_col, multi_col] if c and c != "(ninguno)"]
        franjas_edad = False
        if edad_col_real and edad_col_real in grupo_cols_preview:
            franjas_edad = st.checkbox(
                "Convertir edad en franjas etarias (≤3, 4–8, 9–13, 14–18, 19–23, 24–28, 29+)",
                value=False, key="agr_franjas",
                help="Reduce la cantidad de grupos (y de fórmulas distintas) al agrupar por edad.",
            )

        var_num = st.selectbox(
            "Selecciona variable numérica:",
            options=candidate_num, index=0, key="agr_var",
        )
        top_n = st.number_input(
            "Mostrar top N grupos (0 = todos):",
            min_value=0, max_value=1000, value=0, step=10, key="agr_top_n",
        )

    grupo_cols = list(dict.fromkeys(
        [c for c in [agrup_col, multi_col] if c and c != "(ninguno)"]
    ))

    trabajo = df[grupo_cols].copy()

    if franjas_edad and edad_col_real and edad_col_real in grupo_cols:
        nombre_f = f"{edad_col_real}_franja"
        trabajo[nombre_f] = trabajo[edad_col_real].apply(franja_edad)
        grupo_cols[grupo_cols.index(edad_col_real)] = nombre_f
        trabajo = trabajo.drop(columns=[edad_col_real])

    serie_num = pd.to_numeric(df[var_num], errors="coerce")
    trabajo[var_num] = serie_num

    try:
        agg = trabajo.groupby(grupo_cols, dropna=False)[var_num].agg(
            ['count', 'sum', 'mean', 'min', 'max']
        )
    except Exception as e:
        st.error(f"Error al agrupar: {e}")
        return

    decimals = 2
    agg['sum']  = agg['sum'].round(decimals)
    agg['mean'] = agg['mean'].round(decimals)
    agg['min']  = agg['min'].round(decimals)
    agg['max']  = agg['max'].round(decimals)
    agg = agg.reset_index()

    order_col = 'sum' if agg['sum'].notna().sum() > 0 and agg['sum'].abs().sum() > 0 else 'count'
    agg = agg.sort_values(by=order_col, ascending=False)

    view = agg.head(top_n) if top_n > 0 else agg

    st.markdown("#### Resultado")
    st.dataframe(view, use_container_width=True)

    csv_bytes = agg.to_csv(index=False).encode("utf-8-sig")
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_var = re.sub(r'[^0-9A-Za-z_]+', '', "_".join(grupo_cols) + "_" + var_num)
    st.download_button(
        "📥 Descargar agrupaciones (CSV)", data=csv_bytes,
        file_name=f"agrupaciones_{safe_var}_{now}.csv", mime="text/csv",
    )

    st.markdown("---")

DENSIDAD_PALMAS_HA = 143


def _n_palmas_constante(index) -> "pd.Series":
    return pd.Series(DENSIDAD_PALMAS_HA, index=index, dtype=float)


def _to_g_palma_from_kgha(kgha_s, densidad: float = DENSIDAD_PALMAS_HA) -> "pd.Series":
    kgha_s = pd.to_numeric(kgha_s, errors="coerce")
    return (kgha_s * 1000.0 / densidad).replace([np.inf, -np.inf], np.nan)


def _find_caldol_columns(df_cols):
    result = {"kgha": None, "kglote": None, "gpalma": None, "tonlote": None, "kgpalma": None}

    def is_caldol(name):
        n = str(name).lower()
        if re.search(r"caldol", n): return True
        if re.search(r"cal[\s_\-]*dol", n): return True
        if re.search(r"dolom", n): return True
        if re.search(r"calam", n): return True
        return False

    def detect_unit(name):
        n = str(name).lower()
        if re.search(r"ton", n) and re.search(r"lote", n): return "tonlote"
        if re.search(r"kg", n) and re.search(r"lote", n): return "kglote"
        if re.search(r"kgpalma|kg_palma", n) or (re.search(r"kg", n) and re.search(r"palma", n)):
            return "kgpalma"
        if (re.search(r"g\b", n) or "gpalma" in n or "g_palma" in n or "/palma" in n or "_palma" in n) \
           and "lote" not in n and "kg" not in n:
            return "gpalma"
        if re.search(r"ha\b|/ha|kgha", n): return "kgha"
        return "kgha"

    for c in df_cols:
        if not is_caldol(c):
            continue
        unit = detect_unit(c)
        if result[unit] is None:
            result[unit] = c
    return result


def _to_kg_palma_from_g_palma(g_palma_series):
    return (g_palma_series / 1000.0)


def _round_cols(df, decimals_default=2, decimals_gpalma=4):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype in [np.float64, np.float32, float]:
            cl = c.lower()
            if cl.endswith("_gpalma") or cl.endswith("_kgpalma"):
                df[c] = df[c].round(decimals_gpalma)
            elif cl.startswith("fol_pct_"):
                df[c] = df[c].round(5)
            else:
                df[c] = df[c].round(decimals_default)
    return df

def _safe_find(df, candidates):
    for cand in candidates:
        col = find_col(df.columns, [cand])
        if col:
            return col
    return None

EXPORT_NAME_MAP_BASE = {
    # Agronómica (kg/ha, kg/lote, g/palma)
    "da_{}_kg_ha": "Rec_da_{}_kgha",
    "da_{}_kg_lote": "Rec_da_{}_kglote",
    "da_{}_g_palma": "Rec_da_{}_gpalma",
    # Totales por palma (g)
    "total_g_palma": "Rec_total_n_gpalma",
    # Fuentes (kg/ha, kg/lote, g/palma)
    "fuente_{}_kg_ha": "Rec_{}_kgha",
    "fuente_{}_kg_lote": "Rec_{}_kglote",
    "fuente_{}_g_palma": "Rec_{}_gpalma",
    "total_fuentes_g_palma": "Rec_total_f_gpalma",
    # Cal dolomita (separada)
    "fuente_cal_dolomita_kg_ha": "Rec_caldol_kgha",
    "fuente_cal_dolomita_kg_lote": "Rec_caldol_kglote",
    "fuente_cal_dolomita_g_palma": "Rec_caldol_gpalma",
    # Fórmula
    "Formula_Kalini": "Formula (N-P-K-MgO-B)",
}

COLUMNAS_REC_ORDEN = [
    # Demanda ajustada — kg/ha
    "Rec_da_N_kgha", "Rec_da_P_kgha", "Rec_da_K_kgha", "Rec_da_Ca_kgha",
    "Rec_da_Mg_kgha", "Rec_da_B_kgha", "Rec_da_S_kgha",
    # Demanda ajustada — kg/palma (unidad base de la recomendación)
    "Rec_da_N_kgpalma", "Rec_da_P_kgpalma", "Rec_da_K_kgpalma", "Rec_da_Ca_kgpalma",
    "Rec_da_Mg_kgpalma", "Rec_da_B_kgpalma", "Rec_da_S_kgpalma",
    # Demanda ajustada — g/palma
    "Rec_da_N_gpalma", "Rec_da_P_gpalma", "Rec_da_K_gpalma", "Rec_da_Ca_gpalma",
    "Rec_da_Mg_gpalma", "Rec_da_B_gpalma", "Rec_da_S_gpalma",
    # Demanda ajustada — kg/lote (densidad real)
    "Rec_da_N_kglote", "Rec_da_P_kglote", "Rec_da_K_kglote", "Rec_da_Ca_kglote",
    "Rec_da_Mg_kglote", "Rec_da_B_kglote", "Rec_da_S_kglote",
    # Totales de nutrientes por palma
    "Rec_total_n_kgpalma", "Rec_total_n_gpalma",
    # Fuentes comerciales — kg/ha
    "Rec_urea_kgha", "Rec_SPT_kgha", "Rec_kcl_kgha", "Rec_kieserita_kgha",
    "Rec_granubor_kgha", "Rec_sulfato_kgha",
    # Fuentes comerciales — kg/palma
    "Rec_urea_kgpalma", "Rec_SPT_kgpalma", "Rec_kcl_kgpalma", "Rec_kieserita_kgpalma",
    "Rec_granubor_kgpalma", "Rec_sulfato_kgpalma",
    # Fuentes comerciales — g/palma
    "Rec_urea_gpalma", "Rec_SPT_gpalma", "Rec_kcl_gpalma", "Rec_kieserita_gpalma",
    "Rec_granubor_gpalma", "Rec_sulfato_gpalma",
    # Fuentes comerciales — kg/lote
    "Rec_urea_kglote", "Rec_SPT_kglote", "Rec_kcl_kglote", "Rec_kieserita_kglote",
    "Rec_granubor_kglote", "Rec_sulfato_kglote",
    # Totales de fuentes
    "Rec_total_f_kgpalma", "Rec_total_f_gpalma",
    # Fórmula compuesta
    "Formula (N-P-K-MgO-B)",
    # Cal dolomita (enmienda, fuera del compuesto)
    "Rec_caldol_kgha", "Rec_caldol_kglote", "Rec_caldol_gpalma",
    "Rec_caldol_kgpalma", "Rec_caldol_tonlote",
]

def tab_exportar(df: pd.DataFrame, nombre_excel_salida: str = None):

    df = df.copy()

    lote_col = _safe_find(df, ["lote", "block", "cod_lote", "id_lote"])
    finca_col = _safe_find(df, ["finca", "farm", "hacienda"])
    departamento_col = _safe_find(df, ["departamento", "depto", "region", "zona"])
    edad_col = _safe_find(df, ["edad", "age", "anos"])
    area_col = _safe_find(df, ["area", "ha", "hectareas", "superficie"])
    n_palmas_col = _safe_find(df, ["n_palmas", "numero_palmas", "palmas", "plantas", "plantas_ha"])

    if "densidad_real" in df.columns:
        dens_s = pd.to_numeric(df["densidad_real"], errors="coerce").fillna(DENSIDAD_TEORICA_HA)
    else:
        dens_s = pd.Series(DENSIDAD_TEORICA_HA, index=df.index)

    area_s = df[area_col].apply(lambda x: to_float_safe(x, np.nan)) if area_col in df.columns else pd.Series(np.nan, index=df.index)
    n_palmas_s = df[n_palmas_col].apply(lambda x: to_float_safe(x, np.nan)) if n_palmas_col in df.columns else pd.Series(np.nan, index=df.index)

    for e in ["N", "P", "K", "Ca", "Mg", "B", "S"]:
        if e == "S":
            cand_ha = find_col(df.columns, [f"nec_{e.lower()}_kg_ha", f"da_{e.lower()}_kg_ha",
                                            f"nec_{e} (kg/ha)", f"da_{e} (kg/ha)"])
            cand_lote = find_col(df.columns, [f"nec_{e.lower()}_kg_lote", f"da_{e.lower()}_kg_lote",
                                              f"{e.lower()}_lote", f"da_{e}_lote"])
            cand_gpalma = find_col(df.columns, [f"nec_{e.lower()}_g_palma", f"da_{e.lower()}_g_palma",
                                                f"{e.lower()}_g_palma", f"da_{e}/palma (g)"])
        else:
            cand_ha = find_col(df.columns, [f"da_{e.lower()}_kg_ha", f"da_{e} (kg/ha)", f"da_{e.lower()} (kg/ha)"])
            cand_lote = find_col(df.columns, [f"da_{e.lower()}_kg_lote", f"da_{e}_lote (kg)", f"da_{e}_lote"])
            cand_gpalma = find_col(df.columns, [f"da_{e.lower()}_g_palma", f"da_{e}/palma (g)"])

        if cand_ha:
            df[f"Rec_da_{e}_kgha"] = df[cand_ha].apply(lambda x: to_float_safe(x, np.nan))

        if cand_lote:
            df[f"Rec_da_{e}_kglote"] = df[cand_lote].apply(lambda x: to_float_safe(x, np.nan))
        elif cand_ha and area_col in df.columns:
            df[f"Rec_da_{e}_kglote"] = kg_lote_desde_kgha(df[f"Rec_da_{e}_kgha"], area_s, dens_s)

        if cand_gpalma:
            df[f"Rec_da_{e}_gpalma"] = df[cand_gpalma].apply(lambda x: to_float_safe(x, np.nan))
        elif cand_ha:
            df[f"Rec_da_{e}_gpalma"] = kg_a_g_palma(df[f"Rec_da_{e}_kgha"])
        elif f"Rec_da_{e}_kglote" in df.columns and n_palmas_col in df.columns:
            df[f"Rec_da_{e}_gpalma"] = _to_g_palma_from_kg_lote(df[f"Rec_da_{e}_kglote"], n_palmas_s)

        if f"Rec_da_{e}_kgha" in df.columns:
            df[f"Rec_da_{e}_kgpalma"] = kg_a_kg_palma(df[f"Rec_da_{e}_kgha"])
        elif f"Rec_da_{e}_gpalma" in df.columns:
            df[f"Rec_da_{e}_kgpalma"] = pd.to_numeric(df[f"Rec_da_{e}_gpalma"], errors="coerce") / 1000.0

    gpal_cols = [c for c in df.columns if c.startswith("Rec_da_") and c.endswith("_gpalma")]
    if gpal_cols:
        df["Rec_total_n_gpalma"] = df[gpal_cols].sum(axis=1, min_count=1)
        df["Rec_total_n_kgpalma"] = pd.to_numeric(df["Rec_total_n_gpalma"], errors="coerce") / 1000.0

    fuente_map = {
        "urea": ["fuente_urea_kg_ha", "urea (kg/ha)", "urea_ (kg/ha)"],
        "spt": ["fuente_spt_kg_ha", "spt (kg/ha)"],
        "kcl": ["fuente_kcl_kg_ha", "kcl (kg/ha)"],
        "kieserita": ["fuente_kieserita_kg_ha", "kieserita (kg/ha)"],
        "granubor": ["fuente_granubor_kg_ha", "granubor (kg/ha)"],
        "sulfato": ["fuente_sulfato_kg_ha", "sulfato (kg/ha)", "sulfato_ (kg/ha)"],
    }
    for key, cands in fuente_map.items():
        found = None
        for cand in cands:
            _col = find_col(df.columns, [cand])
            if _col:
                found = _col
                break
        if not found:
            continue

        df[f"Rec_{key}_kgha"] = df[found].apply(lambda x: to_float_safe(x, np.nan))
        df[f"Rec_{key}_gpalma"] = kg_a_g_palma(df[f"Rec_{key}_kgha"])
        df[f"Rec_{key}_kgpalma"] = kg_a_kg_palma(df[f"Rec_{key}_kgha"])

        cand_lote_motor = find_col(df.columns, [f"fuente_{key}_kg_lote"])
        if cand_lote_motor:
            df[f"Rec_{key}_kglote"] = df[cand_lote_motor].apply(lambda x: to_float_safe(x, np.nan))
        elif area_col in df.columns:
            df[f"Rec_{key}_kglote"] = kg_lote_desde_kgha(df[f"Rec_{key}_kgha"], area_s, dens_s)
        else:
            cand_lote_alt = None
            for _cand in [found.replace(" (kg/ha)", "_lote (kg)"),
                          found.replace("_kg_ha", "_kg_lote")]:
                if _cand != found:
                    _col = find_col(df.columns, [_cand])
                    if _col and _col != found:
                        cand_lote_alt = _col
                        break
            if cand_lote_alt:
                df[f"Rec_{key}_kglote"] = df[cand_lote_alt].apply(lambda x: to_float_safe(x, np.nan))
            else:
                cand_lote_any = find_col(df.columns, [f"{key}_lote (kg)", f"{key}_lote"])
                if cand_lote_any:
                    df[f"Rec_{key}_kglote"] = df[cand_lote_any].apply(lambda x: to_float_safe(x, np.nan))

    for old, new in [("Rec_spt_kgha", "Rec_SPT_kgha"),
                     ("Rec_spt_kglote", "Rec_SPT_kglote"),
                     ("Rec_spt_gpalma", "Rec_SPT_gpalma"),
                     ("Rec_spt_kgpalma", "Rec_SPT_kgpalma")]:
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    caldol = _find_caldol_columns(df.columns)
    if caldol["kgha"]:
        df["Rec_caldol_kgha"] = df[caldol["kgha"]].apply(lambda x: to_float_safe(x, np.nan))

    if caldol["kglote"]:
        df["Rec_caldol_kglote"] = df[caldol["kglote"]].apply(lambda x: to_float_safe(x, np.nan))
    elif "Rec_caldol_kgha" in df.columns and area_col in df.columns:
        df["Rec_caldol_kglote"] = kg_lote_desde_kgha(df["Rec_caldol_kgha"], area_s, dens_s)

    if caldol["gpalma"]:
        df["Rec_caldol_gpalma"] = df[caldol["gpalma"]].apply(lambda x: to_float_safe(x, np.nan))
    elif "Rec_caldol_kgha" in df.columns:
        df["Rec_caldol_gpalma"] = kg_a_g_palma(df["Rec_caldol_kgha"])
    elif "Rec_caldol_kglote" in df.columns and n_palmas_col in df.columns:
        df["Rec_caldol_gpalma"] = _to_g_palma_from_kg_lote(df["Rec_caldol_kglote"], n_palmas_s)

    if "Rec_caldol_kgha" in df.columns:
        df["Rec_caldol_kgpalma"] = kg_a_kg_palma(df["Rec_caldol_kgha"])
    elif "Rec_caldol_gpalma" in df.columns:
        df["Rec_caldol_kgpalma"] = pd.to_numeric(df["Rec_caldol_gpalma"], errors="coerce") / 1000.0

    if "Rec_caldol_kglote" in df.columns:
        df["Rec_caldol_tonlote"] = (
                pd.to_numeric(df["Rec_caldol_kglote"], errors="coerce") / 1000.0
        ).replace([np.inf, -np.inf], np.nan)

    df["flag_caldol_detectada"] = any(v is not None for v in caldol.values())

    if "Formula_Kalini" in df.columns and "Formula (N-P-K-MgO-B)" not in df.columns:
        df["Formula (N-P-K-MgO-B)"] = df["Formula_Kalini"]

    fuentes_principales_g = ["Rec_urea_gpalma", "Rec_SPT_gpalma", "Rec_kcl_gpalma",
                             "Rec_kieserita_gpalma", "Rec_granubor_gpalma", "Rec_sulfato_gpalma"]
    fuentes_existentes_g = [c for c in fuentes_principales_g if c in df.columns]
    if fuentes_existentes_g:
        df["Rec_total_f_gpalma"] = (
            df[fuentes_existentes_g].apply(pd.to_numeric, errors="coerce")
            .sum(axis=1, min_count=1)
        )
        df["Rec_total_f_kgpalma"] = pd.to_numeric(df["Rec_total_f_gpalma"], errors="coerce") / 1000.0

    estado_cols = [c for c in df.columns
                   if c.lower().startswith(("estado_", "status_", "fol_fuente_", "fol_pct_"))]

    id_cols_candidates = [lote_col, finca_col, departamento_col, edad_col,
                          "material", "variedad", "manejo", area_col, n_palmas_col]
    id_cols = [c for c in id_cols_candidates if c and c in df.columns]

    prod_cols = [c for c in ["ton_ha_observada", "ton_ha_modelada", "rff_calculo",
                             "fuente_rff", "especie", "flag_0g_1h", "flag_inferido",
                             "flag_inmaduro", "flag_edad_faltante", "densidad_real",
                             "n_palmas_lote", "Foliar"]
                 if c in df.columns]

    rec_orden_existentes = [c for c in COLUMNAS_REC_ORDEN if c in df.columns]

    final_cols = id_cols + prod_cols + rec_orden_existentes + estado_cols
    final_cols = list(dict.fromkeys(final_cols))

    df_resultados = df[final_cols].copy()
    df_resultados = _round_cols(df_resultados, decimals_default=2, decimals_gpalma=4)

    eval_agro_cols = [c for c in COLUMNAS_REC_ORDEN
                      if c.startswith("Rec_da_") or c == "Rec_total_n_kgpalma"]
    eval_agro_cols = [c for c in eval_agro_cols if c in df_resultados.columns]
    eval_agro = df_resultados[id_cols + eval_agro_cols].copy()

    eval_admin_cols = [c for c in COLUMNAS_REC_ORDEN
                       if c not in eval_agro_cols and c != "Formula (N-P-K-MgO-B)"]
    eval_admin_cols = [c for c in eval_admin_cols if c in df_resultados.columns]
    eval_admin = (df_resultados[id_cols + eval_admin_cols].copy()
                  if eval_admin_cols else pd.DataFrame())

    agg_candidates = [finca_col, departamento_col, "material", "variedad", "manejo", lote_col]
    agg_by = next((c for c in agg_candidates if c and c in df.columns), None)
    if agg_by:
        agg_metrics = [c for c in df_resultados.columns if c.startswith("Rec_da_") and c.endswith("_kgha")]
        agg_metrics_g = [c for c in df_resultados.columns if c.startswith("Rec_da_") and c.endswith("_gpalma")]
        agg_cols = agg_metrics + agg_metrics_g
        if agg_cols:
            agrup = df_resultados.groupby(agg_by, dropna=False)[agg_cols].agg(['count', 'mean', 'sum', 'min', 'max'])
            agrup.columns = ['_'.join([str(i) for i in col]).strip() for col in agrup.columns.to_flat_index()]
            agrup = agrup.reset_index()
        else:
            agrup = pd.DataFrame()
    else:
        agrup = pd.DataFrame()

    resumen_finca = pd.DataFrame()
    if finca_col and finca_col in df_resultados.columns:
        resumen_metrics = [c for c in df_resultados.columns if c.startswith("Rec_da_") and c.endswith("_kgha")]
        if resumen_metrics:
            resumen = df_resultados.groupby(finca_col, dropna=False)[resumen_metrics].agg(['count', 'sum', 'mean']).round(4)
            resumen.columns = ['_'.join([str(i) for i in col]).strip() for col in resumen.columns.to_flat_index()]
            resumen_finca = resumen.reset_index()

    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if nombre_excel_salida is None:
        nombre_excel_salida = f"fertilizacion_resultados_std_{now}.xlsx"

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_resultados.to_excel(writer, index=False, sheet_name="Resultados")
        eval_agro.to_excel(writer, index=False, sheet_name="Evaluacion_Agronomica")
        if not eval_admin.empty:
            eval_admin.to_excel(writer, index=False, sheet_name="Evaluacion_Administrativa")
        if not agrup.empty:
            agrup.to_excel(writer, index=False, sheet_name="Agrupaciones")
        if not resumen_finca.empty:
            resumen_finca.to_excel(writer, index=False, sheet_name="Resumen_Finca")
    buffer.seek(0)

    st.markdown('<div class="section-title">Vista previa — Resultados estandarizados</div>', unsafe_allow_html=True)
    st.dataframe(df_resultados.head(60), use_container_width=True)

    st.download_button("📥 Descargar Excel estandarizado", data=buffer.getvalue(),
                       file_name=nombre_excel_salida,
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)

    csv_bytes = df_resultados.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Descargar CSV Resultados", data=csv_bytes,
                       file_name=f"fertilizacion_resultados_std_{now}.csv",
                       mime="text/csv", use_container_width=True)

    st.markdown("---")

    return {
        "df_resultados": df_resultados,
        "eval_agro": eval_agro,
        "eval_admin": eval_admin,
        "agrupaciones": agrup,
        "resumen_finca": resumen_finca,
        "excel_bytes": buffer.getvalue(),
        "csv_bytes": csv_bytes,
    }

# ════════════════════════════════════════════════════
# 7. SIDEBAR
# ════════════════════════════════════════════════════


def render_sidebar():
    with st.sidebar:
        try:
            img = Image.open("logo_sidebar.png")
            st.image(img, width=260)
        except Exception:
            st.markdown("## 🌴 FarmPrecision")

        st.markdown("---")

        st.markdown("### 📂 Cargar datos")

        uploaded = st.file_uploader(
            "Sube tu archivo Excel o CSV",
            type=["xlsx", "xls", "csv"],
            help=(
                "Columnas principales: Lote, Finca, Edad, ton/ha y datos foliares "
                "N_f, P_f, K_f, Ca_f, Mg_f y B_f. S_f, Cu_f, Fe_f, Mn_f y Zn_f son "
                "opcionales. La clasificación del material puede realizarse mediante "
                "0G_1H o mediante Variedad/Material."
            ),
        )

        st.markdown("---")
        st.markdown("### ℹ️ Columnas requeridas")

        st.markdown("""
        | **Variable** | **Nombre interno** | **Variantes aceptadas** |
        |:-------------|:-------------------|:-------------------------|
        | **Lote** | `lote` | lote, block, bloque, codigo_lo te, cod_lote, id_lote, lote_id |
        | **Finca** | `finca` | finca, fazenda, farm, hacienda |
        | **Edad** | `edad` | edad, age, anos, ano_planta, idade |
        | **Producción (ton/ha)** | `ton_ha` | ton_ha, ton/ha, t_ha, rff, cff, produtividade, productividad, produccion_ha |
        | **Variedad** | `variedad` | variedad, variety, cultivar, tipo_variedad |
        | **Material** | `material` | material, material_genetico, genetica, origen_material |
        | **Indicador 0G / 1H** | `0g_1h` | 0g_1h, og_1h, g_h, tipo_material, flag_0g_1h, indicador_0g_1h |
        | **Nitrógeno foliar** | `fol_n` | n_f, n_fol, fol_n, foliar_n |
        | **Fósforo foliar** | `fol_p` | p_f, p_fol, fol_p, foliar_p |
        | **Potasio foliar** | `fol_k` | k_f, k_fol, fol_k, foliar_k |
        | **Calcio foliar** | `fol_ca` | ca_f, ca_fol, fol_ca, foliar_ca |
        | **Magnesio foliar** | `fol_mg` | mg_f, mg_fol, fol_mg, foliar_mg |
        | **Boro foliar** | `fol_b` | b_f, b_fol, fol_b, foliar_b |
        | **Azufre foliar** | `fol_s` | s_f, s_fol, fol_s, foliar_s |
        | **Cobre foliar** | `fol_cu` | cu_f, cu_fol, fol_cu, foliar_cu |
        | **Hierro foliar** | `fol_fe` | fe_f, fe_fol, fol_fe, foliar_fe |
        | **Manganeso foliar** | `fol_mn` | mn_f, mn_fol, fol_mn, foliar_mn |
        | **Zinc foliar** | `fol_zn` | zn_f, zn_fol, fol_zn, foliar_zn |
        """)

        st.caption(
            "El sistema normaliza automáticamente mayúsculas, minúsculas, "
            "acentos, espacios y caracteres especiales."
        )
        st.markdown("---")
    return uploaded

# ════════════════════════════════════════════════════
# 8. MAIN
# ════════════════════════════════════════════════════


def main():
    uploaded = render_sidebar()

    st.markdown("""
    <div class="main-header">
        <h1>Calculadora de Fertilización  — Presupuesto — FarmPrecision</h1>
        <p>Motor de cálculo basado en ecuaciones de regresión AGROPALMA · Guineensis & Híbrido</p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div class="upload-zone">
            <h3>📂 Sube tu archivo Excel o CSV para comenzar</h3>
            <p><strong>Formatos soportados:</strong> .xlsx · .xls · .csv</p>
            <p><strong>Columnas esperadas:</strong> Lote · Finca · Edad · ton/ha · 0G_1H · N_f · P_f · K_f · Ca_f · Mg_f · B_f</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📌 ¿Qué puedes analizar?")
        c1, c2, c3, c4 = st.columns(4)
        c1.info("**Estado foliar** (N, P, K, Ca, Mg, S, B, Cu, Fe, Mn, Zn) frente a rangos óptimos por lote")
        c2.info("**Recomendación de fertilización** (kg/ha) ajustada por déficit foliar y especie (Guineensis / Híbrido OxG)")
        c3.info("**Mapa de calor** de necesidad nutricional por lote × nutriente")
        c4.info("**Exportación** de resultados y resumen estadístico por nutriente en CSV / Excel")
        return

    with st.spinner("⏳ Procesando datos y calculando requerimientos..."):
        try:
            file_bytes = uploaded.read()
            cargar_niveles_optimos(file_bytes)
            data = cargar_dataset(file_bytes, file_name=uploaded.name)
            data = calculadora_fert(data)
        except Exception as e:
            st.error(f"❌ Error al procesar archivo: {e}")
            st.exception(e)
            return

    if data.empty:
        st.warning("El archivo no contiene datos válidos.")
        return

    finca_col = find_col(data.columns, ["finca"])
    departamento_col = find_col(data.columns, ["departamento", "depto", "region"])
    material_col = find_col(data.columns, ["material", "material_genetico"])
    edad_col = find_col(data.columns, ["edad", "age"])

    with st.sidebar:
        st.markdown("### 🔎 Filtros")
        if finca_col:
            fincas = sorted(data[finca_col].dropna().unique().tolist())
            sel_finca = st.multiselect("Finca:", fincas, default=fincas, key="main_finca")
        else:
            sel_finca = []

        if departamento_col:
            deps = sorted(data[departamento_col].dropna().unique().astype(str).tolist())
            sel_departamento = st.multiselect("Departamento:", deps, default=deps, key="main_departamento")
        else:
            sel_departamento = []

        if material_col:
            mats = sorted(data[material_col].dropna().unique().astype(str).tolist())
            sel_material = st.multiselect("Material:", mats, default=mats, key="main_material")
        else:
            sel_material = []

        if edad_col:
            vals = data[edad_col].dropna().apply(lambda x: to_float_safe(x, np.nan))
            if len(vals) > 0:
                mn = int(vals.min())
                mx = int(vals.max())
            else:
                mn, mx = 0, 30
            sel_edad = st.slider("Edad (años):", min_value=mn, max_value=mx, value=(mn, mx), key="main_edad")
        else:
            sel_edad = None

        st.markdown("---")
        st.caption("Los filtros se aplican al dataset procesado.")

    df = data.copy()
    if sel_finca and finca_col:
        df = df[df[finca_col].isin(sel_finca)]
    if sel_departamento and departamento_col:
        df = df[df[departamento_col].astype(str).isin(sel_departamento)]
    if sel_material and material_col:
        df = df[df[material_col].astype(str).isin(sel_material)]
    if sel_edad and edad_col:
        df = df[df[edad_col].isna() | ((df[edad_col] >= sel_edad[0]) & (df[edad_col] <= sel_edad[1]))]

    if df.empty:
        st.warning("Sin datos con los filtros actuales.")
        return

    seccion_kpis(df)

    st.markdown("---")

    tab_names = ["🔍 Información",
                 "📌 Resumen",
                 "🧮 Agrupaciones",
                 "💾 Exportar",
                 ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_info()
    with tabs[1]:
        tab_resumen(df)
    with tabs[2]:
        tab_agrupaciones(df)
    with tabs[3]:
        tab_exportar(df)


if __name__ == "__main__":
    main()