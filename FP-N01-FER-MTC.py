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

FOLIAR_RANGES = {
    "N":  {"min": 2.50,   "max": 2.80,   "unidad": "g/kg"},
    "P":  {"min": 0.16,   "max": 0.19,   "unidad": "g/kg"},
    "K":  {"min": 1.25,   "max": 1.45,   "unidad": "g/kg"},
    "Ca": {"min": 0.55,   "max": 0.75,   "unidad": "g/kg"},
    "Mg": {"min": 0.25,   "max": 0.40,   "unidad": "g/kg"},
    "S":  {"min": 0.20,   "max": 0.25,   "unidad": "g/kg"},
    "B":  {"min": 0.0020, "max": 0.0045, "unidad": "ppm"},
    "Cu": {"min": 0.0005, "max": 0.0007, "unidad": "ppm"},
    "Fe": {"min": 0.0080, "max": 0.0160, "unidad": "ppm"},
    "Mn": {"min": 0.0060, "max": 0.0200, "unidad": "ppm"},
    "Zn": {"min": 0.0015, "max": 0.0040, "unidad": "ppm"},
}

EXPORT_COEF = {
    "S": 0.5, "Cu": 0.004, "Fe": 0.03, "Mn": 0.025, "Zn": 0.01,
}

EFICIENCIA = {
    "S": 0.40, "Cu": 1.00, "Fe": 1.00, "Mn": 1.00, "Zn": 1.00,
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

ELEMENTOS_OFICIALES = {"N", "P", "K", "Ca", "Mg", "B"}

ELEMENTOS_CALCULO = ["N", "P", "K", "Ca", "Mg", "S", "B", "Cu", "Fe", "Mn", "Zn"]

DB_COEF_GUINEENSIS = {
    "N":  (5.8305,   -0.1165),
    "P":  (4.3580,   -0.1892),
    "K":  (10.4364,  -0.1356),
    "Ca": (3.4953,   -0.3007),
    "Mg": (4.3580,   -0.1892),
    "B":  (0.013141, -0.001582),
}

DB_COEF_HIBRIDO = {
    "N":  (6.3014,   0.2818),
    "P":  (4.7183,  -0.0361),
    "K":  (11.3221, -0.3537),
    "Ca": (3.7766,  -0.1794),
    "Mg": (4.7183,  -0.0361),
    "B":  (0.014119, -0.000262),
}

PROD_COEF_GUINEENSIS = (-0.1283, 4.0247, -3.9629)
PROD_COEF_HIBRIDO = (-0.1450, 4.5496, -4.4798)
PROD_MAX_MAYOR_26 = 14.0
PROD_DEFAULT = 20.0

# ════════════════════════════════════════════════════
# 1.b MODELO DE KALINI
# ════════════════════════════════════════════════════

FUENTES_KALINI = {
    "N":  {"fuente": "Urea",          "aporte": 0.45},
    "P":  {"fuente": "SPT",           "aporte": 0.46},
    "K":  {"fuente": "KCl",           "aporte": 0.60},
    "Ca": {"fuente": "Cal_dolomita",  "aporte": 0.35},
    "Mg": {"fuente": "Kieserita",     "aporte": 0.25},
    "B":  {"fuente": "Granubor",      "aporte": 0.15},
}

# FIX 2: minúscula para coincidir con FUENTES_KALINI[e]["fuente"].lower()
# La cal dolomita es enmienda/correctivo de pH y NO entra en el compuesto.
# Si el equipo decide incluirla, cambiar a set() vacío.
FUENTES_EXCLUIDAS_FORMULA = {"cal_dolomita"}

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
    """
    Crea/normaliza columnas 'Rec_*' empleadas por todos los tabs (Resumen, Agrupaciones, Export).
    Usa las mismas reglas de tab_exportar pero sin generar el Excel.
    Idempotente: puede ejecutarse varias veces sin duplicar o romper datos.
    """
    df = df.copy()

    area_col = _safe_find(df, ["area", "ha", "hectareas", "superficie"])
    n_palmas_col = _safe_find(df, ["n_palmas", "numero_palmas", "palmas", "plantas", "plantas_ha"])
    area_s = df[area_col].apply(lambda x: to_float_safe(x, np.nan)) if area_col in df.columns else pd.Series(np.nan, index=df.index)
    n_palmas_s = df[n_palmas_col].apply(lambda x: to_float_safe(x, np.nan)) if n_palmas_col in df.columns else pd.Series(np.nan, index=df.index)

    for e in ["N","P","K","Ca","Mg","B","S"]:
        # buscar posibles columnas existentes
        cand_ha = find_col(df.columns, [f"da_{e.lower()}_kg_ha", f"nec_{e.lower()}_kg_ha", f"da_{e} (kg/ha)", f"nec_{e} (kg/ha)"])
        cand_lote = find_col(df.columns, [f"da_{e.lower()}_kg_lote", f"nec_{e.lower()}_kg_lote", f"da_{e}_lote", f"{e.lower()}_lote"])
        cand_gpalma = find_col(df.columns, [f"da_{e.lower()}_g_palma", f"{e.lower()}_g_palma", f"da_{e}/palma (g)"])

        if cand_ha and cand_ha in df.columns:
            df[f"Rec_da_{e}_kgha"] = pd.to_numeric(df[cand_ha], errors="coerce")
        if cand_lote and cand_lote in df.columns:
            df[f"Rec_da_{e}_kglote"] = pd.to_numeric(df[cand_lote], errors="coerce")
        elif cand_ha and area_col in df.columns:
            df[f"Rec_da_{e}_kglote"] = pd.to_numeric(df[cand_ha], errors="coerce") * area_s
        if cand_gpalma and cand_gpalma in df.columns:
            df[f"Rec_da_{e}_gpalma"] = pd.to_numeric(df[cand_gpalma], errors="coerce")
        elif f"Rec_da_{e}_kglote" in df.columns and n_palmas_col in df.columns:
            df[f"Rec_da_{e}_gpalma"] = _to_g_palma_from_kg_lote(df[f"Rec_da_{e}_kglote"], n_palmas_s)

    gpal_cols = [c for c in df.columns if c.startswith("Rec_da_") and c.endswith("_gpalma")]
    if gpal_cols:
        df["Rec_total_n_gpalma"] = df[gpal_cols].sum(axis=1, min_count=1)

    fuente_map_keys = {
        "urea": ["fuente_urea_kg_ha", "urea_kg_ha", "urea (kg/ha)"],
        "spt": ["fuente_spt_kg_ha", "spt_kg_ha", "spt (kg/ha)"],
        "kcl": ["fuente_kcl_kg_ha", "kcl_kg_ha", "kcl (kg/ha)"],
        "kieserita": ["fuente_kieserita_kg_ha", "kieserita_kg_ha"],
        "granubor": ["fuente_granubor_kg_ha", "granubor_kg_ha"],
        "cal_dolomita": ["fuente_cal_dolomita_kg_ha", "cal_dolomita_kg_ha", "cal dolomita (kg/ha)"],
    }
    for key, candidates in fuente_map_keys.items():
        found = None
        for cand in candidates:
            c = find_col(df.columns, [cand])
            if c:
                found = c
                break
        if found:
            df[f"Rec_{key}_kgha"] = pd.to_numeric(df[found], errors="coerce")
            # kglote
            if area_col in df.columns:
                df[f"Rec_{key}_kglote"] = df[f"Rec_{key}_kgha"] * area_s
            # gpalma
            if n_palmas_col in df.columns and f"Rec_{key}_kglote" in df.columns:
                df[f"Rec_{key}_gpalma"] = _to_g_palma_from_kg_lote(df[f"Rec_{key}_kglote"], n_palmas_s)

    all_rec_gpalma = [c for c in df.columns if c.startswith("Rec_") and c.endswith("_gpalma")]
    if all_rec_gpalma:
        df["Rec_total_f_gpalma"] = pd.to_numeric(df[all_rec_gpalma].sum(axis=1, min_count=1), errors="coerce")

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


def find_col(df_cols, candidates):
    norm_map = {normalize_col(c): c for c in df_cols}
    for cand in candidates:
        nc = normalize_col(cand)
        if nc in norm_map:
            return norm_map[nc]
    return None


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


def get_foliar_status(val_pct, elem):
    r = FOLIAR_RANGES.get(elem)
    if not r or pd.isna(val_pct):
        return "sin_dato", "—"
    mn, mx = r["min"], r["max"]
    if val_pct < mn * 0.80:
        return "critico", "🔴 Muy Bajo"
    if val_pct < mn:
        return "bajo", "🟡 Bajo"
    if val_pct <= mx:
        return "optimo", "🟢 Óptimo"
    return "alto", "🔵 Alto"


def calc_fator_reajuste(val_pct, elem):
    if elem not in FOLIAR_RANGES or pd.isna(val_pct):
        return 1.0
    mn, mx = FOLIAR_RANGES[elem]["min"], FOLIAR_RANGES[elem]["max"]
    if val_pct < mn * 0.80:
        return 1.4
    if val_pct < mn:
        return 1.2
    if val_pct > mx:
        return 0.5
    return 1.0


def calc_demanda_bruta(elem, rff, flag_0g_1h):
    rff = to_float_safe(rff, default=np.nan)
    if pd.isna(rff):
        return 0.0
    if elem in ELEMENTOS_OFICIALES:
        coefs = DB_COEF_GUINEENSIS if int(flag_0g_1h) == 0 else DB_COEF_HIBRIDO
        a, b = coefs[elem]
        return max(0.0, a * rff + b)
    return max(0.0, EXPORT_COEF.get(elem, 0.0) * rff)


def calc_demanda_ajustada(elem, val_pct, demanda_bruta):
    if pd.isna(val_pct) or elem not in FOLIAR_RANGES:
        return max(0.0, demanda_bruta)
    mn = FOLIAR_RANGES[elem]["min"]
    mx = FOLIAR_RANGES[elem]["max"]
    delta = DELTA.get(elem, np.nan)
    g_planta = G_PLANTA.get(elem, np.nan)
    if pd.isna(delta) or pd.isna(g_planta) or delta <= 0:
        return max(0.0, demanda_bruta)
    if val_pct < mn:
        correccion = ((mn - val_pct) / delta) * g_planta * N_PALMAS / 1000
        return max(0.0, demanda_bruta + correccion)
    if val_pct > mx:
        return max(0.0, demanda_bruta * 0.5)
    return max(0.0, demanda_bruta)


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

    if not pd.isna(edad) and edad > 26:
        if rff_col and rff_col in row.index:
            rff_real = to_float_safe(row.get(rff_col, np.nan), default=np.nan)
            if not pd.isna(rff_real) and rff_real > 0:
                return round(min(rff_real, PROD_MAX_MAYOR_26), 3), "dato_real_mayor_26"
        rff_techo = calcular_produccion_modelada(edad, especie=especie)
        if not pd.isna(rff_techo) and rff_techo > 0:
            return rff_techo, "techo_mayor_26"

    if not pd.isna(edad):
        rff_modelado = calcular_produccion_modelada(edad, especie=especie)
        if not pd.isna(rff_modelado) and rff_modelado > 0:
            return rff_modelado, "curva_variedad_edad"

    if rff_col and rff_col in row.index:
        rff = to_float_safe(row.get(rff_col, np.nan), default=np.nan)
        if not pd.isna(rff) and rff > 0:
            return rff, "dato_real_respaldo"

    return PROD_DEFAULT, "respaldo"


def descomponer_fuentes_kalini(df, area_serie, n_palmas_serie):

    df = df.copy()

    ha_cols, lote_cols, gpalma_cols = [], [], []

    for elem, meta in FUENTES_KALINI.items():
        fuente_nombre = meta["fuente"]
        apor = to_float_safe(meta["aporte"], default=np.nan)
        da_ha_col = f"da_{elem.lower()}_kg_ha"

        if da_ha_col not in df.columns or pd.isna(apor) or apor <= 0:
            continue

        nombre_key = fuente_nombre.lower()

        ha_c = f"fuente_{nombre_key}_kg_ha"
        lo_c = f"fuente_{nombre_key}_kg_lote"
        gp_c = f"fuente_{nombre_key}_g_palma"

        df[ha_c] = (df[da_ha_col] / apor).round(6)
        df[lo_c] = (df[ha_c] * area_serie).round(6)
        df[gp_c] = (df[lo_c] / n_palmas_serie * 1000.0).round(6)

        ha_cols.append(ha_c)
        lote_cols.append(lo_c)
        gpalma_cols.append(gp_c)

    if gpalma_cols:
        df["total_fuentes_g_palma"] = (
            df[gpalma_cols].sum(axis=1, min_count=1).round(6)
        )
    else:
        df["total_fuentes_g_palma"] = np.nan

    compuesto_ha_cols = [
        c for c, key in zip(
            ha_cols,
            [FUENTES_KALINI[e]["fuente"].lower() for e in FUENTES_KALINI]
        )
        if key not in FUENTES_EXCLUIDAS_FORMULA
    ]
    if compuesto_ha_cols:
        df["total_compuesto_kg_ha"] = (
            df[compuesto_ha_cols].sum(axis=1, min_count=1).round(6)
        )
    else:
        df["total_compuesto_kg_ha"] = np.nan

    def formula_row(r):
        idx = {e: f"da_{e.lower()}_kg_ha"
               for e in ("N", "P", "K", "Mg", "B")}
        tot_c = r.get("total_compuesto_kg_ha", np.nan)
        if pd.isna(tot_c) or tot_c <= 0:
            return np.nan
        def pct(e):
            v = r.get(idx[e], np.nan)
            if pd.isna(v):
                return 0
            return round(v / tot_c * 100)
        return f"{pct('N')}-{pct('P')}-{pct('K')}-{pct('Mg')} MgO-{pct('B')} B"

    df["Formula_Kalini"] = df.apply(formula_row, axis=1)

    return df


def calculadora_fert(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    edad_col = find_col(
        df.columns,
        ["edad", "age", "anos", "ano_planta", "idade", "idade_ano"],
    )

    rff_col = find_col(
        df.columns,
        [
            "ton_ha", "ton/ha", "ton ha", "t_ha", "rff", "cff",
            "produtividade", "productividad", "produccion_ha",
            "estimativa_cff_t_ha",
        ],
    )

    variedad_col = find_col(
        df.columns,
        ["variedad", "variety", "cultivar", "tipo_variedad"],
    )

    material_col = find_col(
        df.columns,
        ["material", "material_genetico", "genetica", "origen_material"],
    )

    flag_col = find_col(
        df.columns,
        [
            "0g_1h", "og_1h", "g_h", "flag_0g_1h",
            "tipo_material", "indicador_0g_1h",
        ],
    )

    area_col = find_col(
        df.columns,
        ["area", "ha", "hectareas", "superficie"],
    )

    n_palmas_col = find_col(
        df.columns,
        ["n_palmas", "numero_palmas", "palmas", "plantas", "plantas_ha"],
    )

    rff_res = []
    fuente_res = []
    especie_res = []
    flag_res = []
    prod_obs_res = []
    prod_mod_res = []

    for _, row in df.iterrows():
        variedad = row.get(variedad_col, "") if variedad_col is not None else ""
        material = row.get(material_col, "") if material_col is not None else ""

        flag = get_flag_0g_1h(
            row, flag_col=flag_col,
            variedad_col=variedad_col, material_col=material_col,
        )

        especie = determinar_especie(
            variedad=variedad, material=material, indicador_0g_1h=flag,
        )

        edad_val = row.get(edad_col, np.nan) if edad_col is not None else np.nan
        prod_modelada = calcular_produccion_modelada(edad_val, especie=especie)
        prod_observada = row.get(rff_col, np.nan) if rff_col is not None else np.nan
        prod_observada = to_float_safe(prod_observada, default=np.nan)

        rff, fuente = obtener_rff_calculo(
            row, rff_col=rff_col, edad_col=edad_col, especie=especie,
        )

        rff_res.append(round(rff, 3))
        fuente_res.append(fuente)
        especie_res.append(especie)
        flag_res.append(flag)
        prod_obs_res.append(
            round(prod_observada, 3) if not pd.isna(prod_observada) else np.nan
        )
        prod_mod_res.append(
            round(prod_modelada, 3) if not pd.isna(prod_modelada) else np.nan
        )

    df["ton_ha_observada"] = prod_obs_res
    df["ton_ha_modelada"] = prod_mod_res
    df["rff_calculo"] = rff_res
    df["fuente_rff"] = fuente_res
    df["especie"] = especie_res

    for elem in ELEMENTOS_CALCULO:
        fol_col = find_col(
            df.columns,
            [
                f"fol_{elem.lower()}", f"{elem}_f", f"{elem.lower()}_f",
                f"{elem}_fol", f"{elem.lower()}_fol", f"foliar_{elem.lower()}",
            ],
        )

        fol_res, db_res, da_res, rec_res, fator_res, status_res = [], [], [], [], [], []

        for _, row in df.iterrows():
            rff = to_float_safe(row.get("rff_calculo", np.nan), default=np.nan)
            flag = to_float_safe(row.get("flag_0g_1h", 0), default=0)

            val_orig = (
                row.get(fol_col, np.nan)
                if fol_col is not None and fol_col in df.columns
                else np.nan
            )

            val_pct = foliar_a_pct(val_orig, elem)
            demanda_bruta = calc_demanda_bruta(elem=elem, rff=rff, flag_0g_1h=flag)
            demanda_ajustada = calc_demanda_ajustada(
                elem=elem, val_pct=val_pct, demanda_bruta=demanda_bruta
            )
            recomendacion_final = calc_recomendacion_final(
                demanda_ajustada=demanda_ajustada, elem=elem
            )
            fator = calc_fator_reajuste(val_pct=val_pct, elem=elem)
            status, _ = get_foliar_status(val_pct=val_pct, elem=elem)

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

    if area_col is not None and area_col in df.columns:
        area_serie = df[area_col].apply(lambda x: to_float_safe(x, default=np.nan))
    else:
        area_serie = pd.Series(np.nan, index=df.index, dtype="float64")

    if n_palmas_col is not None and n_palmas_col in df.columns:
        n_palmas_serie = df[n_palmas_col].apply(
            lambda x: to_float_safe(x, default=np.nan)
        )
    else:
        n_palmas_serie = pd.Series(np.nan, index=df.index, dtype="float64")

    area_serie = area_serie.replace([np.inf, -np.inf, 0], np.nan)
    n_palmas_serie = n_palmas_serie.replace([np.inf, -np.inf, 0], np.nan)

    for elem in ELEMENTOS_CALCULO:
        nec_ha_col = f"nec_{elem.lower()}_kg_ha"
        if nec_ha_col not in df.columns:
            continue
        df[f"nec_{elem.lower()}_kg_lote"] = (df[nec_ha_col] * area_serie).round(6)

    elementos_oficiales_ordenados = [
        elem for elem in ["N", "P", "K", "Ca", "Mg", "B"] if elem in ELEMENTOS_OFICIALES
    ]

    g_palma_cols = []

    for elem in elementos_oficiales_ordenados:
        da_ha_col = f"da_{elem.lower()}_kg_ha"
        if da_ha_col not in df.columns:
            continue
        da_lote = df[da_ha_col] * area_serie
        df[f"da_{elem.lower()}_kg_lote"] = da_lote.round(6)
        da_g_palma = da_lote / n_palmas_serie * 1000.0
        g_palma_col = f"da_{elem.lower()}_g_palma"
        df[g_palma_col] = da_g_palma.round(6)
        g_palma_cols.append(g_palma_col)

    if g_palma_cols:
        df["total_g_palma"] = (
            df[g_palma_cols].sum(axis=1, min_count=1).round(6)
        )
    else:
        df["total_g_palma"] = np.nan

    df = descomponer_fuentes_kalini(df, area_serie, n_palmas_serie)
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
         "La demanda se compara contra el rango foliar objetivo. Si está bajo, "
         "se corrige; si está alto, se reduce.", "#f39c12"),
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

    n_inferidos = 0
    n_total = len(df)
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

    # Mapa de calor: preferimos Rec_da_*_kgha
    rec_cols = [c for c in df.columns if c.startswith("Rec_da_") and c.endswith("_kgha")]
    lote_col = find_col(df.columns, ["lote", "Lote", "block", "id_lote"])

    st.markdown(
        '<div class="section-title">Mapa de Calor — Recomendación por Lote × Nutriente</div>',
        unsafe_allow_html=True
    )

    if rec_cols and lote_col:
        df_heat = df[[lote_col] + rec_cols].copy()
        # normalizar nombres columnas a sólo nutriente en mayúscula
        df_heat = df_heat.set_index(lote_col)
        df_heat.columns = [re.sub(r"Rec_da_(.+?)_kgha", r"\1", c).str.upper() if False else re.sub(r"Rec_da_(.+?)_kgha", lambda m: m.group(1).upper(), c) for c in df_heat.columns]
        # la línea anterior usa re; si hay problemas, hacemos simple replace:
        # df_heat.columns = [c.replace("Rec_da_", "").replace("_kgha","").upper() for c in df_heat.columns]
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

        # Fórmulas compuestas: chequear ambos nombres
        formula_col = find_col(df.columns, ["Formula_Kalini", "Formula (N-P-K-MgO-B)", "formula_kalini", "formula"])
        if formula_col and formula_col in df.columns:
            st.markdown(
                '<div class="section-title">Fórmulas compuestas más frecuentes</div>',
                unsafe_allow_html=True
            )
            top_formulas = df[formula_col].dropna().value_counts().head(8).reset_index()
            top_formulas.columns = ["Fórmula (N-P-K-MgO-B)", "Lotes"]
            if len(top_formulas) > 0:
                cols = st.columns(min(len(top_formulas), 6))
                for col, (_, r) in zip(cols, top_formulas.iterrows()):
                    col.markdown(
                        f'<div class="formula-badge">{r["Fórmula (N-P-K-MgO-B)"]}</div>'
                        f'<div style="font-size:0.75rem;color:#7A8899;margin-left:0.3rem;">'
                        f'{int(r["Lotes"])} lotes</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("No hay fórmulas compuestas para mostrar.")

    # Mensaje sobre inferidos (mantener comportamiento original)
    if n_inferidos > 0:
        st.warning(
            f"⚠️ **{n_inferidos} de {n_total} lotes ({pct:.1f}%)** no tenían "
            f"el flag `0G_1H` en el dataset original y se les asumió "
            f"**Guineensis** por defecto (inferido por Variedad/Material). "
            f"Revisar con campo si corresponde."
        )

    st.markdown("---")


def tab_agrupaciones(df: pd.DataFrame):
    """
    Agrupaciones priorizando columnas 'Rec_*' generadas por tab_exportar_v2.
    UI:
    - elegir variable categórica (leniente)
    - elegir tipo de variable numérica: kgha, kglote, gpalma, todas
    - opción: incluir fuentes comerciales (Rec_* que no sean da_)
    - elegir variable numérica de la lista resultante
    - exportar CSV
    """
    st.markdown('<div class="section-title">Agrupaciones — Estadísticas por grupo</div>', unsafe_allow_html=True)

    cat_candidates = []
    for c in ["finca", "departamento", "manejo", "material", "variedad", "lote"]:
        col = find_col(df.columns, [c])
        if col:
            cat_candidates.append(col)
    if not cat_candidates:
        st.info("No se detectaron columnas categóricas (finca, departamento, manejo, material, variedad, lote).")
        return

    unidad_opt = st.radio("Tipo de unidad a listar", options=["kgha", "kglote", "gpalma", "todas"], index=0, horizontal=True, key="agr_unidad_opt")
    incluir_fuentes = st.checkbox("Incluir fuentes comerciales (Rec_* sin da_)", value=True, key="agr_incluir_fuentes")

    rec_cols = [c for c in df.columns if c.startswith("Rec_")]
    if not rec_cols:
        rec_cols = [c for c in df.columns if c.lower().startswith("da_") or c.lower().startswith("nec_") or c.lower().startswith("total_")]

    def _match_unidad(col, unidad):
        cl = col.lower()
        if unidad == "kgha":
            return cl.endswith("_kgha") or "_kgha" in cl
        if unidad == "kglote":
            return cl.endswith("_kglote") or "_kglote" in cl
        if unidad == "gpalma":
            return cl.endswith("_gpalma") or "_gpalma" in cl or "_g_palma" in cl
        return True

    candidate_num = []
    for c in rec_cols:
        if unidad_opt == "todas" or _match_unidad(c, unidad_opt):
            if not incluir_fuentes and re.match(r"Rec_((urea|spt|kcl|kieserita|granubor|cal|caldol).*)", c, re.I):
                continue
            candidate_num.append(c)

    if not candidate_num:
        for c in df.columns:
            cl = c.lower()
            if unidad_opt == "kgha" and (cl.endswith("_kgha") or cl.endswith("_kg_ha")):
                candidate_num.append(c)
            if unidad_opt == "kglote" and (cl.endswith("_kglote") or cl.endswith("_kg_lote")):
                candidate_num.append(c)
            if unidad_opt == "gpalma" and ("gpalma" in cl or "_g_palma" in cl):
                candidate_num.append(c)
        candidate_num = list(dict.fromkeys(candidate_num))  # unique preserve order

    if not candidate_num:
        st.info("No se encontraron columnas numéricas tras aplicar filtros. Elige 'todas' o revisa el dataset.")
        return

    with st.expander("Configuración de agrupación", expanded=True):
        agrup_col = st.selectbox("Selecciona variable categórica para agrupar:", options=cat_candidates, index=0, key="agr_col")
        var_num = st.selectbox("Selecciona variable numérica:", options=candidate_num, index=0, key="agr_var")
        top_n = st.number_input("Mostrar top N grupos (0 = todos):", min_value=0, max_value=1000, value=0, step=10, key="agr_top_n")

    serie_num = pd.to_numeric(df[var_num], errors="coerce")
    trabajo = df[[agrup_col]].copy()
    trabajo[var_num] = serie_num

    try:
        agg = trabajo.groupby(agrup_col, dropna=False)[var_num].agg(['count', 'sum', 'mean', 'min', 'max'])
    except Exception as e:
        st.error(f"Error al agrupar: {e}")
        return

    decimals = 4 if ("gpalma" in var_num.lower() or "_g_palma" in var_num.lower()) else 2
    agg['sum'] = agg['sum'].round(decimals)
    agg['mean'] = agg['mean'].round(decimals)
    agg['min'] = agg['min'].round(decimals)
    agg['max'] = agg['max'].round(decimals)
    agg = agg.reset_index()

    order_col = 'sum' if agg['sum'].notna().sum() > 0 and agg['sum'].abs().sum() > 0 else 'count'
    agg = agg.sort_values(by=order_col, ascending=False)

    view = agg.head(top_n) if top_n > 0 else agg

    st.markdown("#### Resultado (tabla única)")
    st.dataframe(view, use_container_width=True)

    csv_bytes = agg.to_csv(index=False).encode("utf-8-sig")
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_var = re.sub(r'[^0-9A-Za-z_]+', '', var_num)
    st.download_button("📥 Descargar agrupaciones (CSV)", data=csv_bytes, file_name=f"agrupaciones_{safe_var}_{now}.csv", mime="text/csv")

    st.markdown("---")

def _to_g_palma_from_kg_lote(kg_lote_series, n_palmas_series):
    """Convierte kg/lote -> g/palma de forma robusta (maneja NaN, ceros, inf)."""
    s = kg_lote_series.copy().astype(float)
    n = n_palmas_series.copy().astype(float)
    n = n.replace([0, np.inf, -np.inf], np.nan)
    g = (s * 1000.0) / n
    return g

def _to_kg_palma_from_g_palma(g_palma_series):
    """Convierte g/palma -> kg/palma."""
    return (g_palma_series / 1000.0)

def _round_cols(df, decimals_default=2, decimals_gpalma=4):
    """Aplica redondeo: por defecto 2 dec, pero columnas *_gpalma -> 4 dec."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype in [np.float64, np.float32, float]:
            if c.lower().endswith("_gpalma"):
                df[c] = df[c].round(decimals_gpalma)
            else:
                df[c] = df[c].round(decimals_default)
    return df

def _safe_find(df, candidates):
    """Busca primer nombre de columna presente en df entre candidatos (leniente)."""
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

def tab_exportar(df: pd.DataFrame, nombre_excel_salida: str = None):
    """
    Exportador estandarizado según plantilla 'Rec_...'.
    - Crea/normaliza columnas Rec_*
    - Aplica redondeo: 2 dec por defecto, 4 dec para *_gpalma
    - Genera Excel con hojas: Resultados, Evaluacion_Agronomica, Evaluacion_Administrativa, Agrupaciones, Resumen_Finca
    """
    df = df.copy()

    lote_col = _safe_find(df, ["lote", "block", "cod_lote", "id_lote"])
    finca_col = _safe_find(df, ["finca", "farm", "hacienda"])
    departamento_col = _safe_find(df, ["departamento", "depto", "region", "zona"])
    edad_col = _safe_find(df, ["edad", "age", "anos"])
    area_col = _safe_find(df, ["area", "ha", "hectareas", "superficie"])
    n_palmas_col = _safe_find(df, ["n_palmas", "numero_palmas", "palmas", "plantas", "plantas_ha"])

    area_s = df[area_col].apply(lambda x: to_float_safe(x, np.nan)) if area_col in df.columns else pd.Series(np.nan, index=df.index)
    n_palmas_s = df[n_palmas_col].apply(lambda x: to_float_safe(x, np.nan)) if n_palmas_col in df.columns else pd.Series(np.nan, index=df.index)

    oficiales = [e for e in ["N","P","K","Ca","Mg","B","S"] if f"da_{e.lower()}_kg_ha" in df.columns or f"da_{e.lower()}_kg_ha" in [c.lower() for c in df.columns]]

    oficiales = [e for e in ["N","P","K","Ca","Mg","B","S"] if any(find_col(df.columns, [f"da_{e.lower()}_kg_ha", f"da_{e.lower()} (kg/ha)", f"da_{e}_ (kg/ha)"]) for e in [e])]

    for e in ["N","P","K","Ca","Mg","B","S"]:
        cand_ha = find_col(df.columns, [f"da_{e.lower()}_kg_ha", f"da_{e} (kg/ha)", f"da_{e.lower()} (kg/ha)"])
        cand_lote = find_col(df.columns, [f"da_{e.lower()}_kg_lote", f"da_{e}_lote (kg)", f"da_{e}_lote"])
        cand_gpalma = find_col(df.columns, [f"da_{e.lower()}_g_palma", f"da_{e}/palma (g)", f"da_{e}_/palma (g)", f"da_{e}/palma (g)".lower()])

        if cand_ha:
            df[f"Rec_da_{e}_kgha"] = df[cand_ha].apply(lambda x: to_float_safe(x, np.nan))
        if cand_lote:
            df[f"Rec_da_{e}_kglote"] = df[cand_lote].apply(lambda x: to_float_safe(x, np.nan))
        else:
            if cand_ha and area_col in df.columns:
                df[f"Rec_da_{e}_kglote"] = (df[cand_ha].apply(lambda x: to_float_safe(x, np.nan)) * area_s).replace([np.inf, -np.inf], np.nan)
        if cand_gpalma:
            df[f"Rec_da_{e}_gpalma"] = df[cand_gpalma].apply(lambda x: to_float_safe(x, np.nan))
        else:
            if f"Rec_da_{e}_kglote" in df.columns and n_palmas_col in df.columns:
                df[f"Rec_da_{e}_gpalma"] = _to_g_palma_from_kg_lote(df[f"Rec_da_{e}_kglote"], n_palmas_s)

    gpal_cols = [c for c in df.columns if c.startswith("Rec_da_") and c.endswith("_gpalma")]
    if gpal_cols:
        df["Rec_total_n_gpalma"] = df[gpal_cols].sum(axis=1, min_count=1)

    fuente_map = {
        "urea": ["fuente_urea_kg_ha", "urea (kg/ha)", "urea_ (kg/ha)"],
        "spt": ["fuente_spt_kg_ha", "spt (kg/ha)"],
        "kcl": ["fuente_kcl_kg_ha", "kcl (kg/ha)"],
        "kieserita": ["fuente_kieserita_kg_ha", "kieserita (kg/ha)"],
        "granubor": ["fuente_granubor_kg_ha", "granubor (kg/ha)"],
        "cal_dolomita": ["fuente_cal_dolomita_kg_ha", "cal dolomita (kg/ha)", "cal_dolomita (kg/ha)"],
    }
    for key, cands in fuente_map.items():
        found = None
        for cand in cands:
            if find_col(df.columns, [cand]):
                found = find_col(df.columns, [cand])
                break
        if found:
            df[f"Rec_{key}_kgha"] = df[found].apply(lambda x: to_float_safe(x, np.nan))
            df[f"Rec_{key}_kglote"] = None
            if n_palmas_col in df.columns and area_col in df.columns:
                cand_lote = find_col(df.columns, [found.replace(" (kg/ha)", "_lote (kg)"), found.replace("_kg_ha", "_kg_lote")])
                if cand_lote:
                    df[f"Rec_{key}_kglote"] = df[cand_lote].apply(lambda x: to_float_safe(x, np.nan))
                else:
                    df[f"Rec_{key}_kglote"] = df[f"Rec_{key}_kgha"] * area_s
            else:
                cand_lote_any = find_col(df.columns, [f"{key}_lote (kg)", f"{key}_lote"])
                if cand_lote_any:
                    df[f"Rec_{key}_kglote"] = df[cand_lote_any].apply(lambda x: to_float_safe(x, np.nan))

            if f"Rec_{key}_kglote" in df.columns and n_palmas_col in df.columns:
                df[f"Rec_{key}_gpalma"] = _to_g_palma_from_kg_lote(df[f"Rec_{key}_kglote"], n_palmas_s)

    fuente_g_cols = [c for c in df.columns if c.startswith("Rec_") and c.endswith("_gpalma") and any(k in c for k in ["urea","spt","kcl","kieserita","granubor","caldol","cal"])]
    all_rec_gpalma = [c for c in df.columns if c.startswith("Rec_") and c.endswith("_gpalma")]
    if all_rec_gpalma:
        df["Rec_total_f_gpalma"] = df[all_rec_gpalma].sum(axis=1, min_count=1)

    estado_cols = [c for c in df.columns if c.lower().startswith("estado_")]

    id_cols_candidates = [lote_col, finca_col, departamento_col, edad_col, "material", "variedad", "manejo", area_col, n_palmas_col]
    id_cols = [c for c in id_cols_candidates if c and c in df.columns]
    prod_cols = [c for c in ["ton_ha_observada", "ton_ha_modelada", "rff_calculo", "fuente_rff", "especie", "flag_0g_1h", "flag_inferido"] if c in df.columns]
    resultados_cols = id_cols + prod_cols + [c for c in df.columns if c.startswith("Rec_")] + estado_cols

    df_resultados = df[resultados_cols].copy()

    df_resultados = _round_cols(df_resultados, decimals_default=2, decimals_gpalma=4)

    eval_agro_cols = sorted([c for c in df_resultados.columns if c.startswith("Rec_da_") or c == "Rec_total_n_gpalma"])
    eval_agro = df_resultados[id_cols + eval_agro_cols].copy()

    eval_admin_cols = sorted([c for c in df_resultados.columns if (c.startswith("Rec_") and any(x in c for x in ["urea","spt","kcl","kieserita","granubor","caldol","cal"]) )] + ["Rec_total_f_gpalma"])

    eval_admin_cols = [c for c in eval_admin_cols if c in df_resultados.columns]
    eval_admin = df_resultados[id_cols + eval_admin_cols].copy() if eval_admin_cols else pd.DataFrame()

    agg_candidates = [finca_col, departamento_col, "material", "variedad", "manejo", lote_col]
    agg_by = next((c for c in agg_candidates if c and c in df.columns), None)
    if agg_by:
        agg_metrics = [c for c in df_resultados.columns if c.startswith("Rec_da_") and c.endswith("_kgha")]
        agg_metrics_g = [c for c in df_resultados.columns if c.startswith("Rec_da_") and c.endswith("_gpalma")]
        agg_cols = agg_metrics + agg_metrics_g
        if agg_cols:
            agrup = df_resultados.groupby(agg_by, dropna=False)[agg_cols].agg(['count','mean','sum','min','max'])
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
            resumen = df_resultados.groupby(finca_col, dropna=False)[resumen_metrics].agg(['count','sum','mean']).round(4)
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

    st.download_button("📥 Descargar Excel estandarizado", data=buffer.getvalue(), file_name=nombre_excel_salida, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    csv_bytes = df_resultados.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Descargar CSV Resultados", data=csv_bytes, file_name=f"fertilizacion_resultados_std_{now}.csv", mime="text/csv", use_container_width=True)

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
            data = cargar_dataset(uploaded.read(), file_name=uploaded.name)
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
        st.caption("Los filtros se aplican al dataset procesado (main).")

    df = data.copy()
    if sel_finca and finca_col:
        df = df[df[finca_col].isin(sel_finca)]
    if sel_departamento and departamento_col:
        df = df[df[departamento_col].astype(str).isin(sel_departamento)]
    if sel_material and material_col:
        df = df[df[material_col].astype(str).isin(sel_material)]
    if sel_edad and edad_col:
        df = df[(df[edad_col] >= sel_edad[0]) & (df[edad_col] <= sel_edad[1])]

    if df.empty:
        st.warning("Sin datos con los filtros actuales.")
        return

    seccion_kpis(df)
    st.markdown("---")

    tabs = st.tabs(["🔍 Información", "📌 Resumen", "🧮 Agrupaciones", "💾 Exportar"])
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