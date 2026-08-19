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
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# 1. CONSTANTES AGRONÓMICAS
# ════════════════════════════════════════════════════

# Rangos foliares en % (post-conversión: macros /10, micros /10000)
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

# Coeficientes de exportación — Cravo & Viégas 2000 (solo para S, Cu, Fe, Mn, Zn)
EXPORT_COEF = {
    "S": 0.5, "Cu": 0.004, "Fe": 0.03, "Mn": 0.025, "Zn": 0.01,
}

# Eficiencia de uso (solo aplica a S, Cu, Fe, Mn, Zn — los oficiales NO dividen)
EFICIENCIA = {
    "S": 0.40, "Cu": 1.00, "Fe": 1.00, "Mn": 1.00, "Zn": 1.00,
}

# Delta: incremento foliar (%) para subir 1 unidad de ajuste
DELTA = {
    "N": 0.1, "P": 0.1, "K": 0.50, "Ca": 0.01, "Mg": 0.01,
    "S": 0.01, "B": 0.0001, "Cu": 0.0001, "Fe": 0.0001, "Mn": 0.0001, "Zn": 0.0001,
}

# g/planta para subir 1 delta en el foliar
G_PLANTA = {
    "N": 70, "P": 84, "K": 70, "Ca": 105, "Mg": 56,
    "S": 42, "B": 2.1, "Cu": 1.4, "Fe": 7.0, "Mn": 3.5, "Zn": 2.1,
}

N_PALMAS = 143

MACRO_ELEMS = {"N", "P", "K", "Ca", "Mg", "S"}
MICRO_ELEMS = {"B", "Cu", "Fe", "Mn", "Zn"}

# Elementos con ecuación de regresión oficial (NO usan EXPORT_COEF)
ELEMENTOS_OFICIALES = {"N", "P", "K", "Ca", "Mg", "B"}

# Todos los elementos a calcular
ELEMENTOS_CALCULO = ["N", "P", "K", "Ca", "Mg", "S", "B", "Cu", "Fe", "Mn", "Zn"]

# Coeficientes de regresión db_* = a * ton/ha + b
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

# Curvas de producción por edad (AGROPALMA)
CURVA_PROD_BMP = {
    3: 6.739, 4: 9.499, 5: 12.19, 6: 14.812, 7: 17.48,
    8: 20.70, 9: 23.0,  10: 25.07, 11: 26.45, 12: 26.97,
    13: 26.97, 14: 26.97, 15: 26.96, 16: 26.68, 17: 26.61,
    18: 25.88, 19: 25.30, 20: 24.50, 21: 23.46, 22: 22.66,
    23: 20.70, 24: 19.21, 25: 17.83,
}

CURVA_PROD_SIN_BMP = {
    3: 5.86, 4: 8.26, 5: 10.6, 6: 12.88, 7: 15.2,
    8: 18.0, 9: 20.0, 10: 21.8, 11: 23.0, 12: 23.45,
    13: 23.45, 14: 23.45, 15: 23.44, 16: 23.2, 17: 23.14,
    18: 22.5, 19: 22.0, 20: 21.3, 21: 20.4, 22: 19.7,
    23: 18.0, 24: 16.7, 25: 15.5,
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

    text_cols = {"finca", "lote", "zona", "departamento", "variedad", "material"}
    for col in df.columns:
        if col in text_cols:
            df[col] = df[col].astype(str).str.strip().str.upper()
        else:
            df[col] = df[col].apply(lambda x: to_float_safe(x, default=np.nan))

    return df


# ════════════════════════════════════════════════════
# 3. MOTOR DE CÁLCULO — FÓRMULA OFICIAL EXCEL / FILA A FILA
# ════════════════════════════════════════════════════

def determinar_especie(variedad="", material="", indicador_0g_1h=np.nan):
    ind = to_float_safe(indicador_0g_1h, default=np.nan)
    if not pd.isna(ind):
        return "Guineensis" if int(ind) == 0 else "Hibrido_OxG"
    v = str(variedad).strip().lower()
    m = str(material).strip().lower()
    if any(x in v or x in m for x in ["hibrido", "híbrido", "oxg"]):
        return "Hibrido_OxG"
    if any(x in v or x in m for x in ["tenera", "dura", "pisifera", "guineensis", "deli", "nigeria", "ghana"]):
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
    """
    Conversión foliar a % según el Excel de referencia:
    - Todos los elementos del dataset (N_f, P_f, K_f, Ca_f, Mg_f, B_f) → /10
    - S, Cu, Fe, Mn, Zn (micros reales) → /10000
    """
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


def get_rff_esperado(edad, bmp=True):
    edad = to_float_safe(edad, default=np.nan)
    if pd.isna(edad):
        return 20.0
    edad = int(np.clip(edad, 3, 25))
    return CURVA_PROD_BMP.get(edad, 20.0) if bmp else CURVA_PROD_SIN_BMP.get(edad, 18.0)


def obtener_rff_calculo(row, rff_col=None, edad_col=None):
    if rff_col and rff_col in row.index:
        rff = to_float_safe(row.get(rff_col, np.nan), default=np.nan)
        if not pd.isna(rff) and rff > 0:
            return rff, "dato_real"
    if edad_col and edad_col in row.index:
        edad = to_float_safe(row.get(edad_col, np.nan), default=np.nan)
        if not pd.isna(edad):
            return get_rff_esperado(edad), "curva_edad"
    return 20.0, "respaldo"


def calculadora_fert(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    edad_col     = find_col(df.columns, ["edad", "age", "anos", "ano_planta", "idade"])
    rff_col      = find_col(df.columns, ["ton_ha", "ton/ha", "rff", "cff", "produtividade", "productividad"])
    variedad_col = find_col(df.columns, ["variedad", "variety", "cultivar"])
    material_col = find_col(df.columns, ["material", "material_genetico", "genetica"])
    flag_col     = find_col(df.columns, ["0g_1h", "og_1h", "flag_0g_1h", "tipo_material"])

    rff_res, fuente_res, especie_res, flag_res = [], [], [], []

    for _, row in df.iterrows():
        rff, fuente = obtener_rff_calculo(row, rff_col, edad_col)
        variedad = row.get(variedad_col, "") if variedad_col else ""
        material = row.get(material_col, "") if material_col else ""
        flag = get_flag_0g_1h(row, flag_col, variedad_col, material_col)
        especie = determinar_especie(variedad, material, flag)
        rff_res.append(round(rff, 3))
        fuente_res.append(fuente)
        especie_res.append(especie)
        flag_res.append(flag)

    df["rff_calculo"] = rff_res
    df["fuente_rff"]  = fuente_res
    df["especie"]     = especie_res
    df["flag_0g_1h"]  = flag_res

    for elem in ELEMENTOS_CALCULO:
        fol_col = find_col(
            df.columns,
            [f"fol_{elem.lower()}", f"{elem}_f", f"{elem.lower()}_f",
             f"{elem}_fol", f"{elem.lower()}_fol", f"foliar_{elem.lower()}"],
        )

        fol_res, db_res, da_res, rec_res, fator_res, status_res = [], [], [], [], [], []

        for _, row in df.iterrows():
            rff  = row["rff_calculo"]
            flag = row["flag_0g_1h"]

            val_orig = row.get(fol_col, np.nan) if fol_col and fol_col in df.columns else np.nan
            val_pct  = foliar_a_pct(val_orig, elem)

            db  = calc_demanda_bruta(elem, rff, flag)
            da  = calc_demanda_ajustada(elem, val_pct, db)
            rec = calc_recomendacion_final(da, elem)

            fator  = calc_fator_reajuste(val_pct, elem)
            status, _ = get_foliar_status(val_pct, elem)

            fol_res.append(round(val_pct, 8) if not pd.isna(val_pct) else np.nan)
            db_res.append(round(db, 6))
            da_res.append(round(da, 6))
            rec_res.append(round(rec, 6))
            fator_res.append(fator)
            status_res.append(status)

        df[f"fol_pct_{elem.lower()}"]    = fol_res
        df[f"db_{elem.lower()}_kg_ha"]   = db_res
        df[f"da_{elem.lower()}_kg_ha"]   = da_res
        df[f"nec_{elem.lower()}_kg_ha"]  = rec_res   # output final para exportar
        df[f"fator_{elem.lower()}"]      = fator_res
        df[f"status_{elem.lower()}"]     = status_res

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
    rff_col   = find_col(df.columns, ["ton_ha", "rff"])
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
                rendimiento, el material genético y el estado foliar del lote.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    pasos = [
        (
            "1",
            "Rendimiento y genética",
            "Se identifica la variedad o material del lote y se selecciona "
            "la ecuación correspondiente: Guineensis o Híbrido.",
            "#1b60a7",
        ),
        (
            "2",
            "Demanda bruta",
            "Se calcula la cantidad inicial de nutriente requerida según las "
            "toneladas de RFF por hectárea.",
            "#2ca02c",
        ),
        (
            "3",
            "Corrección foliar",
            "La demanda se compara contra el rango foliar objetivo. Si el "
            "nutriente está bajo, se corrige; si está alto, se reduce.",
            "#f39c12",
        ),
        (
            "4",
            "Recomendación final",
            "La demanda ajustada se convierte en la recomendación final "
            "considerando la eficiencia únicamente cuando corresponde.",
            "#8e44ad",
        ),
    ]

    cols = st.columns(4)

    for col, (numero, titulo, descripcion, color) in zip(cols, pasos):
        with col:
            st.markdown(
                f"""
                <div style="
                    min-height: 190px;
                    padding: 1rem;
                    border-radius: 10px;
                    background: #ffffff;
                    border-top: 5px solid {color};
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                ">
                    <div style="
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        width: 30px;
                        height: 30px;
                        border-radius: 50%;
                        background: {color};
                        color: white;
                        font-weight: 700;
                        margin-bottom: 0.55rem;
                    ">
                        {numero}
                    </div>
                    <div style="
                        color: #0A3D62;
                        font-size: 0.98rem;
                        font-weight: 700;
                        margin-bottom: 0.45rem;
                    ">
                        {titulo}
                    </div>
                    <div style="
                        color: #52616B;
                        font-size: 0.82rem;
                        line-height: 1.45;
                    ">
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
            & Min_e \leq Foliar_e \leq Max_e
            \end{cases}
            """
        )

    with col_rules:
        st.markdown("#### Reglas de decisión")

        st.markdown(
            """
            <div style="
                background: #F0F7FF;
                border-left: 4px solid #1b60a7;
                padding: 0.8rem 1rem;
                border-radius: 6px;
                margin-bottom: 0.65rem;
            ">
                <b>Foliar bajo</b><br>
                <span style="font-size: 0.85rem;">
                Se adiciona una corrección proporcional al déficit foliar.
                </span>
            </div>

            <div style="
                background: #F2FBF3;
                border-left: 4px solid #2ca02c;
                padding: 0.8rem 1rem;
                border-radius: 6px;
                margin-bottom: 0.65rem;
            ">
                <b>Foliar óptimo</b><br>
                <span style="font-size: 0.85rem;">
                La demanda ajustada conserva la demanda bruta.
                </span>
            </div>

            <div style="
                background: #FFF8E7;
                border-left: 4px solid #f39c12;
                padding: 0.8rem 1rem;
                border-radius: 6px;
                margin-bottom: 0.65rem;
            ">
                <b>Foliar alto</b><br>
                <span style="font-size: 0.85rem;">
                La demanda bruta se reduce al 50&nbsp;%.
                </span>
            </div>

            <div style="
                background: #F8F1FC;
                border-left: 4px solid #8e44ad;
                padding: 0.8rem 1rem;
                border-radius: 6px;
            ">
                <b>Eficiencia</b><br>
                <span style="font-size: 0.85rem;">
                Los elementos oficiales pasan directamente como recomendación;
                los demás se dividen por su eficiencia.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="
            background: #FAFAFA;
            border: 1px solid #DDE3E8;
            padding: 0.9rem 1rem;
            border-radius: 8px;
            font-size: 0.88rem;
            color: #52616B;
        ">
            <b>Flujo de unidades:</b>
            análisis foliar original → porcentaje foliar comparable →
            demanda bruta en kg/ha → demanda ajustada en kg/ha →
            recomendación final según eficiencia.
        </div>
        """,
        unsafe_allow_html=True
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
                "1.0 (da = rec)"
                if es_oficial
                else f"{EFICIENCIA.get(elem, 1.0)}"
            ),
        })

    st.dataframe(
        pd.DataFrame(ref_rows),
        use_container_width=True,
        hide_index=True,
    )


def tab_resumen(df: pd.DataFrame):

    st.markdown('<div class="section-title">Resumen de resultados</div>', unsafe_allow_html=True)

    elementos = [e for e in ELEMENTOS_CALCULO if f"fol_pct_{e.lower()}" in df.columns]
    if not elementos:
        st.info("No se detectaron columnas foliares en el dataset.")
        return

    rows = []
    for elem in elementos:
        fol_col = f"fol_{elem.lower()}"
        sta_col = f"status_{elem.lower()}"
        nec_col = f"nec_{elem.lower()}_kg_ha"
        r = FOLIAR_RANGES[elem]

        vals = df[fol_col].dropna()
        media = vals.mean() if len(vals) > 0 else np.nan
        status, label = get_foliar_status(foliar_a_pct(media, elem), elem)
        nec_media = df[nec_col].mean() if nec_col in df.columns else np.nan

        dist = df[sta_col].value_counts() if sta_col in df.columns else pd.Series()
        rows.append({
            "Nutriente": elem,
            "Unidad dataset": r["unidad"],
            "Rango Óptimo (%)": f"{r['min']} – {r['max']}",
            "Media Foliar": f"{media:.3f}" if not pd.isna(media) else "—",
            "Estado": label,
            "Nec. media (kg/ha)": f"{nec_media:.2f}" if not pd.isna(nec_media) else "—",
            "Lotes Bajo/Crítico": dist.get("bajo", 0) + dist.get("critico", 0),
            "Lotes Óptimo": dist.get("optimo", 0),
            "Lotes Alto": dist.get("alto", 0),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    nec_cols = [f"nec_{e.lower()}_kg_ha" for e in elementos if f"nec_{e.lower()}_kg_ha" in df.columns]
    lote_col = find_col(df.columns, ["lote"])

    st.markdown('<div class="section-title">Mapa de Calor — Recomendación por Lote × Nutriente</div>', unsafe_allow_html=True)

    if nec_cols and lote_col:
        df_heat = df[[lote_col] + nec_cols].dropna(subset=nec_cols, how="all")
        df_heat = df_heat.set_index(lote_col)
        df_heat.columns = [c.replace("nec_", "").replace("_kg_ha", "").upper() for c in df_heat.columns]
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
    st.markdown("---")

def tab_exportar(df: pd.DataFrame):

    elementos = [e for e in ELEMENTOS_CALCULO if f"fol_pct_{e.lower()}" in df.columns]

    lote_col     = find_col(df.columns, ["lote"])
    finca_col    = find_col(df.columns, ["finca"])
    edad_col     = find_col(df.columns, ["edad"])
    material_col = find_col(df.columns, ["material"])
    variedad_col = find_col(df.columns, ["variedad"])
    rff_col      = find_col(df.columns, ["ton_ha", "rff"])

    id_cols  = [c for c in [lote_col, finca_col, edad_col, material_col, variedad_col, rff_col] if c]
    nec_cols = [f"nec_{e.lower()}_kg_ha" for e in elementos if f"nec_{e.lower()}_kg_ha" in df.columns]
    sta_cols = [f"status_{e.lower()}"    for e in elementos if f"status_{e.lower()}"    in df.columns]

    df_export = df[id_cols + nec_cols + sta_cols].copy()

    rename = {}
    for e in elementos:
        nc = f"nec_{e.lower()}_kg_ha"
        sc = f"status_{e.lower()}"
        if nc in df_export.columns:
            rename[nc] = f"Rec_{e} (kg/ha)"
        if sc in df_export.columns:
            rename[sc] = f"Estado_{e}"
    df_export = df_export.rename(columns=rename)

    st.markdown('<div class="section-title">Vista previa — Resultados de Fertilización</div>', unsafe_allow_html=True)

    c_filtro1, c_filtro2 = st.columns([1, 3])
    with c_filtro1:
        if finca_col and finca_col in df_export.columns:
            fincas_disp = sorted(df_export[finca_col].dropna().unique().tolist())
            sel_finca_exp = st.multiselect(
                "Filtrar por finca:", fincas_disp, default=fincas_disp, key="exp_finca_filter"
            )
            if sel_finca_exp:
                df_export_view = df_export[df_export[finca_col].isin(sel_finca_exp)]
            else:
                df_export_view = df_export
        else:
            df_export_view = df_export

    with c_filtro2:
        estado_cols_disp = [c for c in df_export.columns if c.startswith("Estado_")]
        if estado_cols_disp:
            elem_foco = st.selectbox(
                "Resaltar estado del nutriente:",
                ["Ninguno"] + [c.replace("Estado_", "") for c in estado_cols_disp],
                key="exp_estado_foco"
            )
        else:
            elem_foco = "Ninguno"

    def resaltar_estado(val):
        v = str(val).lower()
        if v == "critico":
            return "background-color:#FDEDEC; color:#C0392B; font-weight:600;"
        if v == "bajo":
            return "background-color:#FEF9E7; color:#B9770E; font-weight:600;"
        if v == "alto":
            return "background-color:#EBF5FB; color:#1F618D; font-weight:600;"
        if v == "optimo":
            return "background-color:#EAFAF1; color:#1E8449; font-weight:600;"
        return ""

    styler = df_export_view.round(3).style
    estado_cols_present = [c for c in df_export_view.columns if c.startswith("Estado_")]
    if estado_cols_present:
        styler = styler.applymap(resaltar_estado, subset=estado_cols_present)

    st.dataframe(styler, use_container_width=True, hide_index=True)

    st.caption(f"Mostrando {len(df_export_view)} de {len(df_export)} lotes.")

    st.markdown('<div class="section-title">Descargar resultados</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        csv_bytes = df_export.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Descargar CSV",
            data=csv_bytes,
            file_name="fertilizacion_resultados.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_export.to_excel(writer, index=False, sheet_name="Resultados")
        buf.seek(0)
        st.download_button(
            label="📊 Descargar Excel",
            data=buf.getvalue(),
            file_name="fertilizacion_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("---")

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
                "Columnas principales: Lote, Finca, Edad, ton/ha "
                "y datos foliares N_f, P_f, K_f, Ca_f, Mg_f y B_f. "
                "S_f, Cu_f, Fe_f, Mn_f y Zn_f son opcionales. "
                "La clasificación del material puede realizarse mediante "
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
        <h1>Calculadora de Fertilización — FarmPrecision</h1>
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

    with st.sidebar:
        finca_col = find_col(data.columns, ["finca"])
        if finca_col:
            fincas = sorted(data[finca_col].dropna().unique().tolist())
            sel_finca = st.multiselect("Finca:", fincas, default=fincas, key="f_finca")
        else:
            sel_finca = []

    df = data.copy()
    if sel_finca and finca_col:
        df = df[df[finca_col].isin(sel_finca)]

    if df.empty:
        st.warning("Sin datos con los filtros actuales.")
        return

    seccion_kpis(df)
    st.markdown("---")

    tabs = st.tabs(["🔍 Información","📌 Resumen", "💾 Exportar"])
    with tabs[0]:
        tab_info()
    with tabs[1]:
        tab_resumen(df)
    with tabs[2]:
        tab_exportar(df)

if __name__ == "__main__":
    main()