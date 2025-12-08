# -------------------------------------------------
# Importaciones
# -------------------------------------------------
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
st.write("✅ VERSION NUEVA CARGADA")


# -------------------------------------------------
# Configuración general
# -------------------------------------------------
st.set_page_config(
    page_title="Sistema de Alerta de Deserción",
    page_icon="🎓",
    layout="wide"
)

st.title("Sistema de Alerta Temprana de Deserción Estudiantil")

# -------------------------------------------------
# Barra superior de navegación
# -------------------------------------------------
tab_proposito, tab_graficos, tab_sistema = st.tabs([
    "Propósito y Modelo",
    "Gráficos de Análisis",
    "Sistema de Riesgo"
])

# =================================================
# TAB 1 — PROPÓSITO
# =================================================
with tab_proposito:

    st.subheader("Propósito")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
        """
        <div style="font-size:16px; line-height:1.6;">

        <b>Detección temprana de deserción estudiantil:</b>  
        Mediante la recopilación de datos de perfiles de estudiantes anteriores, 
        con el propósito de crear un modelo sistemático que analice y alerte 
        las condiciones actuales de los estudiantes, permitiendo detectar casos 
        de <b>riesgo alto de deserción</b>.

        <br>

        <b>Detalle de los nuevos criterios por jerarquía:</b>

        <br>

        <b>Jerarquía 3 (Crítico):</b>

        a. El estudiante presenta la consideración de abandonar la carrera.<br>
        b. El estudiante pondera un promedio menor a 4.0.<br>
        c. El estudiante tiene una o más asignaturas reprobadas.<br>

        <i>Si cualquiera de estos es SÍ, se activa una <b>ALERTA ALTA</b>.</i>

        <br>

        <b>Jerarquía 2 (Moderado):</b>
        e. El estudiante presenta baja asistencia.<br>

        <i>Si este criterio es SÍ y no existe Jerarquía 3, se activa una <b>ALERTA BAJA</b>.</i>

        <br>

        <b>Jerarquía 1 (Bajo):</b>
        d. El estudiante presenta baja participación en clases.<br>

        <i>Si este criterio es SÍ y no existen Jerarquías 3 ni 2, se activa una <b>ALERTA BAJA</b>.</i>

        <br>
        """,
        unsafe_allow_html=True
    )

    with col2:
        st.image("proposito.jpeg", use_container_width=True)

    st.subheader("Flujo del árbol de decisión")

    st.markdown(
        """
        <div style="font-size:16px; line-height:1.6;">

        <b>Inicio:</b> Evaluación del estudiante.<br><br>

        <b>Condiciones críticas:</b>
        Si se cumple al menos una, se genera <b>ALERTA ALTA</b>.<br><br>

        <b>Baja asistencia:</b>
        En ausencia de criterios críticos, se genera <b>ALERTA BAJA</b>.<br><br>

        <b>Baja participación:</b>
        Último nivel de evaluación, también genera <b>ALERTA BAJA</b>.<br><br>

        <b>Riesgo bajo:</b>
        No se activa ninguna alerta.

        </div>
        """,
        unsafe_allow_html=True
    )

# =================================================
# TAB 2 — GRÁFICOS
# =================================================
with tab_graficos:

    st.subheader("Gráficos de Análisis de Deserción")

    @st.cache_data
    def load_data():
        df = pd.read_csv(
            "Cuestionario motivacion academica.csv",
            encoding="latin-1",
            sep=","
        )
        return df

    df = load_data()

    # Detección automática de columnas
    col_motivacion = [c for c in df.columns if "motiv" in c.lower()]
    col_abandono   = [c for c in df.columns if "aband" in c.lower()]
    col_reprobadas = [c for c in df.columns if "reprob" in c.lower()]

    if not col_motivacion or not col_abandono or not col_reprobadas:
        st.error("No se encontraron las columnas necesarias")
        st.stop()

    df = df[[col_motivacion[0], col_abandono[0], col_reprobadas[0]]].copy()
    df.columns = ["Motivacion", "Pensando_Abandonar", "Reprobadas"]
    df = df.dropna()

    df["Reprobadas"] = pd.to_numeric(df["Reprobadas"], errors="coerce").fillna(0)
    df["Pensando_Abandonar"] = df["Pensando_Abandonar"].astype(str).str.strip().str.lower()

    df["Pensando_Abandonar"] = df["Pensando_Abandonar"].replace({
        "sí": "si", "si ": "si", "no ": "no", "SI": "si", "NO": "no"
    })

    # Gráfico 1
    st.header("Motivación según pensamiento de abandono")

    fig1 = px.box(
        df,
        x="Pensando_Abandonar",
        y="Motivacion",
        color="Pensando_Abandonar"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2
    st.header("Motivación promedio según asignaturas reprobadas")

    promedios = df.groupby("Reprobadas")["Motivacion"].mean().reset_index()

    fig2 = px.bar(
        promedios,
        x="Reprobadas",
        y="Motivacion"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab_sistema:

    st.subheader("Sistema de Riesgo: Perfiles Académicos (Clustering)")

    # ------------------------------------------
    # 1. CARGA DE DATOS
    # ------------------------------------------
    @st.cache_data
    def load_facultad():
        try:
            return pd.read_csv(
                "Data_UINN_Facultad.csv",
                sep=";",
                decimal=",",
                encoding="utf-8",
                header=3
            )
        except:
            return pd.read_csv(
                "Data_UINN_Facultad.csv",
                sep=";",
                decimal=",",
                encoding="latin-1",
                header=3
            )

    df_fac = load_facultad()

    if "Código Carrera Nacional" not in df_fac.columns:
        st.error("No se encuentra la columna 'Código Carrera Nacional'")
        st.stop()

    # ------------------------------------------
    # 2. FILTRO DE CARRERAS
    # ------------------------------------------
    relacion_carreras = [
        (13072, 3309),
        (13069, 3310),
        (13070, 3311),
        (13071, 3318),
        (13019, 3303),
        (13073, 3319)
    ]

    codigos_nacionales = [x[0] for x in relacion_carreras]

    df_fac = df_fac[df_fac["Código Carrera Nacional"].isin(codigos_nacionales)].copy()

    # ------------------------------------------
    # 3. LIMPIEZA
    # ------------------------------------------
    cols_numericas = ["Puntaje Ponderado", "Puntaje NEM", "Puntaje Ranking"]
    for col in cols_numericas:
        df_fac[col] = pd.to_numeric(df_fac[col], errors="coerce")

    # Filtro de puntajes mínimos
    df_fac = df_fac[
        (df_fac["Puntaje NEM"] >= 400) &
        (df_fac["Puntaje Ranking"] >= 400)
    ]

    df_fac["Segmento_Geo"] = df_fac["Domicilio Región"].astype(str).str.upper().apply(
        lambda x: "LOCAL (Biobío)" if "BIOBIO" in x else "FORÁNEO (Otras Regiones)"
    )

    # ------------------------------------------
    # 4. CLUSTERING
    # ------------------------------------------
    def crear_clusters(df, n_clusters=3):
        features = df[["Puntaje NEM", "Puntaje Ranking", "Puntaje Ponderado"]]
        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(X)

        return df

    df_fac = crear_clusters(df_fac)

    # ------------------------------------------
    # 5. VISUALIZACIÓN
    # ------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.scatterplot(
        data=df_fac,
        x="Puntaje Ranking",
        y="Puntaje NEM",
        hue="Cluster",
        style="Segmento_Geo",
        palette="viridis",
        s=80,
        alpha=0.75,
        ax=ax
    )

    ax.set_title("Perfiles de Estudiantes según Puntajes y Origen Geográfico")
    ax.set_xlabel("Puntaje Ranking")
    ax.set_ylabel("Puntaje NEM")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.set_xlim(400, df_fac["Puntaje Ranking"].max() + 10)
    ax.set_ylim(400, df_fac["Puntaje NEM"].max() + 10)

    st.pyplot(fig)


    st.subheader("Sistema de Alertas por Árbol de Decisión")

    st.subheader("Sistema de Riesgo Académico")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # ==========================================
    # 1. CARGA DE DATOS
    # ==========================================
    df = pd.read_csv(
        "Cuestionario motivacion academica.csv",
        encoding="latin-1"
    )
    
    st.success("✅ Archivo cargado")
    st.write("Filas totales:", len(df))
    
    # ------------------------------------------
    # FILTRO DE CARRERAS UDEC
    # ------------------------------------------
    codigos_udec = [3309, 3310, 3311, 3318, 3303, 3319]
    
    df.rename(columns={df.columns[0]: "Codigo_Carrera"}, inplace=True)
    df["Codigo_Carrera"] = pd.to_numeric(df["Codigo_Carrera"], errors="coerce")
    
    df = df[df["Codigo_Carrera"].isin(codigos_udec)].copy()
    
    st.write("🎓 Filas tras filtrar carreras:", len(df))
    
    # ==========================================
    # 2. DETECCIÓN DE COLUMNAS
    # ==========================================
    def encontrar_columna(df, keywords):
        for col in df.columns:
            if all(k.lower() in col.lower() for k in keywords):
                return col
        return None
    
    col_reprobadas    = encontrar_columna(df, ["reprob"])
    col_asistencia    = encontrar_columna(df, ["asist"])
    col_participacion = encontrar_columna(df, ["particip"])
    col_motivacion    = encontrar_columna(df, ["motiv"])
    
    st.markdown("### Columnas detectadas")
    st.write({
        "Reprobadas": col_reprobadas,
        "Asistencia": col_asistencia,
        "Participación": col_participacion,
        "Motivación": col_motivacion
    })
    
    # ==========================================
    # 3. LIMPIEZA
    # ==========================================
    for col in [col_reprobadas, col_asistencia, col_participacion, col_motivacion]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # ==========================================
    # 4. ÁRBOL DE DECISIÓN
    # ==========================================
    def calcular_alerta(row):
        if row[col_motivacion] <= 2 and row[col_reprobadas] >= 2:
            return "ALERTA ALTA"
    
        if row[col_asistencia] <= 3:
            return "ALERTA PREVENTIVA - Asistencia"
    
        if row[col_participacion] <= 2:
            return "ALERTA BAJA - Participación"
    
        return "Sin Riesgo"
    
    df["Nivel_Riesgo"] = df.apply(calcular_alerta, axis=1)
    
    # ==========================================
    # 5. RESULTADOS NUMÉRICOS
    # ==========================================
    st.subheader("Distribución total de alertas")
    st.dataframe(df["Nivel_Riesgo"].value_counts())
    
    st.subheader("Distribución por carrera y alerta")
    tabla_carrera = (
        df.groupby(["Codigo_Carrera", "Nivel_Riesgo"])
          .size()
          .unstack(fill_value=0)
    )
    st.dataframe(tabla_carrera)
    
    # ==========================================
    # 6. GRÁFICO GENERAL
    # ==========================================
    st.subheader("Clasificación general de estudiantes")
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.countplot(
        data=df,
        y="Nivel_Riesgo",
        order=df["Nivel_Riesgo"].value_counts().index,
        palette="Reds",
        ax=ax1
    )
    
    ax1.set_xlabel("Cantidad de Estudiantes")
    ax1.set_ylabel("Nivel de Alerta")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    
    st.pyplot(fig1)
    
    # ==========================================
    # 7. GRÁFICO: ALERTAS POR CARRERA
    # ==========================================
    st.subheader("Alertas por carrera")
    
    tabla_alertas = (
        df.groupby(["Codigo_Carrera", "Nivel_Riesgo"])
          .size()
          .reset_index(name="Cantidad")
    )
    
    tabla_alertas = tabla_alertas[
        tabla_alertas["Nivel_Riesgo"] != "Sin Riesgo"
    ]
    
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    
    sns.barplot(
        data=tabla_alertas,
        x="Codigo_Carrera",
        y="Cantidad",
        hue="Nivel_Riesgo",
        palette={
            "ALERTA ALTA": "#c0392b",
            "ALERTA PREVENTIVA - Asistencia": "#f39c12",
            "ALERTA BAJA - Participación": "#2980b9"
        },
        ax=ax2
    )
    
    ax2.set_xlabel("Código de Carrera")
    ax2.set_ylabel("Cantidad de Estudiantes")
    ax2.set_title("Cantidad de Estudiantes por Carrera y Nivel de Alerta")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    st.pyplot(fig2)
    
    # =================================================
    # TABLA CRUZADA: CIUDAD vs NIVEL DE RIESGO
    # =================================================
    st.subheader("Concentración de Riesgo por Ciudad de Origen")
    
    # Usamos el mismo dataframe de encuesta (df)
    df_encuesta = df.copy()
    
    # ------------------------------------------
    # 1. Búsqueda inteligente de la columna ciudad
    # ------------------------------------------
    col_ciudad_real = None
    for col in df_encuesta.columns:
        if "ciudad" in col.lower() and "origen" in col.lower():
            col_ciudad_real = col
            break
    
    if col_ciudad_real:
        st.success(f"✅ Columna de ciudad detectada: {col_ciudad_real}")
        col_ciudad = col_ciudad_real
    else:
        st.warning("⚠️ No se encontró columna explícita de ciudad, usando columna 2 por defecto")
        col_ciudad = df_encuesta.columns[1]
    
    # ------------------------------------------
    # 2. Limpieza y normalización
    # ------------------------------------------
    df_encuesta["Ciudad_Norm"] = (
        df_encuesta[col_ciudad]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    
    df_encuesta["Ciudad_Norm"] = df_encuesta["Ciudad_Norm"].replace({
        "CONCEPCION": "CONCEPCIÓN",
        "LOS ANGELES": "LOS ÁNGELES",
        "SAN PEDRO": "SAN PEDRO DE LA PAZ",
        "CHILLAN": "CHILLÁN"
    })
    
    # ------------------------------------------
    # 3. Filtro Top 15 ciudades
    # ------------------------------------------
    top_ciudades = df_encuesta["Ciudad_Norm"].value_counts().nlargest(15).index
    df_top_ciudades = df_encuesta[df_encuesta["Ciudad_Norm"].isin(top_ciudades)]
    
    # ------------------------------------------
    # 4. Tabla cruzada
    # ------------------------------------------
    crosstab = pd.crosstab(
        df_top_ciudades["Ciudad_Norm"],
        df_top_ciudades["Nivel_Riesgo"]
    )
    
    orden_columnas = [
        "ALERTA ALTA",
        "ALERTA PREVENTIVA - Asistencia",
        "ALERTA BAJA - Participación",
        "Sin Riesgo"
    ]
    
    cols_existentes = [c for c in orden_columnas if c in crosstab.columns]
    crosstab = crosstab[cols_existentes]
    
    st.markdown("### Tabla de Frecuencias")
    st.dataframe(crosstab)
    
    # ------------------------------------------
    # 5. Heatmap
    # ------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(
        crosstab,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax3
    )
    
    ax3.set_title("Concentración de Riesgo por Ciudad de Origen", fontsize=14)
    ax3.set_xlabel("Nivel de Riesgo")
    ax3.set_ylabel("Ciudad de Origen")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    
    st.pyplot(fig3)

