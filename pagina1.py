import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración general
st.set_page_config(
    page_title="Sistema de Alerta de Deserción",
    page_icon="🎓",
    layout="wide"
)

st.title("Sistema de Alerta Temprana de Deserción Estudiantil")

# Barra superior de navegación
tab_proposito, tab_graficos, tab_sistema = st.tabs([
    "Propósito y Modelo",
    "Gráficos de Análisis",
    "Sistema de Riesgo"
])

# PROPÓSITO
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
    st.markdown(' ')
    st.markdown(
        """
        <div style="font-size:16px; line-height:1.6;">

        
        <b>
        Inicio:</b> Evaluación del estudiante.<br><br>

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

# GRÁFICOS

with tab_graficos:

    st.subheader("Gráficos de Análisis de Deserción")
    st.markdown(
        """
        Para este análisis, nos enfocamos principalmente en el documento Cuestionario de Motivación.
        Este contiene las respuestas de los estudiantes cuando se les preguntó
        sobre su etapa universitaria. Por lo que revisamos los datos, para luego limpiar las
        variables que nos interesan y así hacer un análisis gráfico, para poder detectar tendencias
        generales en el abandono de carrera.
        
        En esta limpieza hicimos uso de tres columnas principales:
        
        --> Motivación por estudiar la carrera
        
        --> Pensamiento de abandonar la carrera
        
        --> Cantidad de asignaturas reprobadas
        
        El primer gráfico que hicimos, compara los niveles de motivación entre quienes han
        pensado en abandonar y quienes no. Esto a través de un diagrama de caja. Aquí
        podemos observar que quienes estaban pensando en dejar la carrera, tenían un nivel de
        motivación más bajo. Lo que se concluye como una relación directa entre las variables.
        En el segundo gráfico, observamos el promedio de motivación en base a las asignaturas
        reprobadas. Donde se observó que, mientras que el número de asignaturas reprobadas
        aumenta, la motivación del estudiante baja. Por lo que el rendimiento del estudiante es
        un factor a considerar dentro del sistema de alerta.
        """
    )
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
    df.columns = ["Motivacion", "Pensamiento_de_abandono_en_escala_del_1_al_5", "Reprobadas"]
    df = df.dropna()

    df["Reprobadas"] = pd.to_numeric(df["Reprobadas"], errors="coerce").fillna(0)
    df["Pensamiento_de_abandono_en_escala_del_1_al_5"] = df["Pensamiento_de_abandono_en_escala_del_1_al_5"].astype(str).str.strip().str.lower()

    df["Pensamiento_de_abandono_en_escala_del_1_al_5"] = df["Pensamiento_de_abandono_en_escala_del_1_al_5"].replace({
        "sí": "si", "si ": "si", "no ": "no", "SI": "si", "NO": "no"
    })

    # Gráfico 1
    st.header("Motivación según pensamiento de abandono")

    fig1 = px.box(
        df,
        x="Pensamiento_de_abandono_en_escala_del_1_al_5",
        y="Motivacion",
        color="Pensamiento_de_abandono_en_escala_del_1_al_5"
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

   
    # FILTRO DE CARRERAS
    
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

    # LIMPIEZA
    
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

    # CLUSTERING
    def crear_clusters(df, n_clusters=3):
        features = df[["Puntaje NEM", "Puntaje Ranking", "Puntaje Ponderado"]]
        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(X)

        return df

    df_fac = crear_clusters(df_fac)

    # VISUALIZACIÓN
    column1, column2 = st.columns([2, 1])
    with column1:
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
    with column2:
        st.markdown(
            """
            Lo primero que vemos en el sistema de riesgo, es un clustering, es decir, un algoritmo de agrupamiento. 
            Esto permite identificar perfiles de estudiantes según sus puntajes de ingreso, considerando Puntaje Nem, Ranking y Ponderado. Además,
            se diferencia según el origen geográfico, siendo los locales nativos de la región del Bío Bío representados con Círculos, y los Foráneos,
            provenientes de otras regiones, representados con Equis.

            Este análisis no busca predecir la deserción, sino entender cómo se distribuyen los perfiles académicos al momento de ingresar a la universidad.
            Para así observar si existen diferencias estructurales según el origen geográfico. 
            Se distinguen claramente tres perfiles académicos:
            
            Cluster 0: Un grupo con puntajes altos y consistentes, asociado a menor riesgo académico.
            
            Cluster 1: Un grupo intermedio, con mayor dispersión.
            
            Cluster 2: Un grupo con puntajes más bajos, donde se concentra el riesgo potencial
            
            Aquí observamos que los estudiantes foráneos aparecen más dispersos, esto indica mayores diferencias de adaptación académica.
            
            Esto permite detectar perfiles de ingreso con mayor vulnerabilidad antes de que aparezcan problemas académicos graves. Es decir,
            motiva a tener reforzamientos tempranos según el tipo de perfil.
            """
        )


    st.subheader("")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df = pd.read_csv(
        "Cuestionario motivacion academica.csv",
        encoding="latin-1"
    )
    
    
    #st.write("Filas totales:", len(df))
    
    # FILTRO DE CARRERAS UDEC
    codigos_udec = [3309, 3310, 3311, 3318, 3303, 3319]
    
    df.rename(columns={df.columns[0]: "Codigo_Carrera"}, inplace=True)
    df["Codigo_Carrera"] = pd.to_numeric(df["Codigo_Carrera"], errors="coerce")
    
    df = df[df["Codigo_Carrera"].isin(codigos_udec)].copy()
    
    #st.write("🎓 Filas tras filtrar carreras:", len(df))
    

    # DETECCIÓN DE COLUMNAS
    def encontrar_columna(df, keywords):
        for col in df.columns:
            if all(k.lower() in col.lower() for k in keywords):
                return col
        return None
    
    col_reprobadas    = encontrar_columna(df, ["reprob"])
    col_asistencia    = encontrar_columna(df, ["asist"])
    col_participacion = encontrar_columna(df, ["particip"])
    col_motivacion    = encontrar_columna(df, ["motiv"])
    
    #st.markdown("Columnas detectadas")
    #st.write({
    #   "Reprobadas": col_reprobadas,
    #    "Asistencia": col_asistencia,
    #    "Participación": col_participacion,
    #    "Motivación": col_motivacion
    #})
    
   
    # LIMPIEZA
    for col in [col_reprobadas, col_asistencia, col_participacion, col_motivacion]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    

    # ÁRBOL DE DECISIÓN
    def calcular_alerta(row):
        if row[col_motivacion] <= 2 and row[col_reprobadas] >= 2:
            return "ALERTA ALTA"
    
        if row[col_asistencia] <= 3:
            return "ALERTA PREVENTIVA - Asistencia"
    
        if row[col_participacion] <= 2:
            return "ALERTA BAJA - Participación"
    
        return "Sin Riesgo"
    
    df["Nivel_Riesgo"] = df.apply(calcular_alerta, axis=1)
    
  
    # RESULTADOS NUMÉRICOS
 
    #st.subheader("Distribución total de alertas")
    #st.dataframe(df["Nivel_Riesgo"].value_counts())
    
    #st.subheader("Distribución por carrera y alerta")
    #tabla_carrera = (
    #   df.groupby(["Codigo_Carrera", "Nivel_Riesgo"])
    #      .size()
    #      .unstack(fill_value=0)
    #)
    #st.dataframe(tabla_carrera)
    
   
    # GRÁFICO GENERAL
    st.subheader("Clasificación general de estudiantes")

    pilar1, pilar2 = st.columns([2, 1])
    with pilar1:
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
        
    with pilar2:
       st.markdown(
           """
           Este gráfico clasifica a los estudiantes según el nivel de alerta definido en el modelo del sistema de riesgo.
           Esto a través de un árbol de decisión basado en variables tales como, la motivación, rendimiento académico, asistencia,
           y participación en clases. 

            Observamos que la gran parte de los alumnos no presenta riesgo, esto es importante ya que indica que el sistema no sobrerreacciona
            ni clasifica erróneamente a los alumnos. Luego, vemos lo que se puede esperar de un sistema de este tipo, los estudiantes en cada
            nivel de riesgo disminuyen mientras más alto sea el riesgo. 
            
            El hecho de considerar la participación y asistencia, muestra que, el riesgo no solo está siempre relacionado directamente con
            el rendimiento académico, sino con factores conductuales que pueden abordarse mediante apoyo oportuno. Estas son entonces las 
            primeras señales de deserción futura.
            
            Con esta información, se puede intervenir antes de que los estudiantes lleguen al punto de reprobar asignaturas, aplicando medidas
            de bajo costo, como por ejemplo, incitar a la participación en clases.
            """
       )
       
  
    # GRÁFICO: ALERTAS POR CARRERA

    st.subheader("Alertas por carrera")

    pil1, pil2 = st.columns([2, 1])

    with pil1:
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

    with pil2:
                st.markdown(
            """
            3309: Ing. Civil Industrial
            
            3310: Ing. Civil 
            
            3311: Ing. Civil Eléctrica
            
            3318: Ing. Civil Electrónica
            
            3303: Ing. Comercial
            
            3319: Ing. Civil Informática
            
            Este gráfico nos muestra la cantidad de estudiantes en cada nivel de riesgo, según cada carrera. Esto nos permite identificar si existen
            carreras con una mayor cantidad de estudiantes en riesgo, para así segmentar las medidas y mejorar la gestión académica, junto con la toma
            de decisiones institucionales. Por lo que, en lugar de evaluar el riesgo de forma aislada, se consigue evaluar la realidad de cada carrera,
            con el fin de priorizar los recursos. 
            
            Observamos que varias carreras presentan una alta concentración de alumnos en alerta baja, lo que interpretamos como la existencia de una
            cierta carga académica o problemas en cuanto a la metodología de enseñanza. Por otro lado, hay carreras que presentan un mayor número de
            estudiantes en alerta alta, lo que demuestra que la deserción no es homogénea, por lo que no todas las carreras requieren las mismas estrategias.
            
            De esta manera, el sistema de alerta no solo identifica estudiantes en riesgo, sino que también entrega información relevante para el diseño
            de estrategias preventivas focalizadas. Así se evitan medidas ''genéricas'' que suelen ser poco eficientes.
            """
        )
        
    
    # CIUDAD vs NIVEL DE RIESGO
    st.subheader("Concentración de Riesgo por Ciudad de Origen")
    
    # Usamos el mismo dataframe de encuesta (df)
    df_encuesta = df.copy()
    
 
    # Búsqueda inteligente de la columna ciudad
    col_ciudad_real = None
    for col in df_encuesta.columns:
        if "ciudad" in col.lower() and "origen" in col.lower():
            col_ciudad_real = col
            break
    
    if col_ciudad_real:
        col_ciudad = col_ciudad_real
    else:
        st.warning("⚠️ No se encontró columna explícita de ciudad, usando columna 2 por defecto")
        col_ciudad = df_encuesta.columns[1]
    
    # Limpieza y normalización
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
    
 
    # Filtro Top 15 ciudades
    top_ciudades = df_encuesta["Ciudad_Norm"].value_counts().nlargest(15).index
    df_top_ciudades = df_encuesta[df_encuesta["Ciudad_Norm"].isin(top_ciudades)]
    
    # Tabla cruzada
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
    
    #st.markdown("### Tabla de Frecuencias")
    #st.dataframe(crosstab)
    
    # Heatmap
    hill1, hill2 = st.columns([2,1])
    with hill1:
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

    with hill2:
        st.markdown(
            """
            El mapa de calor muestra la distribución del nivel de riesgo según la ciudad de origen de los estudiantes,
            permitiendo observar patrones geográficos asociados a distintas categorías de alerta.

            Este análisis no busca establecer causalidades ni estigmatizar a los estudiantes por su origen, sino identificar
            posibles concentraciones que puedan relacionarse con procesos de adaptación, distancia geográfica o diferencias en
            el contexto previo al ingreso a la universidad.
            
            Observamos que ciudades con un gran volumen de estudiantes, concentran naturalmente más casos, con y sin riesgo.
            Sin embargo, también existen ciudades con una mayor proporción de alertas altas o preventivas. Esto puede reflejar 
            dificultades de adaptación, traslado, contexto socioeconómico o redes de apoyo.
            
            
            Esta información es crucial para políticas de acompañamiento territorial, como redes de apoyo, tutorías o programas de apoyo 
            diferenciados para estudiantes foráneos, que deberían tener un seguimiento inicial.
            """
        )


    st.subheader(' ')
    st.subheader('Conclusión')
    st.markdown(' ')
    st.markdown(
        """
        De este modo, se logra construir un sistema de alerta temprana que incorpora datos contextuales,
        de comportamiento y académicos de los alumnos, a partir de los gráficos mostrados previamente.
        
        Este sistema no se centra en analizar casos individuales de manera aislada, ni espera a que se produzcan
        fracasos académicos para alertar; más bien, es una herramienta de ayuda que posibilita la detección de patrones
        de riesgo y señales tempranas de eventual abandono.
        
        
        Así, el análisis posibilita orientar la toma de decisiones preventivas, como el seguimiento de la asistencia
        y la participación en las clases, así como el acompañamiento académico, dando prioridad a los recursos en aquellos
        grupos y contextos donde hay mayor riesgo.  Esto favorece que el riesgo de deserción en la universidad sea administrado
        de manera más proactiva y efectiva.
        """
    )


