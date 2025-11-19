"""Interfaz web construida con Streamlit sobre los servicios de ``solver.py``.

El archivo actúa como una “capa de presentación” que hace lo siguiente:

1. Configura una página ancha con estilos personalizados para los botones.
2. Utiliza la barra lateral como asistente resumido (cantidad de variables,
   restricciones y algoritmo deseado).
3. Muestra un formulario con todos los coeficientes/tipos y, antes de resolver,
   enseña un resumen tabular para que el estudiante verifique su modelo.

No contiene lógica matemática propia: cada validación y algoritmo se delega al
módulo ``solver``, garantizando que CLI y web se comporten igual.
"""

import pandas as pd
import streamlit as st

from solver import (
    ALGORITMOS_DISPONIBLES,
    resolver_modelo,
    seleccionar_algoritmo_automaticamente,
    validar_entrada,
)

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Solver de Programación Lineal", layout="wide")
st.title("Asistente interactivo de optimización")
st.write(
    "Configura tu problema de programación lineal, confirma un resumen tabular y "
    "ejecuta el algoritmo que prefieras sin salir del navegador."
)

# Aplicamos un pequeño tema para que el botón de envío sea más atractivo.
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        border-radius: 999px;
        background: linear-gradient(120deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        padding: 0.6em 1.6em;
        font-weight: 600;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    div.stButton > button:first-child:focus {
        outline: none;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    # -----------------------------------------------------------------------
    # Asistente rápido en la barra lateral
    # -----------------------------------------------------------------------
    st.header("Parámetros generales")
    # Estos valores definen cuántos campos dinámicos aparecerán en el formulario.
    num_variables = int(
        st.number_input("Número de variables", min_value=1, max_value=10, value=2, step=1)
    )
    num_restricciones = int(
        st.number_input("Número de restricciones", min_value=1, max_value=10, value=2, step=1)
    )

    opciones_algoritmo = {
        "Selección automática": "auto",
        "Método Simplex": "simplex",
        "Método de Pivoteo Interior": "punto_interior",
        "Programación Entera": "programacion_entera",
        "Método Dual": "algoritmo_dual",
        "Relajación Lagrangiana": "relajacion_lagrangiana",
    }
    algoritmo_label = st.selectbox("Algoritmo", list(opciones_algoritmo.keys()))
    metodo_seleccionado = opciones_algoritmo[algoritmo_label]

with st.form("configuracion_modelo"):
    # -----------------------------------------------------------------------
    # Formulario principal (función objetivo, tipos y restricciones)
    # -----------------------------------------------------------------------
    # Agrupamos la captura de datos para validar/enviar todo en un único submit.
    st.subheader("Función objetivo ")
    coef_objetivo = []
    for i in range(num_variables):
        coef = st.number_input(
            f"Coeficiente de x{i + 1}",
            key=f"coef_obj_{i}",
            value=1.0 if i == 0 else 0.0,
            format="%0.3f",
        )
        coef_objetivo.append(float(coef))

    st.subheader("Tipo de variables")
    tipo_variables = []
    for i in range(num_variables):
        tipo = st.selectbox(
            f"Tipo de x{i + 1}",
            options=["continua", "entera"],
            index=0,
            key=f"tipo_var_{i}",
        )
        tipo_variables.append(tipo)

    st.subheader("Restricciones")
    restricciones = []
    tipo_restricciones = []
    valores_restricciones = []
    for r in range(num_restricciones):
        st.markdown(f"**Restricción {r + 1}**")
        coeficientes = []
        for v in range(num_variables):
            coef = st.number_input(
                f"Coeficiente de x{v + 1} en R{r + 1}",
                key=f"rest_{r}_{v}",
                value=0.0,
                format="%0.3f",
            )
            coeficientes.append(float(coef))
        restricciones.append(coeficientes)

        tipo = st.selectbox(
            f"Tipo de restricción {r + 1}",
            options=["<=", ">=", "="],
            key=f"tipo_rest_{r}",
        )
        tipo_restricciones.append(tipo)

        rhs = st.number_input(
            f"Valor del lado derecho R{r + 1}",
            key=f"rhs_{r}",
            value=0.0,
            format="%0.3f",
        )
        valores_restricciones.append(float(rhs))

    submitted = st.form_submit_button(
        "Resolver modelo",
        use_container_width=True,
        help="Ejecuta el algoritmo seleccionado con los parámetros configurados.",
        type="primary",
    )

st.subheader("Resumen del modelo")
# Mostramos un resumen de todo lo capturado antes de resolver. Es equivalente
# a la tabla que se enseña en la CLI con Rich, pero en versión tabular.
if coef_objetivo:
    st.write("Función objetivo: ")
    st.latex(
        "z = "
        + " + ".join(f"{coef} x_{{{i + 1}}}" for i, coef in enumerate(coef_objetivo))
        + " \\rightarrow \\max"
    )

if restricciones:
    headers = [f"x{i + 1}" for i in range(num_variables)]
    data = []
    for coef, tipo, rhs in zip(restricciones, tipo_restricciones, valores_restricciones):
        fila = {h: valor for h, valor in zip(headers, coef)}
        fila["Tipo"] = tipo
        fila["RHS"] = rhs
        data.append(fila)
    df_resumen = pd.DataFrame(data)
    st.table(df_resumen)

if submitted:
    # -----------------------------------------------------------------------
    # Validación y ejecución (solo ocurre si se pulsó el botón del formulario)
    # -----------------------------------------------------------------------
    # Validamos la consistencia de los datos antes de ejecutar cualquier algoritmo.
    valido, error = validar_entrada(
        coef_objetivo, restricciones, tipo_restricciones, valores_restricciones
    )
    if not valido:
        st.error(f"Error de validación: {error}")
    else:
        metodo = metodo_seleccionado
        if metodo == "auto":
            # Reutilizamos la heurística definida en solver.py para escoger el mejor método.
            metodo = seleccionar_algoritmo_automaticamente(tipo_variables, restricciones)
            st.info(
                f"Selección automática: se ejecutará {ALGORITMOS_DISPONIBLES.get(metodo, metodo)}"
            )
        resultado = resolver_modelo(
            coef_objetivo,
            restricciones,
            tipo_restricciones,
            valores_restricciones,
            tipo_variables=tipo_variables,
            metodo=metodo,
        )

        if resultado.get("exito"):
            st.success(
                f"Estado: {resultado.get('estado')} — Valor objetivo: {resultado.get('valor_objetivo')}"
            )
        else:
            st.warning(resultado.get("mensaje") or "No se encontró solución")

        if resultado.get("variables"):
            st.write("Variables óptimas")
            st.json(resultado["variables"])