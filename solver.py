"""Colección de utilidades y algoritmos para resolver modelos de optimización.

El archivo concentra tanto el asistente interactivo de consola (entradas, menús y
resumen de datos) como las implementaciones de cada solver disponible. La idea es
mantener toda la lógica de negocio centralizada aquí para que pueda ser consumida
por cualquier interfaz (CLI, Streamlit u otras integraciones futuras).
"""

from pulp import LpMaximize, LpMinimize, LpProblem, LpVariable, LpStatus, value
from scipy.optimize import linprog

try:
    from rich.console import Console
    from rich.prompt import Prompt, IntPrompt
    from rich.table import Table
except ImportError:  # pragma: no cover - degradado amable si rich no está instalado
    Console = None
    Prompt = None
    IntPrompt = None
    Table = None


console = Console() if Console else None

ALGORITMOS_DISPONIBLES = {
    'simplex': "Método Simplex",
    'punto_interior': "Método de Pivoteo Interior",
    'programacion_entera': "Programación Entera",
    'algoritmo_dual': "Método Dual",
    'relajacion_lagrangiana': "Relajación Lagrangiana",
    'auto': "Selección automática"
}


def _imprimir(texto, style=None):
    """Imprime texto usando Rich si está disponible, o recurre a print estándar."""

    if console:
        console.print(texto, style=style)
    else:
        print(texto)


def _solicitar_texto(mensaje):
    """Pide una entrada de texto mostrando un mensaje homogéneo en toda la CLI."""

    if Prompt:
        return Prompt.ask(mensaje)
    return input(f"{mensaje}\n> ").strip()


def _solicitar_entero(mensaje, minimo=1):
    """Solicita un entero validando que cumpla el mínimo requerido."""

    while True:
        try:
            if IntPrompt:
                valor = IntPrompt.ask(mensaje, default=minimo)
            else:
                valor = int(input(f"{mensaje}\n> "))
            if valor < minimo:
                raise ValueError
            return valor
        except ValueError:
            _imprimir(f"Ingrese un número entero mayor o igual a {minimo}.", style="bold red")


def _solicitar_lista_floats(mensaje):
    """Convierte una entrada separada por espacios en una lista de flotantes."""

    while True:
        entrada = _solicitar_texto(mensaje)
        try:
            valores = [float(num) for num in entrada.split()]
            if not valores:
                raise ValueError
            return valores
        except ValueError:
            _imprimir("Ingrese números válidos separados por espacio.", style="bold red")


def _formatear_variables(valores):
    """Asigna nombres x1, x2, ... a una secuencia de valores numéricos."""

    return {f"x{i + 1}": valor for i, valor in enumerate(valores)}


def _resultado_base(metodo, exito, estado, variables=None, valor_objetivo=None, mensaje=None):
    """Crea el diccionario homogéneo con el que responden todos los algoritmos."""

    return {
        'metodo': metodo,
        'exito': exito,
        'estado': estado,
        'variables': variables or {},
        'valor_objetivo': valor_objetivo,
        'mensaje': mensaje
    }

def validar_entrada(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones):
    """Realiza comprobaciones básicas antes de intentar resolver el modelo."""

    try:
        if not isinstance(coef_objetivo, list) or not all(isinstance(i, (int, float)) for i in coef_objetivo):
            raise ValueError("Los coeficientes de la función objetivo deben ser una lista de números.")

        if not isinstance(restricciones, list) or not all(isinstance(r, list) for r in restricciones):
            raise ValueError("Las restricciones deben ser una lista de listas.")

        if not isinstance(tipo_restricciones, list) or not all(tr in ['<=', '>=', '='] for tr in tipo_restricciones):
            raise ValueError("Las restricciones deben tener tipos válidos: ['<=', '>=', '=']")

        if len(restricciones) != len(tipo_restricciones) or len(restricciones) != len(valores_restricciones):
            raise ValueError("Cada restricción debe tener un tipo y un valor del lado derecho asociados.")

        num_vars = len(coef_objetivo)
        if any(len(restriccion) != num_vars for restriccion in restricciones):
            raise ValueError(
                "Todas las restricciones deben tener el mismo número de coeficientes que la función objetivo."
            )

        return True, None

    except Exception as e:
        return False, str(e)


def seleccionar_algoritmo():
    """Muestra las opciones de solver disponibles y devuelve la clave interna elegida."""

    opciones = {
        '1': 'simplex',
        '2': 'punto_interior',
        '3': 'programacion_entera',
        '4': 'algoritmo_dual',
        '5': 'relajacion_lagrangiana',
        '6': 'auto'
    }

    _imprimir("Seleccione el algoritmo que desea usar:", style="bold cyan")
    _imprimir("1. Método Simplex")
    _imprimir("2. Método de Pivoteo Interior (Método de Punto Interior)")
    _imprimir("3. Método de Programación Entera (Branch and Bound)")
    _imprimir("4. Método Dual")
    _imprimir("5. Método de Relajación Lagrangiana")
    _imprimir("6. Selección automática (si no sabes qué elegir)")

    opcion = _solicitar_texto("Ingrese el número del algoritmo")
    while opcion not in opciones:
        _imprimir("Opción inválida. Intente nuevamente.", style="bold red")
        opcion = _solicitar_texto("Ingrese el número del algoritmo")

    return opciones[opcion]


def mostrar_menu_principal(datos):
    """Pinta el menú principal del asistente e indica pasos completados."""

    opciones = [
        ("1", "Configurar función objetivo", bool(datos.get('coef_objetivo'))),
        ("2", "Definir tipo de variables", len(datos.get('tipo_variables', [])) == len(datos.get('coef_objetivo') or [])),
        ("3", "Añadir restricciones", bool(datos.get('restricciones'))),
        ("4", "Mostrar resumen y continuar"),
        ("0", "Salir")
    ]

    _imprimir("\n=== Asistente de configuración ===", style="bold green")
    for clave, etiqueta, *estado in opciones:
        completado = estado[0] if estado else False
        marca = "✅" if completado else "⬜"
        if clave in {"4", "0"}:
            marca = "➡" if clave == "4" else marca
        _imprimir(f"{clave}. {etiqueta} {marca}")

    opcion = _solicitar_texto("Seleccione una opción del menú")
    while opcion not in {op[0] for op in opciones}:
        _imprimir("Opción inválida. Intente nuevamente.", style="bold red")
        opcion = _solicitar_texto("Seleccione una opción del menú")
    return opcion


def capturar_funcion_objetivo():
    """Devuelve la lista de coeficientes de la función objetivo capturada por CLI."""

    return _solicitar_lista_floats("Ingrese los coeficientes de la función objetivo (separados por espacio)")


def capturar_tipo_variables(num_variables):
    """Pide el tipo (continua/entera) para cada variable del modelo."""

    tipos = []
    for idx in range(num_variables):
        tipo = _solicitar_texto(f"Tipo de la variable x{idx + 1} (continua/entera)").strip().lower()
        while tipo not in {'continua', 'entera'}:
            _imprimir("Tipo inválido. Debe ser 'continua' o 'entera'.", style="bold red")
            tipo = _solicitar_texto(f"Reingrese el tipo de x{idx + 1}").strip().lower()
        tipos.append(tipo)
    return tipos


def capturar_restricciones(num_variables):
    """Recoge coeficientes, signo y RHS para todas las restricciones del problema."""

    num_restricciones = _solicitar_entero("¿Cuántas restricciones tiene el problema?", minimo=1)
    restricciones, tipos, valores = [], [], []

    for i in range(num_restricciones):
        mensaje_coef = f"Coeficientes de la restricción {i + 1} (separados por espacio)"
        restriccion = _solicitar_lista_floats(mensaje_coef)
        while len(restriccion) != num_variables:
            _imprimir(
                f"La restricción debe tener {num_variables} coeficientes para coincidir con la función objetivo.",
                style="bold red"
            )
            restriccion = _solicitar_lista_floats(mensaje_coef)
        restricciones.append(restriccion)

        tipo = _solicitar_texto(f"Tipo de restricción {i + 1} (<=, >=, =)").strip()
        while tipo not in {'<=', '>=', '='}:
            _imprimir("Tipo inválido de restricción. Debe ser <=, >= o =.", style="bold red")
            tipo = _solicitar_texto(f"Tipo de restricción {i + 1} (<=, >=, =)").strip()
        tipos.append(tipo)

        valor = float(_solicitar_texto(f"Valor del lado derecho para la restricción {i + 1}"))
        valores.append(valor)

    return restricciones, tipos, valores


def mostrar_resumen(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones):
    """Muestra tabla/resumen textual y pregunta si se desea continuar."""

    _imprimir("\nResumen del modelo ingresado", style="bold cyan")
    num_vars = len(coef_objetivo)
    headers = [f"x{i + 1}" for i in range(num_vars)]

    if Table:
        table = Table(show_lines=True)
        table.add_column("Expresión", justify="left", style="bold")
        for header in headers:
            table.add_column(header, justify="center")
        table.add_column("Tipo")
        table.add_column("RHS")

        table.add_row("Función objetivo", *[str(c) for c in coef_objetivo], "Max", "-")
        for idx, (rest, signo, rhs) in enumerate(zip(restricciones, tipo_restricciones, valores_restricciones), start=1):
            table.add_row(f"Restricción {idx}", *[str(c) for c in rest], signo, str(rhs))
        console.print(table) if console else print(table)
    else:
        _imprimir(f"Función objetivo: Max z = {' + '.join(f'{coef}*x{i + 1}' for i, coef in enumerate(coef_objetivo))}")
        for idx, (rest, signo, rhs) in enumerate(zip(restricciones, tipo_restricciones, valores_restricciones), start=1):
            lhs = ' + '.join(f"{coef}*x{i + 1}" for i, coef in enumerate(rest))
            _imprimir(f"Restricción {idx}: {lhs} {signo} {rhs}")

    respuesta = _solicitar_texto("¿Desea continuar con estos datos? (s/n)").lower()
    while respuesta not in {'s', 'n'}:
        respuesta = _solicitar_texto("Responda con 's' para sí o 'n' para no").lower()
    return respuesta == 's'


def mostrar_resultado(resultado):
    """Imprime en consola el resumen del solver ejecutado y sus métricas."""

    estilo = "bold green" if resultado.get('exito') else "bold yellow"
    _imprimir(f"\nResultado ({ALGORITMOS_DISPONIBLES.get(resultado.get('metodo'), 'Algoritmo')})", style=estilo)
    _imprimir(f"Estado: {resultado.get('estado')}")
    if resultado.get('mensaje'):
        _imprimir(resultado['mensaje'])

    if resultado.get('variables'):
        _imprimir("Variables óptimas:", style="bold")
        for nombre, valor in resultado['variables'].items():
            _imprimir(f"  {nombre} = {valor}")

    if resultado.get('valor_objetivo') is not None:
        _imprimir(f"Valor de la función objetivo: {resultado['valor_objetivo']}")

def resolver_con_punto_interior(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones):
    """Implementa una formulación estándar y llama a SciPy para resolverla."""

    c = [-coef for coef in coef_objetivo]
    A_ub, b_ub, A_eq, b_eq = [], [], [], []

    for restriccion, signo, valor in zip(restricciones, tipo_restricciones, valores_restricciones):
        # SciPy espera el formato estándar Ax <= b; convertimos cada restricción.
        if signo == '<=':
            A_ub.append(restriccion)
            b_ub.append(valor)
        elif signo == '>=':
            A_ub.append([-coef for coef in restriccion])
            b_ub.append(-valor)
        else:  # '='
            A_eq.append(restriccion)
            b_eq.append(valor)

    # El asistente únicamente modela variables con cota inferior cero.
    bounds = [(0, None) for _ in coef_objetivo]
    res = linprog(
        c,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq if A_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=bounds,
        method='interior-point'
    )

    if res.success:
        variables = _formatear_variables(res.x)
        return _resultado_base(
            'punto_interior',
            True,
            res.message or 'Optimal',
            variables=variables,
            valor_objetivo=-res.fun,
            mensaje='Solución encontrada con el método de punto interior.'
        )

    return _resultado_base(
        'punto_interior',
        False,
        res.message or 'Sin solución',
        mensaje='No se encontró una solución óptima con el método de punto interior.'
    )

def _construir_restriccion_pulp(prob, expr, signo, rhs, nombre):
    """Añade la restricción apropiada al modelo PuLP indicado."""

    if signo == '<=':
        prob += expr <= rhs, nombre
    elif signo == '>=':
        prob += expr >= rhs, nombre
    else:
        prob += expr == rhs, nombre


def resolver_simplex(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones, max_iter=None):
    """Resuelve un modelo de maximización continuo mediante PuLP Simplex."""

    prob = LpProblem("Metodo_Simplex", LpMaximize)
    variables = [LpVariable(f"x{i+1}", lowBound=0, cat='Continuous') for i in range(len(coef_objetivo))]

    # Maximiza z = c^T x.
    prob += sum(coef * var for coef, var in zip(coef_objetivo, variables)), "Funcion_Objetivo"

    for idx, (restriccion, signo, valor) in enumerate(zip(restricciones, tipo_restricciones, valores_restricciones), start=1):
        expr = sum(coef * var for coef, var in zip(restriccion, variables))
        _construir_restriccion_pulp(prob, expr, signo, valor, f"Restriccion_{idx}")

    prob.solve()

    estado = LpStatus.get(prob.status, "Inconnu")
    if estado == "Optimal":
        valores = {var.name: var.varValue for var in variables}
        return _resultado_base(
            'simplex',
            True,
            estado,
            variables=valores,
            valor_objetivo=value(prob.objective),
            mensaje='Solución encontrada con el método Simplex.'
        )

    return _resultado_base(
        'simplex',
        False,
        estado,
        mensaje='No se encontró una solución óptima con el método Simplex.'
    )


def resolver_con_programacion_entera(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones):
    """Configura el mismo modelo pero usando variables enteras (Branch and Bound)."""

    prob = LpProblem("Optimizacion_Entera", LpMaximize)
    variables = [LpVariable(f"x{i+1}", lowBound=0, cat='Integer') for i in range(len(coef_objetivo))]

    prob += sum(coef * var for coef, var in zip(coef_objetivo, variables)), "Ganancia_Total"

    for idx, (restriccion, signo, valor) in enumerate(zip(restricciones, tipo_restricciones, valores_restricciones), start=1):
        expr = sum(coef * var for coef, var in zip(restriccion, variables))
        _construir_restriccion_pulp(prob, expr, signo, valor, f"Restriccion_{idx}")

    prob.solve()

    estado = LpStatus.get(prob.status, "Inconnu")
    if estado == "Optimal":
        valores = {var.name: var.varValue for var in variables}
        return _resultado_base(
            'programacion_entera',
            True,
            estado,
            variables=valores,
            valor_objetivo=value(prob.objective),
            mensaje='Solución encontrada con programación entera.'
        )

    return _resultado_base(
        'programacion_entera',
        False,
        estado,
        mensaje='No se encontró una solución óptima en programación entera.'
    )


def _normalizar_restricciones_en_menor_igual(restricciones, tipo_restricciones, valores_restricciones):
    """Convierte >= y = en restricciones equivalentes en formato <=."""

    restricciones_normalizadas = []
    valores_normalizados = []

    for coefs, signo, rhs in zip(restricciones, tipo_restricciones, valores_restricciones):
        if signo == '<=':
            restricciones_normalizadas.append(list(coefs))
            valores_normalizados.append(rhs)
        elif signo == '>=':
            restricciones_normalizadas.append([-coef for coef in coefs])
            valores_normalizados.append(-rhs)
        else:  # '=' se descompone en dos restricciones
            restricciones_normalizadas.append(list(coefs))
            valores_normalizados.append(rhs)
            restricciones_normalizadas.append([-coef for coef in coefs])
            valores_normalizados.append(-rhs)

    return restricciones_normalizadas, valores_normalizados


def resolver_con_algoritmo_dual(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones):
    """Construye explícitamente el dual y lo resuelve como minimización en PuLP."""

    restricciones_canonicas, rhs_canonicos = _normalizar_restricciones_en_menor_igual(
        restricciones, tipo_restricciones, valores_restricciones
    )

    if not restricciones_canonicas:
        return _resultado_base(
            'algoritmo_dual',
            False,
            'Sin restricciones',
            mensaje='El método dual requiere al menos una restricción para construir el problema dual.'
        )

    prob = LpProblem("Metodo_Dual", LpMinimize)
    dual_vars = [LpVariable(f"y{i+1}", lowBound=0) for i in range(len(restricciones_canonicas))]

    prob += sum(b * y for b, y in zip(rhs_canonicos, dual_vars)), "Funcion_Objetivo_Dual"

    for idx_var in range(len(coef_objetivo)):
        expr = sum(restriccion[idx_var] * dual_vars[idx] for idx, restriccion in enumerate(restricciones_canonicas))
        prob += expr >= coef_objetivo[idx_var], f"cota_variable_{idx_var + 1}"

    prob.solve()
    estado = LpStatus.get(prob.status, "Inconnu")
    if estado == "Optimal":
        valores_duales = {var.name: var.varValue for var in dual_vars}
        valor_objetivo = value(prob.objective)
        mensaje = (
            "El método dual se resolvió correctamente. Puedes interpretar estas variables como"
            " precios sombra asociados a las restricciones originales."
        )
        return _resultado_base(
            'algoritmo_dual',
            True,
            estado,
            variables=valores_duales,
            valor_objetivo=valor_objetivo,
            mensaje=mensaje
        )

    return _resultado_base(
        'algoritmo_dual',
        False,
        estado,
        mensaje='No se pudo encontrar una solución dual factible con los datos proporcionados.'
    )


def resolver_con_relajacion_lagrangiana(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones):
    """Aplica penalizaciones a las violaciones de restricciones para flexibilizar el modelo."""

    restricciones_canonicas, rhs_canonicos = _normalizar_restricciones_en_menor_igual(
        restricciones, tipo_restricciones, valores_restricciones
    )

    prob = LpProblem("Relajacion_Lagrangiana", LpMaximize)
    variables = [LpVariable(f"x{i+1}", lowBound=0, cat='Continuous') for i in range(len(coef_objetivo))]
    slacks = [LpVariable(f"s_relaj_{i+1}", lowBound=0, cat='Continuous') for i in range(len(restricciones_canonicas))]

    # Penalizamos con suficiente peso las holguras para desalentar violaciones.
    penalizacion = max(10.0, 10 * sum(abs(c) for c in coef_objetivo) or 1.0)
    prob += (
        sum(coef * var for coef, var in zip(coef_objetivo, variables))
        - penalizacion * sum(slacks)
    ), "Funcion_Objetivo_Relajada"

    for idx, (coefs, rhs) in enumerate(zip(restricciones_canonicas, rhs_canonicos)):
        expr = sum(coef * var for coef, var in zip(coefs, variables))
        prob += expr <= rhs + slacks[idx], f"Restriccion_relajada_{idx + 1}"

    prob.solve()
    estado = LpStatus.get(prob.status, "Inconnu")
    if estado == "Optimal":
        valores = {var.name: var.varValue for var in variables}
        violaciones = {var.name: var.varValue for var in slacks}
        violaciones_activas = {k: v for k, v in violaciones.items() if v and v > 1e-6}
        mensaje = "Resolución con relajación lagrangiana y penalización de violaciones."
        if violaciones_activas:
            mensaje += f" Restricciones con holgura positiva: {violaciones_activas}."
        else:
            mensaje += " No se observaron violaciones significativas."
        return _resultado_base(
            'relajacion_lagrangiana',
            True,
            estado,
            variables=valores,
            valor_objetivo=value(prob.objective),
            mensaje=mensaje
        )

    return _resultado_base(
        'relajacion_lagrangiana',
        False,
        estado,
        mensaje='No se logró resolver la relajación lagrangiana con los parámetros actuales.'
    )

def seleccionar_algoritmo_automaticamente(tipo_variables, restricciones):
    """Regresa la heurística escogida cuando el usuario pide selección automática."""

    if todas_las_variables_son_continuas(tipo_variables):
        return "simplex"
    elif todas_las_variables_son_enteras(tipo_variables):
        return "programacion_entera"
    elif problema_de_gran_escala(restricciones, len(tipo_variables)):
        return "relajacion_lagrangiana"
    else:
        return "algoritmo_dual"

def todas_las_variables_son_continuas(tipo_variables):
    """Comprueba si no hay variables enteras declaradas en el modelo."""

    return all(tipo == 'continua' for tipo in tipo_variables)


def todas_las_variables_son_enteras(tipo_variables):
    """Comprueba si todas las variables fueron declaradas como enteras."""

    return all(tipo == 'entera' for tipo in tipo_variables)


def problema_de_gran_escala(restricciones, num_variables):
    """Define una heurística simple para detectar instancias medianas/grandes."""

    return len(restricciones) > 5 or num_variables > 5


def resolver_modelo(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones,
                    tipo_variables=None, metodo='auto'):
    """Punto de entrada único para resolver desde cualquier interfaz."""

    tipo_variables = tipo_variables or []
    if metodo in {None, 'auto'}:
        metodo = seleccionar_algoritmo_automaticamente(tipo_variables, restricciones)

    dispatch = {
        'simplex': resolver_simplex,
        'punto_interior': resolver_con_punto_interior,
        'programacion_entera': resolver_con_programacion_entera,
        'algoritmo_dual': resolver_con_algoritmo_dual,
        'relajacion_lagrangiana': resolver_con_relajacion_lagrangiana
    }

    solver = dispatch.get(metodo)
    if not solver:
        return _resultado_base(
            metodo or 'desconocido',
            False,
            'Método no soportado',
            mensaje='El algoritmo solicitado no está disponible.'
        )

    resultado = solver(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones)
    if resultado.get('metodo') != metodo:
        resultado['metodo'] = metodo
    return resultado

def datos_configurados(datos):
    """Ayuda a determinar si el asistente ya tiene la información mínima."""

    coef = datos.get('coef_objetivo')
    tipos = datos.get('tipo_variables')
    restricciones = datos.get('restricciones')
    return bool(coef) and len(tipos) == len(coef) and bool(restricciones)


def main():
    """Ejecuta el asistente de consola hasta capturar el modelo y resolverlo."""

    datos = {
        'coef_objetivo': None,
        'tipo_variables': [],
        'restricciones': [],
        'tipo_restricciones': [],
        'valores_restricciones': []
    }

    while True:
        opcion_menu = mostrar_menu_principal(datos)

        if opcion_menu == '1':
            datos['coef_objetivo'] = capturar_funcion_objetivo()
            datos['tipo_variables'] = []  # Reiniciar pasos dependientes
            datos['restricciones'] = []
            datos['tipo_restricciones'] = []
            datos['valores_restricciones'] = []
        elif opcion_menu == '2':
            if not datos.get('coef_objetivo'):
                _imprimir("Primero configure la función objetivo.", style="bold red")
                continue
            datos['tipo_variables'] = capturar_tipo_variables(len(datos['coef_objetivo']))
        elif opcion_menu == '3':
            if not datos.get('coef_objetivo'):
                _imprimir("Configure la función objetivo antes de capturar las restricciones.", style="bold red")
                continue
            restricciones, tipos, valores = capturar_restricciones(len(datos['coef_objetivo']))
            datos['restricciones'] = restricciones
            datos['tipo_restricciones'] = tipos
            datos['valores_restricciones'] = valores
        elif opcion_menu == '4':
            if not datos_configurados(datos):
                _imprimir("Complete los pasos anteriores antes de continuar.", style="bold red")
                continue
            continuar = mostrar_resumen(
                datos['coef_objetivo'],
                datos['restricciones'],
                datos['tipo_restricciones'],
                datos['valores_restricciones']
            )
            if continuar:
                break
            else:
                _imprimir("Puede regresar al menú y editar los datos.", style="yellow")
        elif opcion_menu == '0':
            _imprimir("Hasta luego", style="bold")
            return

    coef_objetivo = datos['coef_objetivo']
    tipo_variables = datos['tipo_variables']
    restricciones = datos['restricciones']
    tipo_restricciones = datos['tipo_restricciones']
    valores_restricciones = datos['valores_restricciones']

    valido, error = validar_entrada(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones)
    if not valido:
        _imprimir(f"Error de validación: {error}", style="bold red")
        return

    metodo = seleccionar_algoritmo()
    resultado = resolver_modelo(
        coef_objetivo,
        restricciones,
        tipo_restricciones,
        valores_restricciones,
        tipo_variables=tipo_variables,
        metodo=metodo
    )
    mostrar_resultado(resultado)

if __name__ == "__main__":
    main()