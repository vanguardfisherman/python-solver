from pulp import LpMaximize, LpProblem, LpVariable, LpStatus, value
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


def _imprimir(texto, style=None):
    if console:
        console.print(texto, style=style)
    else:
        print(texto)


def _solicitar_texto(mensaje):
    if Prompt:
        return Prompt.ask(mensaje)
    return input(f"{mensaje}\n> ").strip()


def _solicitar_entero(mensaje, minimo=1):
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
    while True:
        entrada = _solicitar_texto(mensaje)
        try:
            valores = [float(num) for num in entrada.split()]
            if not valores:
                raise ValueError
            return valores
        except ValueError:
            _imprimir("Ingrese números válidos separados por espacio.", style="bold red")

# Función para validar las entradas
def validar_entrada(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones):
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
            raise ValueError("Todas las restricciones deben tener el mismo número de coeficientes que la función objetivo.")

        return True

    except Exception as e:
        print(f"Error de validación: {e}")
        return False

# Función para seleccionar el algoritmo
def seleccionar_algoritmo():
    _imprimir("Seleccione el algoritmo que desea usar:", style="bold cyan")
    _imprimir("1. Método Simplex")
    _imprimir("2. Método de Pivoteo Interior (Método de Punto Interior)")
    _imprimir("3. Método de Programación Entera (Branch and Bound)")
    _imprimir("4. Método de Descomposición de Dantzig-Wolfe")
    _imprimir("5. Método de Relajación Lagrangiana")
    _imprimir("6. Selección automática (si no sabes qué elegir)")

    opcion = _solicitar_texto("Ingrese el número del algoritmo")

    if opcion in ['1', '2', '3', '4', '5']:
        return int(opcion)
    else:
        print("Opción inválida o no elegida, seleccionando automáticamente el algoritmo...")
        return 6  # El programa elegirá el algoritmo automáticamente


def mostrar_menu_principal(datos):
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
    return _solicitar_lista_floats("Ingrese los coeficientes de la función objetivo (separados por espacio)")


def capturar_tipo_variables(num_variables):
    tipos = []
    for idx in range(num_variables):
        tipo = _solicitar_texto(f"Tipo de la variable x{idx + 1} (continua/entera)").strip().lower()
        while tipo not in {'continua', 'entera'}:
            _imprimir("Tipo inválido. Debe ser 'continua' o 'entera'.", style="bold red")
            tipo = _solicitar_texto(f"Reingrese el tipo de x{idx + 1}").strip().lower()
        tipos.append(tipo)
    return tipos


def capturar_restricciones(num_variables):
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

# Función para resolver con el Método de Pivoteo Interior
def resolver_con_punto_interior(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones):
    c = [-coef for coef in coef_objetivo]
    A_ub, b_ub, A_eq, b_eq = [], [], [], []

    for restriccion, signo, valor in zip(restricciones, tipo_restricciones, valores_restricciones):
        if signo == '<=':
            A_ub.append(restriccion)
            b_ub.append(valor)
        elif signo == '>=':
            A_ub.append([-coef for coef in restriccion])
            b_ub.append(-valor)
        else:  # '='
            A_eq.append(restriccion)
            b_eq.append(valor)

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
        for idx, valor in enumerate(res.x, start=1):
            print(f"Cantidad óptima de x{idx}: {valor}")
        print(f"Ganancia máxima: {-res.fun}")
    else:
        print("No se encontró una solución óptima con el método de punto interior.")

# Utilidades para formulaciones con PuLP
def _construir_restriccion_pulp(prob, expr, signo, rhs, nombre):
    if signo == '<=':
        prob += expr <= rhs, nombre
    elif signo == '>=':
        prob += expr >= rhs, nombre
    else:
        prob += expr == rhs, nombre


def resolver_simplex(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones, max_iter=None):
    prob = LpProblem("Metodo_Simplex", LpMaximize)
    variables = [LpVariable(f"x{i+1}", lowBound=0, cat='Continuous') for i in range(len(coef_objetivo))]

    prob += sum(coef * var for coef, var in zip(coef_objetivo, variables)), "Funcion_Objetivo"

    for idx, (restriccion, signo, valor) in enumerate(zip(restricciones, tipo_restricciones, valores_restricciones), start=1):
        expr = sum(coef * var for coef, var in zip(restriccion, variables))
        _construir_restriccion_pulp(prob, expr, signo, valor, f"Restriccion_{idx}")

    prob.solve()

    estado = LpStatus.get(prob.status, "Inconnu")
    print(f"Estado del solucionador (Simplex): {estado}")
    if estado == "Optimal":
        for var in variables:
            print(f"{var.name} = {var.varValue}")
        print(f"Valor óptimo de la función objetivo: {value(prob.objective)}")
    else:
        print("No se encontró una solución óptima con el método Simplex.")


# Función para resolver con el Método de Programación Entera (Branch and Bound)
def resolver_con_programacion_entera(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones):
    prob = LpProblem("Optimizacion_Entera", LpMaximize)
    variables = [LpVariable(f"x{i+1}", lowBound=0, cat='Integer') for i in range(len(coef_objetivo))]

    prob += sum(coef * var for coef, var in zip(coef_objetivo, variables)), "Ganancia_Total"

    for idx, (restriccion, signo, valor) in enumerate(zip(restricciones, tipo_restricciones, valores_restricciones), start=1):
        expr = sum(coef * var for coef, var in zip(restriccion, variables))
        _construir_restriccion_pulp(prob, expr, signo, valor, f"Restriccion_{idx}")

    prob.solve()

    estado = LpStatus.get(prob.status, "Inconnu")
    print(f"Estado del solucionador (Programación Entera): {estado}")
    if estado == "Optimal":
        for var in variables:
            print(f"{var.name} = {var.varValue}")
        print(f"Ganancia máxima: {value(prob.objective)}")
    else:
        print("No se encontró una solución óptima en el método de programación entera.")


def resolver_con_dantzig_wolfe(*_args, **_kwargs):
    print("El método de Descomposición de Dantzig-Wolfe aún no está implementado."
          " Esta función actúa como placeholder para futuras extensiones.")


def resolver_con_relajacion_lagrangiana(*_args, **_kwargs):
    print("El método de Relajación Lagrangiana aún no está implementado."
          " Puedes seleccionar otro algoritmo mientras se desarrolla esta característica.")

# Función para resolver el problema automáticamente
def seleccionar_algoritmo_automaticamente(tipo_variables, restricciones):
    if todas_las_variables_son_continuas(tipo_variables):
        return "simplex"
    elif todas_las_variables_son_enteras(tipo_variables):
        return "programacion_entera"
    elif problema_de_gran_escala(restricciones, len(tipo_variables)):
        return "dantzig_wolfe"
    else:
        return "punto_interior"

# Funciones auxiliares

# Verifica si todas las variables son continuas
def todas_las_variables_son_continuas(tipo_variables):
    return all(tipo == 'continua' for tipo in tipo_variables)

# Verifica si todas las variables son enteras
def todas_las_variables_son_enteras(tipo_variables):
    return all(tipo == 'entera' for tipo in tipo_variables)

# Verifica si el problema es de gran escala (si tiene más de 5 restricciones o variables)
def problema_de_gran_escala(restricciones, num_variables):
    return len(restricciones) > 5 or num_variables > 5

def datos_configurados(datos):
    coef = datos.get('coef_objetivo')
    tipos = datos.get('tipo_variables')
    restricciones = datos.get('restricciones')
    return bool(coef) and len(tipos) == len(coef) and bool(restricciones)


# Función principal
def main():
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

    if not validar_entrada(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones):
        return

    # Selección del algoritmo
    opcion = seleccionar_algoritmo()

    if opcion == 1:
        print("Usando el Método Simplex...")
        resolver_simplex(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones, max_iter=5)
    elif opcion == 2:
        print("Usando el Método de Pivoteo Interior...")
        resolver_con_punto_interior(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones)
    elif opcion == 3:
        print("Usando el Método de Programación Entera...")
        resolver_con_programacion_entera(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones)
    elif opcion == 4:
        print("Usando el Método de Descomposición de Dantzig-Wolfe...")
        resolver_con_dantzig_wolfe(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones)
    elif opcion == 5:
        print("Usando el Método de Relajación Lagrangiana...")
        resolver_con_relajacion_lagrangiana(coef_objetivo, restricciones, valores_restricciones)
    elif opcion == 6:
        print("El programa elegirá el algoritmo adecuado automáticamente...")
        metodo = seleccionar_algoritmo_automaticamente(tipo_variables, restricciones)
        print(f"Algoritmo seleccionado automáticamente: {metodo}")
        # Aquí puedes decidir qué algoritmo ejecutar basado en la selección automática
        if metodo == "simplex":
            resolver_simplex(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones, max_iter=5)
        elif metodo == "programacion_entera":
            resolver_con_programacion_entera(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones)
        elif metodo == "punto_interior":
            resolver_con_punto_interior(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones)
        elif metodo == "dantzig_wolfe":
            resolver_con_dantzig_wolfe(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones)

if __name__ == "__main__":
    main()
