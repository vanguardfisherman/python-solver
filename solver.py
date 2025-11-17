from pulp import LpMaximize, LpProblem, LpVariable, LpStatus, value
from scipy.optimize import linprog

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
    print("Seleccione el algoritmo que desea usar:")
    print("1. Método Simplex")
    print("2. Método de Pivoteo Interior (Método de Punto Interior)")
    print("3. Método de Programación Entera (Branch and Bound)")
    print("4. Método de Descomposición de Dantzig-Wolfe")
    print("5. Método de Relajación Lagrangiana")
    print("6. Selección automática (si no sabes qué elegir)")

    opcion = input("Ingrese el número del algoritmo: ")

    if opcion in ['1', '2', '3', '4', '5']:
        return int(opcion)
    else:
        print("Opción inválida o no elegida, seleccionando automáticamente el algoritmo...")
        return 6  # El programa elegirá el algoritmo automáticamente

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

# Función principal
def main():
    # Ingresar los coeficientes de la función objetivo
    print("Ingrese los coeficientes de la función objetivo (separados por espacio):")
    coef_objetivo = list(map(float, input().split()))  # Ingresar como: 3 5
    tipo_variables = []
    for idx in range(len(coef_objetivo)):
        tipo = input(f"Ingrese el tipo de la variable x{idx+1} (continua/entera): ").strip().lower()
        while tipo not in {'continua', 'entera'}:
            print("Tipo inválido. Debe ser 'continua' o 'entera'.")
            tipo = input(f"Ingrese nuevamente el tipo de la variable x{idx+1}: ").strip().lower()
        tipo_variables.append(tipo)

    # Ingresar las restricciones
    num_restricciones = int(input("¿Cuántas restricciones tiene el problema? "))

    restricciones = []
    tipo_restricciones = []
    valores_restricciones = []

    for i in range(num_restricciones):
        print(f"Ingrese los coeficientes de la restricción {i+1} (separados por espacio):")
        restriccion = list(map(float, input().split()))  # Ingresar como: 2 3 para 2x1 + 3x2 <= valor
        restricciones.append(restriccion)

        print(f"Ingrese el tipo de restricción {i+1} (<=, >=, =):")
        tipo = input().strip()
        while tipo not in {'<=', '>=', '='}:
            print("Tipo inválido de restricción. Debe ser <=, >= o =.")
            tipo = input().strip()
        tipo_restricciones.append(tipo)

        print(f"Ingrese el valor del lado derecho de la restricción {i+1}:")
        valor = float(input())  # Valor del lado derecho de la restricción
        valores_restricciones.append(valor)

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
