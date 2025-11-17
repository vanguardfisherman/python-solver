from pulp import LpMaximize, LpProblem, LpVariable, LpStatus, value
from scipy.optimize import linprog

# Función para validar las entradas
def validar_entrada(coef_objetivo, restricciones, tipo_restricciones):
    try:
        if not isinstance(coef_objetivo, list) or not all(isinstance(i, (int, float)) for i in coef_objetivo):
            raise ValueError("Los coeficientes de la función objetivo deben ser una lista de números.")

        if not isinstance(restricciones, list) or not all(isinstance(r, list) for r in restricciones):
            raise ValueError("Las restricciones deben ser una lista de listas.")

        if not isinstance(tipo_restricciones, list) or not all(tr in ['<=', '>=', '='] for tr in tipo_restricciones):
            raise ValueError("Las restricciones deben tener tipos válidos: ['<=', '>=', '=']")

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
def resolver_con_punto_interior(coef_objetivo, restricciones, valores_restricciones):
    c = [-coef_objetivo[0], -coef_objetivo[1]]  # Maximizar 3x1 + 5x2 (convertido a minimizar)
    A = [restricciones[0], restricciones[1]]  # Coeficientes de las restricciones
    b = valores_restricciones  # Lado derecho de las restricciones

    # Resolver usando el método de punto interior
    res = linprog(c, A_ub=A, b_ub=b, method='interior-point')

    if res.success:
        print(f"Cantidad óptima de producto 1 (x1): {res.x[0]}")
        print(f"Cantidad óptima de producto 2 (x2): {res.x[1]}")
        print(f"Ganancia máxima: {-res.fun}")  # Recordamos que invertimos los signos
    else:
        print("No se encontró una solución óptima con el método de punto interior.")

# Función para resolver con el Método de Programación Entera (Branch and Bound)
def resolver_con_programacion_entera(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones):
    prob = LpProblem("Optimización Entera", LpMaximize)
    x1 = LpVariable("x1", lowBound=0, cat='Integer')
    x2 = LpVariable("x2", lowBound=0, cat='Integer')

    prob += coef_objetivo[0] * x1 + coef_objetivo[1] * x2, "Ganancia Total"
    prob += restricciones[0][0] * x1 + restricciones[0][1] * x2 <= valores_restricciones[0], "Restricción 1"
    prob += restricciones[1][0] * x1 + restricciones[1][1] * x2 <= valores_restricciones[1], "Restricción 2"

    prob.solve()

    if LpStatus[prob.status] == "Optimal":
        print(f"Cantidad óptima de producto 1 (x1): {x1.varValue}")
        print(f"Cantidad óptima de producto 2 (x2): {x2.varValue}")
        print(f"Ganancia máxima: {value(prob.objective)}")
    else:
        print("No se encontró una solución óptima.")

# Función para resolver el problema automáticamente
def seleccionar_algoritmo_automaticamente(coef_objetivo, restricciones, valores_restricciones):
    # Lógica para determinar el algoritmo adecuado
    if todas_las_variables_son_continuas(coef_objetivo):
        return "simplex"
    elif todas_las_variables_son_enteras(coef_objetivo):
        return "programacion_entera"
    elif problema_de_gran_escala(coef_objetivo, restricciones):
        return "dantzig_wolfe"
    else:
        return "punto_interior"

# Funciones auxiliares

# Verifica si todas las variables son continuas
def todas_las_variables_son_continuas(coef_objetivo):
    return all(isinstance(x, (int, float)) for x in coef_objetivo)

# Verifica si todas las variables son enteras
def todas_las_variables_son_enteras(coef_objetivo):
    return all(isinstance(x, int) for x in coef_objetivo)

# Verifica si el problema es de gran escala (si tiene más de 5 restricciones)
def problema_de_gran_escala(coef_objetivo, restricciones):
    return len(restricciones) > 5

# Función principal
def main():
    # Ingresar los coeficientes de la función objetivo
    print("Ingrese los coeficientes de la función objetivo (separados por espacio):")
    coef_objetivo = list(map(float, input().split()))  # Ingresar como: 3 5

    # Ingresar las restricciones
    num_restricciones = int(input("¿Cuántas restricciones tiene el problema? "))

    restricciones = []
    tipo_restricciones = []
    valores_restricciones = []

    for i in range(num_restricciones):
        print(f"Ingrese los coeficientes de la restricción {i+1} (separados por espacio):")
        restriccion = list(map(float, input().split()))  # Ingresar como: 2 3 para 2x1 + 3x2 <= valor
        restricciones.append(restriccion)

        print(f"Ingrese el tipo de restricción {i+1} (<, >, =):")
        tipo = input()
        tipo_restricciones.append(tipo)

        print(f"Ingrese el valor del lado derecho de la restricción {i+1}:")
        valor = float(input())  # Valor del lado derecho de la restricción
        valores_restricciones.append(valor)

    # Selección del algoritmo
    opcion = seleccionar_algoritmo()

    if opcion == 1:
        print("Usando el Método Simplex...")
        resolver_simplex(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones, max_iter=5)
    elif opcion == 2:
        print("Usando el Método de Pivoteo Interior...")
        resolver_con_punto_interior(coef_objetivo, restricciones, valores_restricciones)
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
        metodo = seleccionar_algoritmo_automaticamente(coef_objetivo, restricciones, valores_restricciones)
        print(f"Algoritmo seleccionado automáticamente: {metodo}")
        # Aquí puedes decidir qué algoritmo ejecutar basado en la selección automática
        if metodo == "simplex":
            resolver_simplex(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones, max_iter=5)
        elif metodo == "programacion_entera":
            resolver_con_programacion_entera(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones)
        elif metodo == "punto_interior":
            resolver_con_punto_interior(coef_objetivo, restricciones, valores_restricciones)
        elif metodo == "dantzig_wolfe":
            resolver_con_dantzig_wolfe(coef_objetivo, restricciones, tipo_restricciones, valores_restricciones)

if __name__ == "__main__":
    main()
