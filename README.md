# Solver CLI

Colección de utilidades para resolver modelos de optimización lineal desde consola
(`solver.py`) o a través de una interfaz Streamlit (`streamlit_app.py`).

## Requisitos

Instala las dependencias listadas en `requirements.txt`. Incluyen `pyinstaller`,
que se usa únicamente durante el proceso de construcción del ejecutable CLI.

```bash
python -m pip install -r requirements.txt
```

## Uso interactivo de `solver.py`

Ejecuta directamente el asistente de consola:

```bash
python solver.py
```

## Construir el ejecutable

El script `build_executable.sh` automatiza la creación del binario `solver-cli`:

1. Crea un entorno virtual temporal.
2. Instala las dependencias del proyecto.
3. Ejecuta `pyinstaller solver.py --onefile --name solver-cli`.
4. Limpia artefactos intermedios (`build/` y `solver.spec`).

Para usarlo:

```bash
./build_executable.sh
```

El ejecutable quedará disponible en `dist/solver-cli`. Puedes verificarlo
mostrando su información o ejecutándolo directamente:

```bash
ls -lh dist/solver-cli
./dist/solver-cli
```

## Distribuir la app de Streamlit (opcional)

La app `streamlit_app.py` continúa ejecutándose mediante `streamlit run`. Si en el
futuro necesitas empaquetarla, puedes reutilizar la misma estrategia descrita
arriba cambiando el archivo objetivo en el comando de PyInstaller y documentando
cómo lanzar el servidor incluido en el binario resultante.
