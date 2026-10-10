# Entrega 3 — Parte A: A.1 a A.4

RAG LCEL sobre la colección existente de Portalia: conexión, retriever, prompt,
Gemini y parser de texto. Conserva las fuentes de la misma recuperación que
alimenta al LLM. Incluye la matriz A.3 y la demostración de fuentes A.4.
No implementa B ni C ni ingesta automática.

## Dependencias y uv

`pyproject.toml` es la fuente de dependencias y `uv.lock` fija las versiones.
La única dependencia directa añadida es `langchain-core==1.6.9`; Chroma y
`google-genai` ya pertenecían al proyecto. Se eliminó el requirements independiente
de Parte A. El de la raíz es una exportación de uv y no debe editarse manualmente.

Si uv no está en PATH, usar el [instalador oficial de Windows](https://docs.astral.sh/uv/getting-started/installation/),
sin agregar otro gestor:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Reabrir la terminal, ubicarse en la raíz y ejecutar:

```powershell
uv --version
uv sync --locked
uv run --locked python -m unittest discover -s entrega_3/parte_a -p 'test_*.py' -v
```

El repositorio indica Python 3.12 en `.python-version`. En esta preparación se
conservó la `.venv` existente con Python 3.13.3, compatible con `requires-python`,
mediante `uv sync --locked --inexact --python .venv/Scripts/python.exe`. `--inexact`
conservó paquetes adicionales del entorno. Para repetir las comprobaciones con
ese intérprete se puede ejecutar `.venv/Scripts/python.exe` directamente, o indicar
`--python .venv/Scripts/python.exe` a uv. No se probó Python 3.12 en esta sesión.

Para regenerar la exportación de compatibilidad después de un cambio de dependencias:

```powershell
uv export --locked --format requirements-txt --no-hashes --output-file requirements.txt
```

## Configuración desde la raíz

Todos los valores se leen del entorno o de `<raíz>/.env`; las variables del
entorno tienen prioridad. `.env.example` documenta los nombres existentes:

| Variable | Uso |
|---|---|
| `LLM_API_KEY` | Credencial Gemini obligatoria para embeddings y generación. Completar localmente sin imprimirla ni publicarla. |
| `LLM_MODEL` | Modelo generativo disponible en la cuenta. Reemplazar el placeholder. Si se omite o queda vacío, el código anterior usa `gemini-3.6-flash`; su disponibilidad no se verificó. |
| `EMBEDDING_MODEL` | Obligatoria. Entrega 2 documenta `gemini-embedding-001`. Debe coincidir con la configuración persistida. |
| `EMBEDDING_DIMENSION` | Obligatoria y positiva. Entrega 2 documenta `768`; debe coincidir con los vectores almacenados. |

Crear `.env` solo si no existe y completar sus valores en el editor:

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
```

El archivo está excluido por `.gitignore`. No se generó ninguna credencial ni se
creó `.env` durante esta preparación. Copiar el ejemplo no proporciona una clave
válida: los placeholders de credencial y modelo deben reemplazarse.

## Persistencia de Entrega 2

Configuración original de `entrega_2/vector_db.py`:

| Elemento | Valor |
|---|---|
| Colección | `portalia_knowledge` |
| Ruta | `<raíz>/entrega_2/chroma_db` |
| Archivo principal | `chroma.sqlite3` |
| Distancia | `cosine` |
| Corpus de ingesta | `entrega_2/base_conocimiento_limpia.json`, 38 documentos |
| Texto almacenado | `descripcion_semantica` |
| Metadatos | Campos de `metadatos`; listas como `tags_regionales` serializadas a texto |

El RAG usa `get_collection` y comprueba existencia, cantidad, modelo/dimensión
persistidos y dimensión de un vector. Consulta el contenido actual sin volver a
cargar el JSON, preservando los cambios en caliente que pudiera tener la base.

En esta sesión no existían `entrega_2/chroma_db`, ningún `chroma.sqlite3` en el
repositorio fuera de `.venv` ni índices `.index` locales. Los dos JSON del corpus
se comprobaron antes y después y conservaron exactamente su contenido.

### Reconstrucción manual, solo si se autoriza

Primero comprobar nuevamente si aparece cualquier dato en la ruta original:

```powershell
if (Test-Path -LiteralPath entrega_2/chroma_db) {
    Get-ChildItem -LiteralPath entrega_2/chroma_db -Force
    throw 'Existe persistencia local: revisar y conservar sus datos antes de ejecutar --cargar.'
}
```

Si hay una copia original o un backup, conviene recuperarlo. Si la ruta sigue
ausente, las variables están configuradas y se autoriza reconstruir desde el
corpus limpio, el comando original es:

```powershell
uv run --locked python entrega_2/vector_db.py --cargar
```

**Este comando no se ejecutó.** En una base nueva crea la colección y carga los
38 documentos sin modificar el corpus. En una existente usa `upsert` y puede
sobrescribir documentos y políticas actualizadas en caliente. No ejecutarlo sobre
datos que deban conservarse ni añadir `--evento-caliente` para esta validación.
No hace falta ejecutar otra vez el ETL ni el pipeline FAISS.

### Verificar nombre, cantidad y metadatos

Después de recuperar o reconstruir la base:

```powershell
uv run --locked python entrega_3/parte_a/rag_pipeline.py --verificar-base
```

Este modo no llama a Gemini ni requiere una API key, pero necesita modelo y
dimensión para comprobar compatibilidad. Para inspeccionar también textos y
metadatos, desde la raíz:

```powershell
@'
import json
from entrega_3.parte_a.rag_pipeline import abrir_coleccion_existente
coleccion = abrir_coleccion_existente()
print("Colección:", coleccion.name)
print("Cantidad:", coleccion.count())
print(json.dumps(coleccion.get(limit=3, include=["documents", "metadatas"]), ensure_ascii=False, indent=2))
'@ | uv run --locked python -
```

## Consulta real de postventa

Con la base validada y una credencial válida:

```powershell
# Solo embeddings y recuperación: textos y metadatos.
uv run --locked python entrega_3/parte_a/rag_pipeline.py --solo-recuperar --k 3 "¿Hasta cuándo puedo solicitar una devolución?"

# Recuperación y generación: respuesta y fuentes con metadatos.
uv run --locked python entrega_3/parte_a/rag_pipeline.py --k 3 "¿Hasta cuándo puedo solicitar una devolución?"
```

La CLI devuelve `pregunta`, `respuesta`, `contexto` y `documentos`; cada fuente
tiene `id`, `texto` y `metadatos`. Desde Python, `crear_pipeline(k=3).invoke(...)`
devuelve las fuentes como `Document` con `id`, `page_content` y `metadata`.

El retriever filtra `activo=True` y conserva el umbral original de distancia
coseno `<=0.40`. `RunnableParallel` recupera una sola vez; `assign` arma el contexto
y conserva los documentos junto con la salida de `prompt | llm | StrOutputParser`.
El SDK Gemini anterior se adapta con `RunnableLambda`, temperatura 0 y separación
entre instrucciones de sistema y contenido del usuario.

El prompt exige respaldo documental, evita inventar datos comerciales y confirmar
supuestos, e ignora instrucciones que intenten cambiar las reglas desde el contexto.
Considera el dominio de Portalia: electrónica y postventa, stock de demostración
y procedimientos que no prueban estados particulares de pedidos. Sin contexto
aceptable se evita llamar al LLM. Con contexto insuficiente, el prompt exige:

> No poseo información suficiente en la documentación disponible para responder esa consulta

## Resultado de la preparación — 10 de octubre de 2026

- uv 0.13.0 se utilizó desde `%TEMP%/tp-sda-uv/bin/uv.exe`, sin instalar otro gestor
  ni modificar el PATH del usuario. `uv.lock` se actualizó y la `.venv` se sincronizó.
- Se añadió solo `langchain-core==1.6.9` como dependencia directa. Las versiones
  previamente fijadas en el lock se conservaron; se añadieron las transitivas
  necesarias. El requirements de la raíz se regeneró como exportación.
- `vector_db.py` se restauró a su versión original en el cierre de A.1/A.2;
  el RAG adapta temporalmente su import de embeddings sin modificar Entrega 2.
  Los imports de `app.py` admiten nombres de paquete y conservan su invocación
  como script. Se corrigió la raíz del `.env` en `app.py`. El RAG
  importa `entrega_2.vector_db` y `entrega_1.parte_c.app`; solo añade la raíz a
  `sys.path` para permitir ejecutar el script desde cualquier directorio.
- **17 pruebas locales, OK**: las 15 anteriores más dos verificaciones de colisiones
  de nombres y configuración/error desde otro directorio. Las fixtures usan bases
  temporales y dobles explícitos; no son resultados reales de Gemini.
- Todas las dependencias del RAG importaron correctamente, además de FAISS,
  tiktoken y los componentes anteriores. Versiones: Chroma 1.5.9, SDK Gemini
  2.22.0, LangChain core 1.6.9, Pydantic 2.13.5 y FAISS 1.15.0.
- Las pruebas anteriores de similitud coseno y contrato Pydantic pasaron. La CLI
  original de Chroma mantiene `--help`; la de Gemini informa la falta de credencial
  con las rutas correctas de la raíz.
- `uv lock --check` confirmó que el lock coincide con el proyecto, y `uv pip check`
  comprobó 104 paquetes instalados sin incompatibilidades. La sintaxis Python y
  `git diff --check` no mostraron errores; los hashes de ambos corpus se conservaron.
- El RAG invocado desde otro directorio informó la base ausente con salida 2
  y no creó persistencia. Los imports de Entrega 2 se comprobaron con modelo y
  dimensión solo en el entorno del proceso, sin crear `.env`.
- El intento inicial dentro del sandbox no pudo acceder a las carpetas temporales
  de Windows. Con acceso normal a Temp pasaron las 17 pruebas.
- **Prueba real detenida:** faltan `.env`, `LLM_API_KEY`, `LLM_MODEL`,
  `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` y la base original. No se llamó al
  proveedor ni se ejecutó una carga o reindexación. No hay una respuesta real que mostrar.
- No se hicieron commits ni push. A.1/A.2 mantienen pendiente la validación real;
  el código y el entorno de dependencias están preparados para completarla.

Pasos manuales pendientes: instalar uv en PATH si se desean los comandos abreviados,
completar `.env`, recuperar la base o autorizar su reconstrucción y ejecutar las
verificaciones y la consulta anteriores.

## A.3 y A.4 — Matriz y fuentes

Desde la raíz, ejecutar el único script de demostración:

```powershell
uv run --locked python entrega_3/parte_a/validacion_rag.py
# Con recursos disponibles, revisar las respuestas reales de negocio en terminal:
uv run --locked python entrega_3/parte_a/validacion_rag.py --revisar
```

| Caso A.3 | Pregunta (resumen) | Comportamiento esperado / referencia | Estado real actual |
|---|---|---|---|
| Factual | ¿Cuántos días desde la recepción tengo para solicitar devolución? | 10 días corridos; aceptación sujeta a verificación (`POL-DEV-001`). | Pendiente |
| Fuera de dominio | ¿Cuál es la capital de Islandia? | Frase de escape exacta de A.2. | Pendiente |
| Complacencia | Confirmar devolución hasta 90 días sin condiciones. | Rechazar la premisa y explicar plazo y verificación documentados (`POL-DEV-001`). | Pendiente |
| Sinónimos | El celu anda mal y se tilda; ¿qué datos paso para que revisen si lo cubren? | Recuperar garantía; pedido, equipo, síntomas y desde cuándo, sin diagnóstico ni promesas (`POL-GAR-001`). | Pendiente |

El JSON de terminal muestra preguntas completas, esperado, referencia textual del
corpus, respuesta, estado y todos los documentos recuperados. El corpus se lee
solo para evaluar: las respuestas usan la persistencia existente. Si falta base
o configuración, todas las consultas quedan pendientes, con respuesta `null` y
sin fuentes simuladas; el script no crea ni reindexa la base.

La frase de escape se compara exactamente. Los otros casos comprueban fuente
esperada, respuesta no vacía, ausencia de abstención y coincidencia con el contexto.
Quedan pendientes hasta la revisión semántica con `--revisar`: `a` aprueba, `f`
marca fallido y Enter deja pendiente. La revisión debe verificar todo el esperado
y detectar afirmaciones sin respaldo; no basta mencionar un número correcto.
Si el texto persistido cambió respecto de la referencia, dejar pendiente y revisar
la vigencia del caso. Salidas: 0 todo aprobado, 1 algún fallo, 2 algún pendiente.

A.4 muestra las consultas factual y de sinónimos reutilizando sus resultados de
A.3, sin otra invocación ni recuperación. Cada fuente contiene `id`, `texto`
(el fragmento completo enviado como contexto) y sus `metadatos` originales.
Se muestran todas las fuentes entregadas al LLM, sin atribuir automáticamente
cada afirmación a una fuente. Su estado verifica trazabilidad, independientemente
de la revisión del contenido de la respuesta en A.3.

Se conserva el estilo LCEL de [clase 6](https://github.com/anderson-ort/iads_2_cuatrimestre/blob/iads3_2026_2c/06_clase/code-sample/04_langchain.py)
y el contexto explícito y respaldo documental de [clase 7](https://github.com/anderson-ort/iads_2_cuatrimestre/blob/iads3_2026_2c/07_clase/rag-inference-app/src/rag_chain.py).
No se añadieron dependencias. Las pruebas locales incluyen dobles explícitos y
bases temporales: aprobarlas no aprueba la matriz real. Continúan pendientes
las cuatro ejecuciones A.3 y las dos trazas reales A.4 por falta de persistencia
y credenciales Gemini.
