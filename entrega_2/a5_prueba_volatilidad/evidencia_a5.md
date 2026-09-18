# A.5 — Prueba de volatilidad de la RAM

Prueba repetida el 16/09/2026 con cuatro procesos Python independientes (un PID distinto por
fase, verificable en cada bloque de salida). Se usaron los 38 documentos de
`base_conocimiento_limpia.json` (la salida del ETL de B.5), el modelo `gemini-embedding-001` y
vectores de 768 dimensiones. El índice de prueba `a5_prueba_volatilidad/a5_prueba.index` está
separado del índice principal `indices/portalia.index` de A.4 y se borra después de cada corrida
completa (está en `.gitignore`).

## 1. Construcción solo en RAM

Comando desde la raíz del proyecto: `uv run python entrega_2/a5_prueba_volatilidad/prueba_volatilidad.py ram`

```text
Fase: ram | proceso: 12744 | UTC: 2026-09-16T22:36:17.060334+00:00
Embeddings del catálogo generados por API: 38
Vectores en RAM: 38
faiss.write_index llamado: no
Archivo de prueba en disco: False
Al finalizar este proceso, los vectores de RAM dejan de estar disponibles.
```

## 2. Nuevo proceso: no hay índice que recargar

Comando desde la raíz del proyecto: `uv run python entrega_2/a5_prueba_volatilidad/prueba_volatilidad.py verificar-ram`

```text
Fase: verificar-ram | proceso: 16692 | UTC: 2026-09-16T22:36:27.872297+00:00
Archivo de prueba en disco: False
No hay índice que recargar en este proceso nuevo.
Para reconstruir los vectores habría que volver a llamar a la API.
```

## 3. Reconstrucción y persistencia

Comando desde la raíz del proyecto: `uv run python entrega_2/a5_prueba_volatilidad/prueba_volatilidad.py guardar`

```text
Fase: guardar | proceso: 20156 | UTC: 2026-09-16T22:36:35.536207+00:00
Embeddings del catálogo generados por API: 38
Vectores en RAM: 38
faiss.write_index llamado: sí
Archivo de prueba en disco: True
```

El segundo lote de 38 embeddings se generó porque la primera construcción solo existió en la RAM
del proceso terminado (fase 1): ese proceso ya no existe, así que sus vectores no están
disponibles para este proceso nuevo.

## 4. Nuevo proceso: recarga desde disco

Comando desde la raíz del proyecto: `uv run python entrega_2/a5_prueba_volatilidad/prueba_volatilidad.py recargar`

```text
Fase: recargar | proceso: 2312 | UTC: 2026-09-16T22:36:45.666960+00:00
Vectores recuperados desde disco: 38
Embeddings del catálogo generados por API en este proceso: 0
faiss.read_index llamado: sí
```

Este cuarto proceso recupera los 38 vectores guardados por la fase 3 sin volver a llamar a la API:
`faiss.read_index()` los reconstruye directamente desde `a5_prueba.index`.

## Reflexión

Si el servidor se reinicia y el índice solo estaba en RAM, debe regenerar los embeddings; con
`write_index` puede recargar los vectores guardados sin volver a generarlos.

Con dos servidores, cada uno necesita acceso a la misma versión del índice y del catálogo, o un
mecanismo coordinado de actualización, para evitar respuestas basadas en estados diferentes.
