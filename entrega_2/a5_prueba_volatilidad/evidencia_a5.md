# A.5 — Prueba de volatilidad de la RAM

Prueba repetida el 16/09/2026 con cuatro procesos Python independientes y pausas reales entre ellos. Se usaron los 38 documentos de `base_conocimiento.json`, el modelo `gemini-embedding-001` y vectores de 768 dimensiones. El índice de prueba `a5_prueba_volatilidad/a5_prueba.index` está separado del índice principal `indices/portalia.index` de A.4. Los intervalos entre los inicios de las fases fueron 2:27, 2:29 y 2:43 minutos.

## 1. Construcción solo en RAM

Comando desde la raíz del proyecto: `.\.venv\Scripts\python.exe .\entrega_2\a5_prueba_volatilidad\prueba_volatilidad.py ram`

```text
Fase: ram | proceso: 25016 | UTC: 2026-09-16T19:30:02.231428+00:00
Embeddings del catálogo generados por API: 38
Vectores en RAM: 38
faiss.write_index llamado: no
Archivo de prueba en disco: False
Al finalizar este proceso, los vectores de RAM dejan de estar disponibles.
```

## 2. Nuevo proceso: no hay índice que recargar

Comando desde la raíz del proyecto: `.\.venv\Scripts\python.exe .\entrega_2\a5_prueba_volatilidad\prueba_volatilidad.py verificar-ram`

```text
Fase: verificar-ram | proceso: 31104 | UTC: 2026-09-16T19:32:28.876624+00:00
Archivo de prueba en disco: False
No hay índice que recargar en este proceso nuevo.
Para reconstruir los vectores habría que volver a llamar a la API.
```

## 3. Reconstrucción y persistencia

Comando desde la raíz del proyecto: `.\.venv\Scripts\python.exe .\entrega_2\a5_prueba_volatilidad\prueba_volatilidad.py guardar`

```text
Fase: guardar | proceso: 41352 | UTC: 2026-09-16T19:34:57.849411+00:00
Embeddings del catálogo generados por API: 38
Vectores en RAM: 38
faiss.write_index llamado: sí
Archivo de prueba en disco: True
```

El segundo lote de 38 embeddings se generó porque la primera construcción solo existió en la RAM del proceso terminado.

## 4. Nuevo proceso: recarga desde disco

Comando desde la raíz del proyecto: `.\.venv\Scripts\python.exe .\entrega_2\a5_prueba_volatilidad\prueba_volatilidad.py recargar`

```text
Fase: recargar | proceso: 52804 | UTC: 2026-09-16T19:37:40.879765+00:00
Vectores recuperados desde disco: 38
Embeddings del catálogo generados por API en este proceso: 0
faiss.read_index llamado: sí
```
