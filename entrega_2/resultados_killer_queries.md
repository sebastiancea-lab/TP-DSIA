# B.6 — Killer Queries

Corridas sobre la colección `portalia_knowledge` (38 documentos, distancia coseno, embeddings
`gemini-embedding-001` a 768 dimensiones, vía `vector_db.py`). Umbral de aceptación (C.2):
`UMBRAL_ACEPTACION = 0.40`.

| # | Consulta | Qué pone a prueba | Resultado esperado | Resultado real | ¿Pasó? |
|---|---|---|---|---|---|
| 1 | "se me escrachó el celu recién comprado, ¿me lo cambian?" | Poder semántico: jerga rioplatense sin ninguna palabra exacta del documento ("escrachó", "celu" no aparecen en la base) | Recuperar las políticas sobre productos dañados/con falla, aunque el texto no coincida literalmente | Top 3: `POL-CAM-002` (0.3273, condiciones de cambio ante daño), `POL-GAR-001` (0.3514, garantía por falla), `POL-GAR-002` (0.3613, producto dañado). Los tres por debajo del umbral | ✅ Sí |
| 2 | "necesito un teléfono con red 5G y espacio para guardar muchos videos" (`--categoria celulares`) | El metadato salva el día: la semántica cruda recomienda un producto sin stock, el filtro nativo lo bloquea | Sin `--solo-con-stock`: el top 1 es un celular 5G sin stock. Con `--solo-con-stock`: ese producto queda excluido y aparece el siguiente celular 5G con stock | Sin filtro: top 1 `PROD-003` (0.2853, `en_stock_demo: False`). Con `--solo-con-stock`: top 1 pasa a `PROD-002` (0.3083, `en_stock_demo: True`); `PROD-003` no aparece | ✅ Sí |
| 3 | "cuánto sale alquilar un departamento en Palermo" | Prueba de estrés: consulta totalmente fuera del catálogo — debe responder "no tengo eso" | Ninguna distancia debe superar el umbral; el sistema responde `SIN_RESULTADOS` en vez de forzar el resultado más cercano | Mejor distancia encontrada: 0.4860 (`ENV-001`, envíos) — por encima del umbral 0.40. El sistema respondió: *"No tengo esa información en la base de conocimiento de Portalia."* | ✅ Sí |

## Cómo se calibró el umbral

Antes de fijar `UMBRAL_ACEPTACION` se corrieron consultas del dominio y consultas claramente
ajenas al catálogo con `--umbral 2.0` (que no descarta nada), para medir la distancia real de la
mejor coincidencia en cada caso:

| Tipo de consulta | Ejemplo | Mejor distancia |
|---|---|---|
| Dominio | "cuánto tiempo tengo para devolver un producto" | 0.2085 |
| Dominio | "qué garantía tienen los productos" | 0.2959 |
| Dominio | "quiero cambiar el color de mi celular" | 0.3314 |
| Fuera de catálogo | "cuánto sale alquilar un departamento en Palermo" | 0.4860 |
| Fuera de catálogo | "receta de milanesas napolitanas" | 0.4827 |
| Fuera de catálogo | "quién ganó el mundial de fútbol" | 0.5358 |

Las consultas del dominio no superan 0.33 de distancia; las consultas ajenas al catálogo nunca
bajan de 0.48. El umbral se fijó en el punto medio (0.40), con margen de sobra para ambos lados.

## Evidencia — salida cruda de la terminal

### KQ1 — Poder semántico

```text
$ uv run python entrega_2/vector_db.py "se me escracho el celu recien comprado, me lo cambian?"

--- BUSQUEDA HIBRIDA ---
Consulta: 'se me escracho el celu recien comprado, me lo cambian?'
Filtro nativo enviado a ChromaDB: {'activo': {'$eq': True}}

[1] ID: POL-CAM-002 (distancia: 0.3273)
Metadata: {'categoria': 'cambios', 'activo': True, 'tags_regionales': 'cambio, condiciones, accesorios'}
Texto: Condiciones del producto para realizar un cambio. Para evaluar un cambio, Portalia verifica
que el artículo corresponda al pedido y que se presente con los accesorios incluidos
originalmente. También revisa si tiene señales de uso o daños atribuibles al cliente antes de
aprobar la solicitud. Si el cliente informa que el equipo llegó dañado o presenta una falla, el
caso debe evaluarse por la vía de postventa correspondiente, sin rechazarlo automáticamente como
un cambio por preferencia.

[2] ID: POL-GAR-001 (distancia: 0.3514)
Texto: Garantía por posible falla de funcionamiento. [...]

[3] ID: POL-GAR-002 (distancia: 0.3613)
Texto: Producto recibido con daños visibles. [...]
```

### KQ2 — El metadato salva el día

```text
$ uv run python entrega_2/vector_db.py "necesito un telefono con red 5G y espacio para guardar muchos videos" --categoria celulares -n 3

--- BUSQUEDA HIBRIDA ---
Filtro nativo enviado a ChromaDB: {'$and': [{'categoria': {'$eq': 'celulares'}}, {'activo': {'$eq': True}}]}

[1] ID: PROD-003 (distancia: 0.2853)
Metadata: {'en_stock_demo': False, 'red_movil': '5G', 'almacenamiento_gb': 256, ...}
Texto: Celular Portalia Nova 5G 256 [...]

[2] ID: PROD-002 (distancia: 0.3083)
Metadata: {'en_stock_demo': True, 'red_movil': '5G', 'almacenamiento_gb': 128, ...}

[3] ID: PROD-001 (distancia: 0.3644)
Metadata: {'en_stock_demo': True, 'red_movil': '4G', ...}
```

```text
$ uv run python entrega_2/vector_db.py "necesito un telefono con red 5G y espacio para guardar muchos videos" --categoria celulares --solo-con-stock -n 3

--- BUSQUEDA HIBRIDA ---
Filtro nativo enviado a ChromaDB: {'$and': [{'categoria': {'$eq': 'celulares'}}, {'activo': {'$eq': True}}, {'en_stock_demo': {'$eq': True}}]}

[1] ID: PROD-002 (distancia: 0.3083)
Metadata: {'en_stock_demo': True, 'red_movil': '5G', 'almacenamiento_gb': 128, ...}
Texto: Celular Portalia Nova 5G 128 [...]

[2] ID: PROD-001 (distancia: 0.3644)
Metadata: {'en_stock_demo': True, 'red_movil': '4G', ...}
```

`PROD-003` (sin stock) desaparece de los resultados en la segunda corrida porque el filtro
`en_stock_demo: {"$eq": True}` viaja dentro del `where`, no como un `if` posterior sobre la lista
ya recuperada.

### KQ3 — Prueba de estrés

```text
$ uv run python entrega_2/vector_db.py "cuanto sale alquilar un departamento en Palermo"

--- BUSQUEDA HIBRIDA ---
Consulta: 'cuanto sale alquilar un departamento en Palermo'
Filtro nativo enviado a ChromaDB: {'activo': {'$eq': True}}
No tengo esa informacion en la base de conocimiento de Portalia. (mejor distancia: 0.4860)
```
