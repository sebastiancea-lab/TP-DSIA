# Entrega 4 — Seguimiento de fases

Plan completo en `PLAN_ENTREGA_4.md` (en esta misma carpeta). Consigna en `requerimientos.md`.

- [x] **Fase 0** — Dependencias (`uv add`), `requirements.txt` regenerado, `.env.example` con `DB_PATH`, `MAX_ITERACIONES`, `LOG_LEVEL`.
- [x] **Fase 1** — Backend mock y motor de reglas (`backend.py`, `datos/backend_portalia.json`).
- [x] **Fase 2** — 4 herramientas (`tools.py`): retriever de la E3, 2 sensores y 1 actuador.
- [x] **Fase 3** — Agente ReAct manual, memoria SQLite, prompts (`agente.py`, `memoria.py`, `prompts.py`).
- [x] **Fase 4** — Matriz de validación contra Gemini real (`validacion_agente.py`): 5 casos OK, exit 0.
- [x] **Fase 5** — Guardrails e `security_tests.py` (12 tests).
- [x] **Fase 6** — Logger JSON y métricas (`observabilidad.py`).
- [x] **Fase 7** — API REST y `test_agente_api.py` (24 tests, incluye reinicio).
- [x] **Fase 8** — README (sección Entrega 4) y archivos Docker (en `entrega_4/`).
  - [x] `docker compose -f entrega_4/docker-compose.yaml up --build -d` → `(healthy)` (verificado).
- [x] **Correcciones del análisis** (`ANALISIS_ENTREGA_4.md`, H1 a H12) con sus tests: 36 en total.
- [x] **Fase 9** — `informe.md` (con costos) y `ficha_tecnica.md`.
  - [ ] **Pendiente:** capturar las trazas de LangSmith en `entrega_4/langsmith_*.png` (matriz y sesión de demo).
  - [ ] **Pendiente:** confirmar el precio de Gemini en la página oficial.
- [ ] **Fase 10** — Preparar la defensa (guion abajo); ensayar con el equipo.

## Guion de la defensa (15 min)

**Presentación (3 min):** problema (postventa de Portalia), principio "la IA interpreta, el motor
de reglas decide", diagrama del grafo, las 4 herramientas.

**Demo en vivo (7 min):**
1. Política: "¿Cuántos días tengo para devolver un producto?" → usa `buscar_politicas_postventa`.
2. Estado de un pedido (48327), mostrando que la PII queda enmascarada si se pide.
3. Cambio elegible (pedido 51204 a color blanco) → `consultar_pedido` → `consultar_stock` → ticket.
4. Devolución vencida (50718) → el sistema rechaza con motivo; y un intento de injection → bloqueado.
5. Reiniciar el servidor y mostrar `/estado/{thread_id}` con la memoria intacta; seguir la conversación.
6. `/metricas` (consultas, p95, herramientas, bloqueos) y la traza en LangSmith.

**Preguntas (5 min), respuestas preparadas:**
- ¿Por qué un grafo manual? Límite de iteraciones explícito y defendible.
- ¿Qué pasa si el LLM miente? No puede aprobar: el actuador valida en Python.
- ¿Qué no cubren los guardrails? Paráfrasis de injection y nombres propios; ver limitaciones.
- ¿Cuánto cuesta? ≈ US$ 0,001 por consulta; domina la entrada.
- ¿Y en producción? Falta autenticación, checkpointer compartido y métricas persistentes.
