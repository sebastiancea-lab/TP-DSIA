"""C.2 -- Pipeline de extraccion de postventa de Portalia contra la API de Gemini.

Flujo (por cada input):

    ConsultaPostventa (B.5a)                       # request que entra al sistema
        v
    Gemini + SYSTEM_PROMPT + Structured Outputs    # el LLM interpreta -> JSON
        v
    ExtraccionPostventa.model_validate_json(...)   # Pydantic hace cumplir el contrato
        v
    campos extraidos y validados
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors, types
from pydantic import ValidationError

from schemas import (
    SYSTEM_PROMPT,
    ConsultaPostventa,
    ExtraccionPostventa,
    esquema_respuesta_gemini,
)

RAIZ = Path(__file__).resolve().parent.parent
MODELO_POR_DEFECTO = "gemini-3.6-flash"

# Caso guia del dominio (mismo pedido que la evidencia de A.2).
CONSULTA_DEMO = ConsultaPostventa(
    canal="whatsapp",
    texto_libre=(
        "Hola, soy Gisella. Mi pedido numero 48327 debia llegar ayer y todavia "
        "no lo recibi. Donde esta y cuando va a llegar?"
    ),
    adjuntos=[],
    timestamp=datetime(2026, 9, 5, 20, 30, 0),
)


class ErrorDeConfig(RuntimeError):
    """Falta una variable de entorno obligatoria."""
def cargar_config() -> tuple[str, str]:
    """Devuelve ``(api_key, modelo)`` leyendo ``<raiz>/.env``."""
    load_dotenv(RAIZ / ".env")
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        raise ErrorDeConfig(
            f"Falta LLM_API_KEY. Copia {RAIZ / '.env.example'} a {RAIZ / '.env'} "
            "y pone tu API key."
        )
    modelo = os.environ.get("LLM_MODEL", "").strip() or MODELO_POR_DEFECTO
    return api_key, modelo


def construir_cliente(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def pedir_a_gemini(cliente: genai.Client, modelo: str, consulta: ConsultaPostventa) -> str:
    respuesta = cliente.models.generate_content(
        model=modelo,
        contents=consulta.texto_libre,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            response_mime_type="application/json",
            response_schema=esquema_respuesta_gemini(),
        ),
    )
    json_crudo = (respuesta.text or "").strip()
    if not json_crudo:
        raise ValueError("Gemini devolvio una respuesta vacia (sin texto).")
    return json_crudo


def procesar(cliente: genai.Client, modelo: str, consulta: ConsultaPostventa) -> bool:
    """Corre un input por el pipeline e imprime el resultado. Devuelve True si valido."""
    print(f"Canal      : {consulta.canal}")
    print(f"Texto      : {consulta.texto_libre}")

    try:
        json_crudo = pedir_a_gemini(cliente, modelo, consulta)
    except genai_errors.APIError as exc:
        print(
            f"[api] Gemini rechazo la llamada (codigo {getattr(exc, 'code', '?')}): "
            f"{getattr(exc, 'message', exc)}",
            file=sys.stderr,
        )
        return False
    except (httpx.HTTPError, ConnectionError, TimeoutError, OSError) as exc:
        print(f"[red] Fallo de conexion con la API: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False

    print("JSON crudo del modelo:")
    try:
        print(json.dumps(json.loads(json_crudo), indent=2, ensure_ascii=False))
    except ValueError:
        print(json_crudo)

    try:
        extraccion = ExtraccionPostventa.model_validate_json(json_crudo)
    except ValidationError as exc:
        print("[contrato] La salida del modelo no paso la validacion Pydantic:", file=sys.stderr)
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(modelo)"
            print(f"  - {loc}: {err['msg']}", file=sys.stderr)
        return False
    except ValueError as exc:
        print(f"[respuesta] Salida no parseable de Gemini: {exc}", file=sys.stderr)
        return False

    print("Campos extraidos y validados:")
    for campo, valor in extraccion.model_dump().items():
        print(f"  {campo:<20}: {valor!r}")
    return True


def parsear_argumentos(argv: list[str]) -> list[ConsultaPostventa]:
    """``argv`` -> lista de ConsultaPostventa. ``--canal <valor>`` aplica a todos
    los textos de la corrida (default whatsapp). Sin textos -> la consulta demo."""
    canal = "whatsapp"
    textos: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--canal":
            if i + 1 >= len(argv):
                raise ErrorDeConfig("--canal necesita un valor (whatsapp o web).")
            canal = argv[i + 1]
            i += 2
        else:
            textos.append(argv[i])
            i += 1

    if not textos:
        return [CONSULTA_DEMO]

    consultas: list[ConsultaPostventa] = []
    for texto in textos:
        consultas.append(
            ConsultaPostventa(
                canal=canal,
                texto_libre=texto,
                adjuntos=[],
                timestamp=datetime.now(),
            )
        )
    return consultas


def main(argv: list[str]) -> int:
    try:
        api_key, modelo = cargar_config()
        consultas = parsear_argumentos(argv[1:])
    except ErrorDeConfig as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        # canal fuera del contrato B.5a, texto vacio, etc.
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(request)"
            print(f"[request] {loc}: {err['msg']}", file=sys.stderr)
        return 2

    cliente = construir_cliente(api_key)
    print(f"Modelo: {modelo}\n")

    validaron = 0
    for numero, consulta in enumerate(consultas, start=1):
        print(f"===== Input {numero}/{len(consultas)} =====")
        if procesar(cliente, modelo, consulta):
            validaron += 1
        print()

    print(f"{validaron}/{len(consultas)} validaron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
