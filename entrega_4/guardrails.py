"""Entrega 4, Fase 5: guardrails deterministas de input y de output.

Son funciones puras en Python (sin LLM): predecibles, testeables y sin costo de tokens.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from entrega_4.config import MAX_LARGO_INPUT

# Tipos de bloqueo (claves de /metricas y de motivo_bloqueo).
INPUT_INJECTION = "input_injection"
INPUT_LONGITUD = "input_longitud"
OUTPUT_PII = "output_pii"

MENSAJES_BLOQUEO = {
    INPUT_INJECTION: (
        "No puedo procesar ese mensaje porque intenta modificar mis instrucciones. "
        "Si necesitás ayuda con un pedido, un cambio o una devolución, contame el detalle."
    ),
    INPUT_LONGITUD: (
        f"Tu mensaje está vacío o supera los {MAX_LARGO_INPUT} caracteres. "
        "Reformulalo de forma más breve."
    ),
}


@dataclass(frozen=True)
class Bloqueo:
    tipo: str
    mensaje: str


def _normalizar(texto: str) -> str:
    """Minúsculas y sin tildes, para que 'ignorá' e 'ignora' matcheen igual."""
    descompuesto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def validar_longitud(texto: str, maximo: int = MAX_LARGO_INPUT) -> Optional[Bloqueo]:
    """Bloquea mensajes vacíos o más largos que ``maximo`` caracteres."""
    if not isinstance(texto, str) or not texto.strip() or len(texto) > maximo:
        return Bloqueo(INPUT_LONGITUD, MENSAJES_BLOQUEO[INPUT_LONGITUD])
    return None


# Patrones sobre texto normalizado (sin tildes). Español e inglés.
_PATRONES_INJECTION = [re.compile(p) for p in (
    r"\b(ignora|ignore|olvida|olvidate|omiti|saltea|descarta|desestima)\w*\s+(\w+\s+){0,3}"
    r"(instrucciones|reglas|indicaciones|directivas|restricciones|prompt)",
    # "no hagas caso a tus instrucciones", "hace caso omiso de las reglas", "deja de lado tus reglas":
    # exigen el objeto para no bloquear "no hagas caso a lo que dije antes".
    r"\b(no\s+hagas\s+caso|hace\s+caso\s+omiso|deja\s+de\s+lado|dejale\s+de\s+lado)\s+"
    r"(\w+\s+){0,3}(instrucciones|reglas|indicaciones|directivas|restricciones|prompt)",
    r"\b(desde|a\s+partir\s+de)\s+ahora\s+(sos|eres|seras|vas\s+a\s+ser)\b",
    r"\b(ignore|disregard|forget|override|bypass)\s+(\w+\s+){0,3}"
    r"(instructions|rules|guidelines|prompt|restrictions)",
    r"\b(system|developer)\s+prompt\b",
    r"\bprompt\s+(del|de)\s+(sistema|desarrollador)\b",
    r"\binstrucciones\s+(del\s+)?sistema\b",
    r"\b(actua|comportate|responde|hace\s+de\s+cuenta|finge|fingi|simula)\w*\s+"
    r"(\w+\s+){0,2}(como|que\s+sos|que\s+eres)\s+(\w+\s+){0,2}"
    r"(otro|otra|dan|admin|administrador|root|desarrollador|sin\s+restricciones|"
    r"sin\s+reglas|sin\s+filtros)",
    r"\bact\s+as\b",
    r"\byou\s+are\s+now\b",
    r"\btus\s+nuevas\s+instrucciones\b",
    r"\bnuevas\s+instrucciones\s*:",
    r"\b(modo|mode)\s+(desarrollador|developer|dios|god|admin)\b",
    r"\bjailbreak\b",
    # Revelar el prompt: posesivo (tus/tu/your) o un objeto inequívoco.
    r"\b(revela|revelame|muestra|mostrame|imprime|repeti|decime|dame|reveal|show|print)\w*\s+"
    r"(\w+\s+){0,3}(tus|tu|your)\s+(\w+\s+){0,2}"
    r"(instrucciones|prompt|reglas|instructions|rules)",
    r"\b(cual\s+es|decime|dame|mostrame|revelame)\s+(\w+\s+){0,2}(tu|tus|your)\s+prompts?\b",
    r"\binstrucciones\s+(internas|originales|anteriores|iniciales)\b",
)]


def detectar_prompt_injection(texto: str) -> Optional[Bloqueo]:
    """Bloquea intentos evidentes de cambiar las reglas o de extraer el prompt."""
    normalizado = _normalizar(texto or "")
    for patron in _PATRONES_INJECTION:
        if patron.search(normalizado):
            return Bloqueo(INPUT_INJECTION, MENSAJES_BLOQUEO[INPUT_INJECTION])
    return None


def validar_input(texto: str) -> Optional[Bloqueo]:
    """Corre los validators de input en orden; devuelve el primer bloqueo."""
    return validar_longitud(texto) or detectar_prompt_injection(texto)


# --- Output: enmascarado de datos personales -------------------------------
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_TARJETA = re.compile(r"(?<![\w.])(?:\d[ -]?){13,16}(?!\w)")
_DNI = re.compile(
    r"(?<![\w.\-])(?:(?P<signo>\$\s*))?(?P<num>\d{2}\.\d{3}\.\d{3}|\d{7,8})(?!\w|\.\d)"
    r"(?P<pesos>\s+pesos\b)?",
    re.IGNORECASE,
)
_TELEFONO = re.compile(
    r"(?<![\w.\-])(?:\+\d{1,3}[\s-]?)?(?:\d{1,4}[\s-]){0,3}\d{3,4}[\s-]\d{4}(?!\w)"
    r"|(?<![\w.\-])\d{10,11}(?!\w)"
)
_MIN_DIGITOS_TELEFONO = 10


def _reemplazo_telefono(coincidencia: re.Match) -> str:
    """Enmascara solo con '+' o con 10+ dígitos: '2026 5120' o '5G 128 2024' no son teléfonos."""
    texto = coincidencia.group()
    digitos = sum(c.isdigit() for c in texto)
    if texto.startswith("+") or digitos >= _MIN_DIGITOS_TELEFONO:
        return "[TELÉFONO OCULTO]"
    return texto


def _reemplazo_dni(coincidencia: re.Match) -> str:
    """Un monto ('$ 1299999', '12.999.999 pesos') no es un DNI. Sin contexto, se enmascara."""
    if coincidencia.group("signo") or coincidencia.group("pesos"):
        return coincidencia.group()
    return "[DNI OCULTO]"


def _luhn_valido(candidato: str) -> bool:
    """Distingue una tarjeta real de un teléfono largo (ambos son tiras de 13+ dígitos)."""
    digitos = [int(c) for c in candidato if c.isdigit()]
    if not 13 <= len(digitos) <= 16:
        return False
    suma = 0
    for i, d in enumerate(reversed(digitos)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        suma += d
    return suma % 10 == 0


def _reemplazo_tarjeta(coincidencia: re.Match) -> str:
    return "[TARJETA OCULTA]" if _luhn_valido(coincidencia.group()) else coincidencia.group()


# El orden importa: tarjetas antes que teléfonos y DNI, que son subcadenas de dígitos.
_REGLAS_PII = (
    (_EMAIL, "[EMAIL OCULTO]", "email"),
    (_TARJETA, _reemplazo_tarjeta, "tarjeta"),
    (_TELEFONO, _reemplazo_telefono, "teléfono"),
    (_DNI, _reemplazo_dni, "dni"),
)


def enmascarar_pii(texto: str) -> tuple[str, list[str]]:
    """Devuelve ``(texto_enmascarado, tipos_detectados)``. Sin PII, el texto queda intacto."""
    tipos: list[str] = []
    for patron, reemplazo, tipo in _REGLAS_PII:
        antes = texto
        texto = patron.sub(reemplazo, texto)
        if texto != antes:
            tipos.append(tipo)
    return texto, tipos
