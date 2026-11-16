"""Entrega 4, Fase 1: backend mock y motor de reglas de Portalia.

La IA interpreta, el motor de reglas decide: la elegibilidad de un cambio o una
devolución se calcula acá, en Python, nunca en el LLM.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import ValidationError

from entrega_4.config import (
    ARCHIVO_BACKEND,
    DB_SOLICITUDES,
    PLAZO_CAMBIO_DIAS,
    PLAZO_DEVOLUCION_DIAS,
)
# Se reutiliza el validator del contrato de la Entrega 1 en vez de duplicar la regex.
from entrega_1.parte_c.schemas import ExtraccionPostventa

TipoSolicitud = Literal["cambio", "devolucion"]

# Plazos de las políticas de Entrega 2/4: el backend los aplica como reglas
# deterministas y no delega la decisión de elegibilidad al modelo.
PLAZOS_DIAS: dict[str, int] = {
    "cambio": PLAZO_CAMBIO_DIAS,          # POL-CAM-001
    "devolucion": PLAZO_DEVOLUCION_DIAS,  # POL-DEV-001
}
POLITICA_POR_TIPO = {"cambio": "POL-CAM-001", "devolucion": "POL-DEV-001"}


class ErrorBackend(ValueError):
    """Error de dominio con mensaje seguro para devolverle al agente."""


def normalizar_pedido_id(valor: object) -> int:
    """Normaliza '#48327' / 'pedido 48327' a int, con las reglas del contrato de la E1."""
    # Reutiliza el validador Pydantic de Entrega 1 para que las herramientas del
    # agente y el contrato de extracción acepten el mismo formato de pedido.
    try:
        pedido_id = ExtraccionPostventa(intencion="otro", pedido_id=valor).pedido_id
    except ValidationError as exc:
        detalle = exc.errors()[0]["msg"].removeprefix("Value error, ")
        raise ErrorBackend(f"Número de pedido inválido: {detalle}") from exc
    if pedido_id is None:
        raise ErrorBackend("Falta el número de pedido.")
    return pedido_id


def _normalizar_texto(valor: Optional[str]) -> str:
    # Comparación de texto de negocio sin diferencias por mayúsculas o espacios.
    return (valor or "").strip().lower()


class BackendPortalia:
    """Pedidos, stock y solicitudes. ``hoy`` se puede inyectar para tests deterministas."""

    def __init__(
        self,
        ruta_datos: Path = ARCHIVO_BACKEND,
        ruta_solicitudes: Path = DB_SOLICITUDES,
        hoy: Optional[date] = None,
    ) -> None:
        # Los pedidos y el stock son datos mock cargados en memoria; los tickets
        # sí se persisten en SQLite. hoy se puede inyectar para pruebas estables.
        self._datos = json.loads(Path(ruta_datos).read_text(encoding="utf-8"))
        self._ruta_solicitudes = Path(ruta_solicitudes)
        self._hoy = hoy
        self._inicializar_solicitudes()

    # -- fecha de referencia ----------------------------------------------
    def hoy(self) -> date:
        # Producción usa la fecha actual; los tests pueden fijar una fecha.
        return self._hoy or date.today()

    # -- pedidos ------------------------------------------------------------
    def obtener_pedido(self, pedido_id: object) -> dict:
        # El ID se normaliza antes de buscarlo porque las claves del JSON son texto.
        numero = normalizar_pedido_id(pedido_id)
        bruto = self._datos["pedidos"].get(str(numero))
        if bruto is None:
            raise ErrorBackend(f"No existe el pedido {numero}.")
        hoy = self.hoy()
        entregado_hace = bruto.get("entregado_hace_dias")
        estimada_en = bruto.get("entrega_estimada_en_dias")
        # Las fechas de este backend de demostración se calculan relativas a hoy,
        # y se devuelven copias para no exponer ni modificar el JSON interno.
        pedido = {
            "pedido_id": numero,
            "estado": bruto["estado"],
            "fecha_entrega": (
                (hoy - timedelta(days=entregado_hace)).isoformat()
                if entregado_hace is not None else None
            ),
            "fecha_entrega_estimada": (
                (hoy + timedelta(days=estimada_en)).isoformat()
                if estimada_en is not None else None
            ),
            "items": [dict(item) for item in bruto["items"]],
            "cliente": dict(bruto["cliente"]),
        }
        return pedido

    # -- elegibilidad (POL-CAM-001 / POL-DEV-001) ----------------------------
    def evaluar_elegibilidad(self, pedido: dict, tipo: TipoSolicitud) -> dict:
        # La decisión se toma con el estado del pedido y los días desde recepción:
        # una fecha de compra no reemplaza la fecha de entrega.
        plazo = PLAZOS_DIAS[tipo]
        base = {"tipo": tipo, "plazo_dias": plazo, "politica": POLITICA_POR_TIPO[tipo]}
        if pedido["estado"] == "cancelado":
            return {
                **base, "elegible": False, "dias_desde_entrega": None,
                "motivo": "El pedido fue cancelado y no admite cambios ni devoluciones.",
            }
        if pedido["estado"] != "entregado" or not pedido["fecha_entrega"]:
            return {
                **base, "elegible": False, "dias_desde_entrega": None,
                "motivo": (
                    f"El pedido está '{pedido['estado']}': el plazo de {tipo} corre "
                    "desde la recepción, así que todavía no se puede solicitar."
                ),
            }
        dias = (self.hoy() - date.fromisoformat(pedido["fecha_entrega"])).days
        # El límite es inclusivo: solo se rechaza cuando ya pasaron más días que
        # los permitidos por la política.
        if dias > plazo:
            return {
                **base, "elegible": False, "dias_desde_entrega": dias,
                "motivo": (
                    f"Pasaron {dias} días desde la recepción y el plazo para "
                    f"{tipo} es de {plazo} días corridos."
                ),
            }
        return {
            **base, "elegible": True, "dias_desde_entrega": dias,
            "motivo": f"Dentro del plazo de {plazo} días corridos ({dias} desde la recepción).",
        }

    def elegibilidad_pedido(self, pedido: dict) -> dict:
        # Devuelve ambas evaluaciones para que el agente pueda explicar qué
        # operación está dentro de plazo sin que el LLM calcule fechas.
        return {tipo: self.evaluar_elegibilidad(pedido, tipo) for tipo in PLAZOS_DIAS}

    # -- stock -----------------------------------------------------------------
    def consultar_stock(self, producto_id: str, variante: Optional[str] = None) -> dict:
        # El stock consultado aquí es el del backend mock, no una API de inventario real.
        producto = (producto_id or "").strip().upper()
        variantes = self._datos["stock"].get(producto)
        if variantes is None:
            raise ErrorBackend(f"No existe el producto {producto or '(vacío)'}.")
        if variante is None or not variante.strip():
            # Sin variante se informa el inventario de todas las opciones y si
            # alguna tiene unidades disponibles.
            return {
                "producto_id": producto,
                "variantes": dict(variantes),
                "disponible": any(cantidad > 0 for cantidad in variantes.values()),
            }
        clave = _normalizar_texto(variante)
        if clave not in variantes:
            raise ErrorBackend(
                f"El producto {producto} no tiene la variante '{variante}'. "
                f"Variantes: {', '.join(variantes)}."
            )
        cantidad = variantes[clave]
        return {
            "producto_id": producto, "variante": clave,
            "unidades": cantidad, "disponible": cantidad > 0,
        }

    # -- solicitudes (SQLite) ---------------------------------------------------
    def _conectar(self) -> sqlite3.Connection:
        # Centraliza la apertura de la base de tickets en la ruta configurada.
        return sqlite3.connect(self._ruta_solicitudes)

    def _inicializar_solicitudes(self) -> None:
        # La restricción UNIQUE impide registrar más de un ticket para el mismo
        # pedido, tipo de solicitud y producto.
        self._ruta_solicitudes.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conectar()) as conexion, conexion:
            conexion.execute(
                """CREATE TABLE IF NOT EXISTS solicitudes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pedido_id INTEGER NOT NULL,
                    tipo TEXT NOT NULL,
                    producto_id TEXT NOT NULL,
                    motivo TEXT NOT NULL,
                    variante_solicitada TEXT,
                    creada_en TEXT NOT NULL,
                    UNIQUE (pedido_id, tipo, producto_id)
                )"""
            )

    def _ticket(self, fila: tuple, ya_existia: bool) -> dict:
        # Convierte la fila SQLite a la respuesta pública que usa la herramienta.
        return {
            "ticket": f"SOL-{fila[0]:05d}", "pedido_id": fila[1], "tipo": fila[2],
            "producto_id": fila[3], "motivo": fila[4], "variante_solicitada": fila[5],
            "creada_en": fila[6], "ya_existia": ya_existia,
        }

    def registrar_solicitud(
        self,
        pedido_id: object,
        tipo: TipoSolicitud,
        motivo: str,
        variante_solicitada: Optional[str] = None,
        producto_id: Optional[str] = None,
    ) -> dict:
        """Valida con las reglas y persiste. Idempotente por pedido, tipo y producto.

        Lanza ErrorBackend con el motivo si la solicitud no es posible.
        """
        # El backend vuelve a validar toda operación de escritura aunque el agente
        # ya haya interpretado la intención y reunido los datos.
        if tipo not in PLAZOS_DIAS:
            raise ErrorBackend(f"Tipo de solicitud inválido: {tipo!r}.")
        motivo = (motivo or "").strip()
        if not motivo:
            raise ErrorBackend("Falta el motivo de la solicitud.")
        pedido = self.obtener_pedido(pedido_id)
        item = self._resolver_item(pedido, producto_id)

        # Idempotencia primero: reintentar algo ya registrado devuelve el ticket existente,
        # aunque después se haya agotado el stock.
        existente = self._buscar_ticket(pedido["pedido_id"], tipo, item["producto_id"])
        if existente:
            return self._ticket(existente, ya_existia=True)

        decision = self.evaluar_elegibilidad(pedido, tipo)
        if not decision["elegible"]:
            raise ErrorBackend(f"Solicitud rechazada: {decision['motivo']}")

        # Un cambio requiere una variante distinta y disponible; una devolución
        # no necesita variante y se guarda como tal.
        variante = _normalizar_texto(variante_solicitada) or None
        if tipo == "cambio":
            if variante is None:
                raise ErrorBackend("Para un cambio hace falta indicar la variante solicitada.")
            if variante == _normalizar_texto(item.get("variante")):
                raise ErrorBackend(
                    f"Solicitud rechazada: el pedido ya tiene la variante '{variante}'; "
                    "un cambio requiere una variante distinta."
                )
            stock = self.consultar_stock(item["producto_id"], variante)
            if not stock["disponible"]:
                raise ErrorBackend(
                    f"Solicitud rechazada: no hay stock de {item['producto_id']} "
                    f"en la variante '{variante}'."
                )
        else:
            variante = None

        creada = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._conectar()) as conexion, conexion:
            # OR IGNORE: si otra solicitud igual entró entre el chequeo y acá, no hay excepción
            # cruda; se relee el ticket que ganó la carrera.
            # Los placeholders mantienen los valores como parámetros SQL, no como
            # parte del texto de la consulta.
            cursor = conexion.execute(
                "INSERT OR IGNORE INTO solicitudes (pedido_id, tipo, producto_id, motivo, "
                "variante_solicitada, creada_en) VALUES (?, ?, ?, ?, ?, ?)",
                (pedido["pedido_id"], tipo, item["producto_id"], motivo, variante, creada),
            )
            insertada = cursor.rowcount == 1
            id_nuevo = cursor.lastrowid
        if insertada:
            fila = (id_nuevo, pedido["pedido_id"], tipo, item["producto_id"],
                    motivo, variante, creada)
            return self._ticket(fila, ya_existia=False)
        return self._ticket(
            self._buscar_ticket(pedido["pedido_id"], tipo, item["producto_id"]), ya_existia=True,
        )

    def _buscar_ticket(self, pedido_id: int, tipo: str, producto_id: str) -> Optional[tuple]:
        # La clave compuesta coincide con la regla de idempotencia de la tabla.
        with closing(self._conectar()) as conexion:
            return conexion.execute(
                "SELECT id, pedido_id, tipo, producto_id, motivo, variante_solicitada, creada_en "
                "FROM solicitudes WHERE pedido_id=? AND tipo=? AND producto_id=?",
                (pedido_id, tipo, producto_id),
            ).fetchone()

    @staticmethod
    def _resolver_item(pedido: dict, producto_id: Optional[str]) -> dict:
        # Si el cliente identifica un producto, se verifica que pertenezca al
        # pedido; si no y hay varios artículos, se pide aclaración en vez de adivinar.
        items = pedido["items"]
        if producto_id and producto_id.strip():
            buscado = producto_id.strip().upper()
            for item in items:
                if item["producto_id"] == buscado:
                    return item
            raise ErrorBackend(f"El producto {buscado} no pertenece al pedido {pedido['pedido_id']}.")
        if len(items) == 1:
            return items[0]
        ids = ", ".join(item["producto_id"] for item in items)
        raise ErrorBackend(
            f"El pedido {pedido['pedido_id']} tiene varios productos ({ids}): "
            "indicá a cuál se refiere la solicitud."
        )
