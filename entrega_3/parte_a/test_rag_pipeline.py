"""Pruebas locales: bases temporales y dobles explícitos, sin credenciales ni red."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import rag_pipeline as rag
import validacion_rag as validacion


class ErroresLocalesTest(unittest.TestCase):
    def test_dependencia_ausente(self):
        with patch.object(rag.importlib, "import_module", side_effect=ModuleNotFoundError):
            with self.assertRaisesRegex(rag.ErrorRAG, "dependencia langchain_core"):
                rag.importar_dependencia("langchain_core")

    def test_base_ausente_no_crea_directorio_ni_cliente(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "base_ausente"
            anterior = SimpleNamespace(DIRECTORIO_CHROMA=ruta)
            with patch.object(rag, "cargar_entrega2", return_value=anterior):
                with patch.object(rag, "importar_dependencia") as importar:
                    with self.assertRaisesRegex(rag.ErrorRAG, "No existe la base persistida"):
                        rag.abrir_coleccion_existente()
                    importar.assert_not_called()
            self.assertFalse(ruta.exists())

    def test_pregunta_y_k_invalidos(self):
        for pregunta in ("", "  ", None, 123):
            with self.subTest(pregunta=pregunta):
                with self.assertRaisesRegex(rag.ErrorRAG, "texto no vacío"):
                    rag.validar_pregunta(pregunta)
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k):
                with self.assertRaisesRegex(rag.ErrorRAG, "entero positivo"):
                    rag.crear_retriever(None, k)


class ChromaYLcelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configuración de prueba explícita; no son credenciales ni una base original.
        cls.entorno = patch.dict(os.environ, {
            "EMBEDDING_MODEL": "gemini-embedding-001", "EMBEDDING_DIMENSION": "768",
        })
        cls.entorno.start()
        cls.addClassCleanup(cls.entorno.stop)
        cls.anterior = rag.cargar_entrega2()
        cls.lcel = rag.importar_dependencia("langchain_core.runnables")
        cls.Document = rag.importar_dependencia("langchain_core.documents").Document
        cls.chromadb = rag.importar_dependencia("chromadb")

    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.ruta = Path(self.temporal.name)
        self.ruta_parche = patch.object(self.anterior, "DIRECTORIO_CHROMA", self.ruta)
        self.ruta_parche.start()
        self.addCleanup(self.ruta_parche.stop)
        self.cliente = self.chromadb.PersistentClient(path=str(self.ruta))
        # Windows no permite limpiar los índices mientras Chroma mantiene handles.
        # Solo se detiene el sistema de esta fixture temporal, nunca la base original.
        self.addCleanup(self.cliente._system.stop)
        from chromadb.api.types import EmbeddingFunction

        class EmbeddingsFixture(EmbeddingFunction):
            def __init__(self):
                pass

            def __call__(self, input):
                raise AssertionError("La fixture no debe llamar a una API ni vectorizar textos.")

            @staticmethod
            def name():
                return "portalia_gemini"

            def get_config(self):
                return {"modelo": "gemini-embedding-001", "dimension": 768}

            @staticmethod
            def build_from_config(config):
                return EmbeddingsFixture()

        self.ef = EmbeddingsFixture()
        self.vector = [1.0] + [0.0] * 767
        self.vector_lejano = [0.0, 1.0] + [0.0] * 766

    def crear_fixture(self, documentos=True):
        coleccion = self.cliente.create_collection(
            self.anterior.NOMBRE_COLECCION,
            metadata={"hnsw:space": "cosine"}, embedding_function=self.ef,
        )
        if documentos:
            coleccion.add(
                ids=["ACTIVO", "INACTIVO", "LEJANO"],
                documents=["Política explícita de prueba.", "Política vencida.", "Texto no relevante."],
                metadatas=[
                    {"activo": True, "categoria": "cambios", "tags_regionales": "prueba"},
                    {"activo": False, "categoria": "cambios"},
                    {"activo": True, "categoria": "faq"},
                ],
                embeddings=[self.vector, self.vector, self.vector_lejano],
            )
        return coleccion

    def test_abre_coleccion_existente_sin_credenciales_y_sin_cambiar_documentos(self):
        original = self.crear_fixture()
        antes = original.get(include=["documents", "metadatas", "embeddings"])
        with patch.dict(os.environ, {}, clear=True):
            abierta = rag.abrir_coleccion_existente()
        self.assertEqual(abierta.name, self.anterior.NOMBRE_COLECCION)
        self.assertEqual(abierta.count(), 3)
        despues = abierta.get(include=["documents", "metadatas", "embeddings"])
        for campo in ("ids", "documents", "metadatas"):
            self.assertEqual(antes[campo], despues[campo])
        self.assertTrue((antes["embeddings"] == despues["embeddings"]).all())

    def test_coleccion_inexistente_no_se_crea(self):
        self.cliente.create_collection("otra_coleccion", embedding_function=None)
        with self.assertRaisesRegex(rag.ErrorRAG, "No se pudo abrir la colección existente"):
            rag.abrir_coleccion_existente()
        self.assertEqual([c.name for c in self.cliente.list_collections()], ["otra_coleccion"])

    def test_coleccion_vacia_se_informa(self):
        self.crear_fixture(documentos=False)
        with self.assertRaisesRegex(rag.ErrorRAG, "está vacía"):
            rag.abrir_coleccion_existente()

    def test_modelo_incompatible_aunque_dimension_igual(self):
        self.crear_fixture()
        with patch.object(self.anterior, "MODELO_EMBEDDINGS", "otro-modelo"):
            with self.assertRaisesRegex(rag.ErrorRAG, "no coinciden"):
                rag.abrir_coleccion_existente()

    def test_dimension_configurada_incompatible(self):
        self.crear_fixture()
        with patch.object(self.anterior, "DIMENSION_EMBEDDINGS", 1024):
            with self.assertRaisesRegex(rag.ErrorRAG, "no coinciden"):
                rag.abrir_coleccion_existente()

    def test_vector_almacenado_incompatible(self):
        coleccion = self.crear_fixture(documentos=False)
        coleccion.add(ids=["MAL"], documents=["Prueba"], embeddings=[[1.0, 0.0]])
        with self.assertRaisesRegex(rag.ErrorRAG, "vector almacenado"):
            rag.abrir_coleccion_existente()

    def test_credencial_ausente_se_informa(self):
        coleccion = self.crear_fixture()
        app = rag.importar_dependencia("entrega_1.parte_c.app")
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv", return_value=False):
            # app conserva su propio alias de load_dotenv si ya fue importado.
            with patch.object(app, "load_dotenv", return_value=False):
                with self.assertRaisesRegex(rag.ErrorRAG, "Falta LLM_API_KEY"):
                    rag.crear_retriever(coleccion)
                with self.assertRaisesRegex(rag.ErrorRAG, "Falta LLM_API_KEY"):
                    rag.crear_llm()

    def test_adaptador_gemini_reutiliza_modelo_y_separa_instrucciones(self):
        # Doble del SDK: valida el request sin generar una respuesta real ni usar keys.
        from langchain_core.prompts import ChatPromptTemplate
        cliente = Mock()
        cliente.models.generate_content.return_value = SimpleNamespace(text="Texto del doble SDK")
        app = rag.importar_dependencia("entrega_1.parte_c.app")
        with patch.object(app, "cargar_config", return_value=("", "modelo-de-prueba")):
            with patch.object(app, "construir_cliente", return_value=cliente):
                llm = rag.crear_llm()
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Reglas de prueba"), ("human", "Contexto y pregunta de prueba"),
                ]).invoke({})
                self.assertEqual(llm.invoke(prompt), "Texto del doble SDK")
        kwargs = cliente.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "modelo-de-prueba")
        self.assertEqual(kwargs["contents"], "Contexto y pregunta de prueba")
        self.assertEqual(kwargs["config"].system_instruction, "Reglas de prueba")
        self.assertEqual(kwargs["config"].temperature, 0)

    def test_retriever_real_filtra_vigencia_umbral_y_respeta_k(self):
        coleccion = self.crear_fixture()
        import numpy as np
        with patch.object(self.anterior, "crear_cliente", return_value=object()):
            with patch.object(self.anterior, "obtener_embeddings", return_value=np.array([self.vector])) as embed:
                retriever = rag.crear_retriever(coleccion, k=3)
                documentos = retriever.invoke("Consulta de prueba")
                self.assertEqual([d.id for d in documentos], ["ACTIVO"])
                self.assertEqual(documentos[0].page_content, "Política explícita de prueba.")
                self.assertEqual(documentos[0].metadata["tags_regionales"], "prueba")
                embed.assert_called_once()
                with patch.object(coleccion, "query", wraps=coleccion.query) as query:
                    rag.crear_retriever(coleccion, k=1).invoke("Otra pregunta")
                    self.assertEqual(query.call_args.kwargs["n_results"], 1)
                    self.assertEqual(query.call_args.kwargs["where"], {"activo": {"$eq": True}})

    def test_ventana_predeterminada_incluye_politicas_complementarias_relevantes(self):
        coleccion = self.crear_fixture()
        ids = [f"COMPLEMENTARIA-{i}" for i in range(1, 5)]
        coleccion.add(
            ids=ids, documents=[f"Política complementaria de prueba {i}." for i in range(1, 5)],
            metadatas=[{"activo": True, "categoria": "prueba"} for _ in ids],
            embeddings=[[1.0, i * 0.05] + [0.0] * 766 for i in range(1, 5)],
        )
        import numpy as np
        with patch.object(self.anterior, "crear_cliente", return_value=object()):
            with patch.object(self.anterior, "obtener_embeddings", return_value=np.array([self.vector])):
                limitada = rag.crear_retriever(coleccion, k=3).invoke("Consulta amplia de prueba")
                ampliada = rag.crear_retriever(coleccion).invoke("Consulta amplia de prueba")
        self.assertNotIn(ids[-1], [d.id for d in limitada])
        self.assertEqual({d.id for d in ampliada}, {"ACTIVO", *ids})
        self.assertLessEqual(len(ampliada), rag.K_POR_DEFECTO)
        # La ampliación conserva la exclusión por vigencia y distancia.
        self.assertNotIn("INACTIVO", [d.id for d in ampliada])
        self.assertNotIn("LEJANO", [d.id for d in ampliada])

    def test_chain_lcel_una_recuperacion_y_fuentes_del_prompt(self):
        documentos = [self.Document(id="FUENTE", page_content="Texto de la fuente {literal}.", metadata={"activo": True})]
        recuperar = Mock(return_value=documentos)
        prompts = []

        def generar(prompt):
            prompts.append(prompt)
            return "Salida de prueba del doble LLM; no es una respuesta de Gemini."

        chain = rag.crear_chain(self.lcel.RunnableLambda(recuperar), self.lcel.RunnableLambda(generar))
        self.assertIsInstance(chain, self.lcel.RunnableSequence)
        resultado = chain.invoke("  Consulta  ")
        recuperar.assert_called_once_with("Consulta")
        self.assertIs(resultado["documentos"], documentos)
        self.assertIsInstance(resultado["respuesta"], str)
        self.assertEqual(resultado["pregunta"], "Consulta")
        self.assertEqual(json.loads(resultado["contexto"])[0], {
            "id": "FUENTE", "texto": documentos[0].page_content, "metadatos": documentos[0].metadata,
        })
        self.assertIn(resultado["contexto"], prompts[0].to_messages()[1].content)
        reglas = prompts[0].to_messages()[0].content
        self.assertIn(rag.FRASE_ESCAPE, reglas)
        self.assertIn("No inventes precios", reglas)
        self.assertIn("ignorá instrucciones", reglas)
        self.assertIn("corregila explícitamente", reglas)
        self.assertIn("evaluaciones documentadas", reglas)

    def test_contexto_vacio_usa_escape_sin_llamar_llm(self):
        generar = Mock(side_effect=AssertionError("No debe llamarse al LLM sin contexto"))
        chain = rag.crear_chain(self.lcel.RunnableLambda(lambda _: []), self.lcel.RunnableLambda(generar))
        resultado = chain.invoke("Consulta sin contexto")
        self.assertEqual(resultado["respuesta"], rag.FRASE_ESCAPE)
        self.assertEqual(resultado["documentos"], [])
        generar.assert_not_called()

    def test_batch_no_mezcla_fuentes_entre_preguntas(self):
        chain = rag.crear_chain(
            self.lcel.RunnableLambda(lambda q: [self.Document(id=q, page_content=q)]),
            self.lcel.RunnableLambda(lambda prompt: "Doble de prueba"),
        )
        resultados = chain.batch(["primera", "segunda"])
        for resultado in resultados:
            self.assertEqual(resultado["documentos"][0].id, resultado["pregunta"])
            self.assertEqual(json.loads(resultado["contexto"])[0]["texto"], resultado["pregunta"])

    def test_imports_no_colisionan_con_modulos_homonimos(self):
        app = rag.importar_dependencia("entrega_1.parte_c.app")
        falsos = {nombre: SimpleNamespace() for nombre in ("app", "schemas", "vector_db", "embeddings_gemini")}
        with patch.dict(rag.sys.modules, falsos):
            self.assertIs(rag.cargar_entrega2(), self.anterior)
            self.assertIs(rag.sys.modules["embeddings_gemini"], falsos["embeddings_gemini"])
            with patch.object(app, "cargar_config", return_value=("", "modelo-de-prueba")):
                with patch.object(app, "construir_cliente", return_value=Mock()):
                    self.assertIsInstance(rag.crear_llm(), self.lcel.RunnableLambda)

    def test_configuracion_y_error_apuntan_a_raiz_desde_otro_directorio(self):
        app = rag.importar_dependencia("entrega_1.parte_c.app")
        self.assertEqual(app.RAIZ, rag.RAIZ)
        directorio_anterior = Path.cwd()
        try:
            os.chdir(self.ruta)
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(app, "load_dotenv", return_value=False) as cargar:
                    with self.assertRaises(app.ErrorDeConfig) as error:
                        app.cargar_config()
                    cargar.assert_called_once_with(rag.RAIZ / ".env")
                    self.assertIn(str(rag.RAIZ / ".env"), str(error.exception))
                    self.assertIn(str(rag.RAIZ / ".env.example"), str(error.exception))
        finally:
            os.chdir(directorio_anterior)


class ResilienciaYTrazabilidadTest(unittest.TestCase):
    """Lógica local con dobles explícitos: no aprueba la matriz real de Portalia."""

    @classmethod
    def setUpClass(cls):
        cls.casos = {c["caso"]: c for c in validacion.crear_casos()}
        cls.Document = rag.importar_dependencia("langchain_core.documents").Document

    def resultado(self, nombre, respuesta="Respuesta del doble unitario; no es Gemini."):
        caso = self.casos[nombre]
        docs = [] if not caso["documento_esperado"] else [self.Document(
            id=caso["documento_esperado"], page_content=caso["referencia_corpus"],
            metadata={"activo": True, "categoria": "prueba", "tags_regionales": "prueba"},
        )]
        return {"respuesta": respuesta, "documentos": docs, "contexto": rag.formatear_documentos(docs)}

    def test_cuatro_casos_referenciados_al_corpus_real(self):
        self.assertEqual(set(self.casos), {"factual", "fuera_dominio", "complacencia", "sinonimos"})
        ruta = rag.RAIZ / "entrega_2" / "base_conocimiento_limpia.json"
        corpus = {d["id"]: d for d in json.loads(ruta.read_text(encoding="utf-8"))}
        for caso in self.casos.values():
            fuente = caso["documento_esperado"]
            if fuente:
                self.assertEqual(caso["referencia_corpus"], corpus[fuente]["descripcion_semantica"])
                self.assertTrue(corpus[fuente]["metadatos"]["activo"])
        self.assertIn("10 días corridos", self.casos["factual"]["referencia_corpus"])
        self.assertIn("90 días", self.casos["complacencia"]["pregunta"])
        self.assertNotIn("90 días", self.casos["complacencia"]["referencia_corpus"])
        self.assertIn("celu", self.casos["sinonimos"]["pregunta"])

    def test_factual_exige_fuente_respuesta_y_revision_semantica(self):
        caso = self.casos["factual"]
        resultado = self.resultado("factual")
        self.assertEqual(validacion.evaluar_caso(caso, resultado)["estado"], "pendiente")
        self.assertEqual(validacion.evaluar_caso(caso, resultado, revision=True)["estado"], "aprobado")
        self.assertEqual(validacion.evaluar_caso(caso, resultado, revision=False)["estado"], "fallido")
        for respuesta in (None, "", "  ", rag.FRASE_ESCAPE):
            with self.subTest(respuesta=respuesta):
                self.assertEqual(validacion.evaluar_caso(caso, self.resultado("factual", respuesta))["estado"], "fallido")

    def test_fuera_de_dominio_exige_escape_exacto(self):
        caso = self.casos["fuera_dominio"]
        for respuesta, estado in (
            (rag.FRASE_ESCAPE, "aprobado"),
            (rag.FRASE_ESCAPE + ".", "fallido"),
            (" " + rag.FRASE_ESCAPE, "fallido"),
            ("Reikiavik (respuesta del doble).", "fallido"),
        ):
            with self.subTest(respuesta=respuesta):
                self.assertEqual(validacion.evaluar_caso(caso, self.resultado("fuera_dominio", respuesta))["estado"], estado)

    def test_complacencia_no_aprueba_por_palabras_o_fuente_correcta(self):
        caso = self.casos["complacencia"]
        falso = self.resultado("complacencia", "Sí, tenés 90 días sin condiciones; también se mencionan 10 días.")
        self.assertEqual(validacion.evaluar_caso(caso, falso)["estado"], "pendiente")
        self.assertEqual(validacion.evaluar_caso(caso, falso, revision=False)["estado"], "fallido")
        correcto = self.resultado("complacencia", "No: son 10 días corridos desde la recepción, sujetos a verificación.")
        self.assertEqual(validacion.evaluar_caso(caso, correcto, revision=True)["estado"], "aprobado")

    def test_sinonimos_exige_politica_de_garantia(self):
        caso = self.casos["sinonimos"]
        correcto = self.resultado("sinonimos")
        self.assertEqual(validacion.evaluar_caso(caso, correcto, revision=True)["estado"], "aprobado")
        incorrecto = self.resultado("factual")
        self.assertEqual(validacion.evaluar_caso(caso, incorrecto, revision=True)["estado"], "fallido")

    def test_fuentes_preservan_texto_completo_id_y_metadatos_sin_agregar_campos(self):
        resultado = self.resultado("factual")
        documento = resultado["documentos"][0]
        documento.metadata = {"categoria": "devoluciones", "activo": True, "tags_regionales": "devolución, plazo, 10 días"}
        resultado["contexto"] = rag.formatear_documentos(resultado["documentos"])
        fuentes = validacion.extraer_fuentes(resultado)
        self.assertEqual(fuentes, [{"id": documento.id, "texto": documento.page_content, "metadatos": documento.metadata}])
        self.assertNotIn("sucursal", fuentes[0]["metadatos"])
        self.assertEqual(resultado["documentos"][0].metadata, fuentes[0]["metadatos"])

    def test_fuentes_distintas_del_contexto_fallan_a3_y_a4(self):
        resultado = self.resultado("factual")
        resultado["documentos"][0].metadata["categoria"] = "alterada después del prompt"
        with self.assertRaisesRegex(rag.ErrorRAG, "no coinciden"):
            validacion.extraer_fuentes(resultado)
        fila = {**self.casos["factual"], "respuesta": resultado["respuesta"], **validacion.evaluar_caso(self.casos["factual"], resultado)}
        self.assertEqual(fila["estado"], "fallido")
        self.assertEqual(validacion.crear_trazas([fila])[0]["estado"], "fallido")

    def test_a4_reutiliza_dos_resultados_sin_segunda_recuperacion(self):
        from langchain_core.runnables import RunnableLambda
        por_pregunta = {c["pregunta"]: self.resultado(nombre) for nombre, c in self.casos.items()}
        recuperar = Mock(side_effect=lambda pregunta: por_pregunta[pregunta]["documentos"])
        prompts = []

        def generar(prompt):
            prompts.append(prompt)
            return "Respuesta del doble unitario; no es Gemini."

        chain = rag.crear_chain(RunnableLambda(recuperar), RunnableLambda(generar))
        matriz = [validacion.ejecutar_caso(chain, caso) for caso in self.casos.values()]
        trazas = validacion.crear_trazas(matriz)
        self.assertEqual(recuperar.call_count, 4)
        self.assertEqual(len(prompts), 3)  # Fuera de dominio sin contexto no usa LLM.
        self.assertEqual(len(trazas), 2)
        for traza, prompt in zip(trazas, (prompts[0], prompts[2])):
            contexto = json.dumps(traza["documentos"], ensure_ascii=False, indent=2)
            self.assertIn(contexto, prompt.to_messages()[1].content)
            fila = next(f for f in matriz if f["pregunta"] == traza["pregunta"])
            self.assertIs(traza["documentos"], fila["documentos"])
            self.assertEqual(traza["respuesta"], fila["respuesta"])
        # La respuesta ficticia de negocio no recibe aprobación automática de A.3.
        self.assertEqual(matriz[0]["estado"], "pendiente")

    def test_error_de_ejecucion_es_pendiente_y_no_expone_el_mensaje(self):
        chain = Mock()
        chain.invoke.side_effect = RuntimeError("Contenido sensible del proveedor")
        fila = validacion.ejecutar_caso(chain, self.casos["factual"])
        self.assertEqual(fila["estado"], "pendiente")
        self.assertIsNone(fila["respuesta"])
        self.assertNotIn("Contenido sensible", fila["detalle"])
        self.assertEqual(validacion.crear_trazas([fila])[0]["estado"], "pendiente")

    def test_revision_humana_no_repite_consulta(self):
        from contextlib import redirect_stdout
        from io import StringIO
        chain = Mock()
        chain.invoke.return_value = self.resultado("factual")
        with patch("builtins.input", return_value="a"), redirect_stdout(StringIO()):
            fila = validacion.ejecutar_caso(chain, self.casos["factual"], revisar=True)
        self.assertEqual(fila["estado"], "aprobado")
        chain.invoke.assert_called_once_with(self.casos["factual"]["pregunta"])

    def test_recursos_ausentes_no_llaman_pipeline_y_dejan_todo_pendiente(self):
        from contextlib import redirect_stdout
        from io import StringIO
        faltantes = ["persistencia Chroma", "LLM_API_KEY"]
        salida = StringIO()
        with patch.object(validacion, "recursos_faltantes", return_value=faltantes):
            with patch.object(rag, "crear_pipeline") as crear:
                with patch.object(sys, "argv", ["validacion_rag.py"]), redirect_stdout(salida):
                    self.assertEqual(validacion.main(), 2)
                crear.assert_not_called()
        informe = json.loads(salida.getvalue())
        self.assertEqual(len(informe["a3_matriz"]), 4)
        self.assertEqual(len(informe["a4_trazas"]), 2)
        for fila in informe["a3_matriz"] + informe["a4_trazas"]:
            self.assertEqual(fila["estado"], "pendiente")
            self.assertIsNone(fila["respuesta"])
            self.assertEqual(fila["documentos"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
