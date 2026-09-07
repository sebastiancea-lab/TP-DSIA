# **A.1 El caso**

Portalia es una tienda de comercio electrónico ficticia que procesa más de 1.000 ventas diarias. Actualmente, su equipo de atención gestiona manualmente la postventa a través de WhatsApp, resolviendo consultas sobre seguimiento de pedidos, cambios y devoluciones. El proyecto propone automatizar este proceso mediante un chatbot con IA, con el objetivo de reducir la saturación del equipo, mejorar los tiempos de respuesta y optimizar la experiencia del cliente.

# 

# **A.2 Evidencia de la necesidad (réplica de "El Proveedor Enojado")**

Adjuntamos el link de como se trabajo con gemini y además transcribimos la conversación  
[https://share.gemini.google/AboDKo5ZoDEl](https://share.gemini.google/AboDKo5ZoDEl)

**Prompt inicial:** *"Actuá como el sistema de atención de Portalia. Un usuario pregunta:“Hola soy Gisella, mi pedido número 48327 debía llegar ayer y todavía no lo recibí. ¿Dónde está y cuándo va a llegar? Respondé como lo haría el sistema."*

**Respuesta de gemini:**  *“ ¡Hola, Gisella\! Bienvenida al centro de atención de Portalia. Lamentamos mucho la demora. Entendemos lo frustrante que es esperar una compra y que no cumpla con los plazos establecidos.*  
Ya mismo revisé el estado de tu pedido **\#48327** en nuestro sistema. Te comparto la información actualizada:

* **Estado actual:** En tránsito.  
* **¿Qué pasó?** Tu paquete fue despachado correctamente de nuestro depósito, pero la empresa de logística nos informa que hubo una demora inesperada en la ruta de distribución asignada para el día de ayer en tu zona.  
* **¿Cuándo va a llegar?** La entrega ya fue priorizada y reprogramada. Vas a estar recibiendo tu pedido en el transcurso del día de **hoy** o, a más tardar, **mañana por la mañana**.

Te acabo de enviar un correo electrónico con el enlace de seguimiento actualizado para que puedas ver los movimientos del repartidor en tiempo real.  
¿Hay algo más en lo que te pueda ayudar mientras tanto?

El modelo respondió con un alto nivel de confianza, y sin ningún inconveniente a la hora de inventar accesos a la base de datos, estados de pedido (demorado, priorizado) o soluciones (reprogramar el pedido, enviar un email).

En términos del mensaje el modelo respondió de la manera que se espera, por lo cual le faltaria la conexión al back end para hacerlo de manera real y no con datos inventados, quizas se podria revisar si el tono es el deseado y si se desean hacer “envios de mails”

### 

# **A.3 PEAS**

## **Performance (¿Cómo medimos el éxito?)**

➜ Reducir el tiempo de primera respuesta al cliente a menos de **30 segundos**, con disponibilidad 24/7.  
➜  Resolver de forma autónoma al menos el **80% de las consultas de seguimiento de pedidos y solicitudes de cambios y devoluciones**.  
➜  Alcanzar al menos un **95% de validaciones correctas** al aplicar las políticas de cambios y devoluciones de Portalia.  
➜  Reducir la cantidad de consultas de postventa que requieren intervención de un operador humano.

### 

## **Environment (¿Dónde opera?)**

➜ WhatsApp como canal de atención al cliente.  
➜ Plataforma de e-commerce de Portalia y chatbot del sitio web.  
➜ Backend del e-commerce.  
➜ Base de datos de pedidos, clientes y productos.  
➜ API de la empresa de logística.

## 

## **Actuators (¿Qué puede hacer el sistema?)**

➜  Enviar respuestas a los clientes a través de WhatsApp y del chatbot del sitio web.  
➜ Informar el estado y seguimiento de los pedidos utilizando los datos obtenidos del backend y de la API de logística.  
➜ Registrar tickets y solicitudes de postventa.  
➜ Registrar solicitudes de cambios y devoluciones cuando correspondan.  
➜ Actualizar el estado de las solicitudes de postventa.  
➜ Notificar al cliente sobre avances o cambios en el estado de su solicitud.  
➜ Derivar la conversación a un operador humano cuando el sistema no pueda resolverla de manera autónoma.

### 

## **Sensors (¿Qué información recibe?)**

➜ Mensajes de texto enviados por los clientes a través de WhatsApp o del chatbot, incluyendo mensajes con errores ortográficos, abreviaciones o jerga.  
➜ Notas de voz enviadas por los clientes.  
➜ Fotografías de productos dañados, comprobantes y otros documentos adjuntos.  
➜ Datos proporcionados por el cliente durante la conversación, como número de pedido, producto, variante solicitada o motivo de una devolución.  
➜ Eventos o respuestas recibidas desde las APIs y sistemas integrados necesarios para procesar la consulta.

### 

## **Base de Conocimiento (¿Qué información necesita el sistema?)**

➜ Políticas de cambios, devoluciones, garantías y plazos de Portalia.  
➜ Preguntas frecuentes y procedimientos de atención de postventa.  
➜ Catálogo de productos con sus descripciones, características, categorías, variantes e instrucciones.  
➜ Información de pedidos e historial de compras de los clientes.  
➜ Datos actualizados de precios, stock y disponibilidad de productos.  
➜ Historial de interacciones y solicitudes de postventa.  
➜ Información sobre el estado y la trazabilidad de los envíos.

### 

Las **políticas, procedimientos, preguntas frecuentes y la información descriptiva del catálogo** podrán incorporarse posteriormente a una base vectorial para ser recuperadas mediante **RAG**. En cambio, los datos dinámicos, como stock, precios, pedidos y estados de envío, deberán consultarse en tiempo real desde las bases de datos y APIs correspondientes.

### 

## **Decisión técnica preliminar (¿Qué partes serían deterministas vs. IA?)**

➜ **Parte de IA:** el LLM se utilizará para interpretar los mensajes en lenguaje natural, identificar la intención del cliente y extraer los parámetros necesarios. También podrá procesar entradas no estructuradas, como notas de voz, y posteriormente utilizar RAG para recuperar información relevante de las políticas y procedimientos de Portalia.  
➜ **Parte determinista:** el backend será responsable de consultar y validar los datos reales del negocio. Por ejemplo, verificará el estado de un pedido mediante la base de datos y la API de logística, comprobará la disponibilidad exacta de stock y aplicará las reglas de negocio necesarias para determinar si corresponde realizar un cambio o una devolución.  
➜ **Criterio de separación:** el LLM **interpreta qué quiere el cliente y extrae la información de su mensaje**, mientras que el backend **consulta la fuente de verdad, aplica las reglas y determina qué acción puede realizarse**. El modelo no debe inventar estados de pedidos, stock, fechas de entrega ni decidir por sí mismo si una operación está autorizada.

# **A.4 Anatomía del token**

Para analizar cómo el modelo procesa el lenguaje, se comparó la tokenización de una misma consulta formulada en español y en inglés.

| consulta\_español \= "Hola, mi pedido número 48327 debía llegar ayer y todavía no lo recibí. ¿Dónde está y cuándo va a llegar?" consulta\_inglés \= "Hello, my order number 48327 was supposed to arrive yesterday and I still haven’t received it. Where is it and when will it arrive?" |
| :---- |

![][image1]  
Un token es una unidad en la que el modelo divide el texto para poder procesarlo. Un token no necesariamente equivale a una palabra completa: puede representar una palabra, una parte de ella, un número o un signo de puntuación.

En la prueba realizada, la consulta en inglés requirió aproximadamente un 10% más de tokens que la misma consulta en español. Esta diferencia no implica que el inglés siempre consuma más tokens, ya que la tokenización depende del texto utilizado y del modelo o tokenizer seleccionado.

La cantidad de tokens es relevante porque las APIs de modelos de lenguaje suelen contabilizar su uso en función de los tokens procesados. En un sistema como Portalia, que podría recibir un gran volumen de consultas diariamente, pequeñas diferencias en la cantidad de tokens por interacción pueden acumularse y tener impacto en el costo total de operación.

Además, en un entorno real las consultas pueden ser más extensas, contener errores ortográficos, abreviaciones, información innecesaria o conversaciones con varios intercambios, por lo que resulta importante considerar el consumo de tokens al diseñar y escalar la solución.