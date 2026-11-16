# A.4 - Anatomia del token:
# Comparamos una misma consulta de postventa en espanol y en ingles, como en la
# actividad. La idea es observar que el modelo procesa el texto en tokens, que
# pueden ser palabras completas, fragmentos, numeros o signos de puntuacion.
import tiktoken

# Usamos el tokenizador asociado a GPT-4o para que la comparacion sea concreta:
# el conteo puede variar si se elige otro modelo o tokenizador.
encoding = tiktoken.encoding_for_model("gpt-4o")

# Son dos versiones del mismo reclamo por demora, con el numero de pedido
# conservado para que la comparacion se centre en la formulacion del mensaje.
consulta_español = 'Hola, mi pedido número 48327 debía llegar ayer y todavía no lo recibí. ¿Dónde está y cuándo va a llegar?'
consulta_inglés = 'Hello, my order number 48327 was supposed to arrive yesterday and I still haven’t received it. Where is it and when will it arrive?'

# encode() divide cada texto en tokens y len() cuenta esas unidades. Imprimimos
# ambos conteos para contrastarlos con la observacion de A.4: en esta prueba el
# ingles tuvo aproximadamente un 10 % mas. Es un resultado de estos textos y
# este tokenizador, no una regla general sobre ambos idiomas.
print(f"ES: {len(encoding.encode(consulta_español))} tokens")
print(f"EN: {len(encoding.encode(consulta_inglés))} tokens")