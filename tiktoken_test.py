import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o")

consulta_español = 'Hola, mi pedido número 48327 debía llegar ayer y todavía no lo recibí. ¿Dónde está y cuándo va a llegar?'
consulta_inglés = 'Hello, my order number 48327 was supposed to arrive yesterday and I still haven’t received it. Where is it and when will it arrive?'

print(f"ES: {len(encoding.encode(consulta_español))} tokens")
print(f"EN: {len(encoding.encode(consulta_inglés))} tokens")