import ollama

response = ollama.chat(
    model="qwen3:1.7b",
    messages=[{"role": "user", "content": "Explain recursion briefly"}]
)
print(response["message"]["content"])