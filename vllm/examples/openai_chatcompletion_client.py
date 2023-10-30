import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

# List models API
models = openai.Model.list()
print("Models:", models)

model = models["data"][0]["id"]

# Chat completion API
chat_completion = openai.ChatCompletion.create(
    model=model,
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }, {
        "role":
        "assistant",
        "content":
        "The Los Angeles Dodgers won the World Series in 2020."
    }, {
        "role": "user",
        "content": "Where was it played?"
    }])

print("Chat completion results:")
print(chat_completion)

# save results to a file
with open("results.txt", "w") as f:
    f.write("Models:\n")
    f.write(str(models) + "\n")

    f.write("\nChat completion results:\n")
    f.write(str(chat_completion))
