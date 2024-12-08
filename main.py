import ollama

# Create a new model with modelfile
modelfile = """
FROM llama3.2
SYSTEM You are a very smart assistant who answers questions succintly and informatively.
PARAMETER temperature 0.1
"""
    
ollama.create(model="balogun", modelfile=modelfile)

# res = ollama.chat(
#     model="llama3.2",
#     messages=[
#         {"role": "user", "content": "Define Raleigh scattering"}
#     ],
#     stream=True,
# )

res = ollama.generate(
    model="balogun",
    prompt="what is the composition of air?",
    stream=True
)

print("...........................................................................")
print(res["response"])

