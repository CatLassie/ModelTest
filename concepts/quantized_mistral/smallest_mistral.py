# NOTE:
# The 4-bit GPTQ model uses ~4GB of VRAM, making it suitable for an RTX 4070.
# If you experience VRAM issues, try max_memory={0: "4GB"} inside from_pretrained().



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the 4-bit quantized model from TheBloke
model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Example prompt
prompt = "Explain the theory of relativity in simple terms."

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

# Decode output
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
