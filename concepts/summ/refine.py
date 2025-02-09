from transformers import AutoModelForCausalLM

gpt_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct").to("cuda")

def gpt_refine_summary(text):
    """Uses a GPT-based model to refine the summary."""
    prompt = f"Refine the following summary: {text}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = gpt_model.generate(inputs["input_ids"], max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)
