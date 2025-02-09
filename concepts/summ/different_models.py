first_pass_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6").to("cuda")
second_pass_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to("cuda")

def generate_summary_with_two_models(text):
    """Summarizes using two different models for better quality."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    
    # First-pass summarization (faster model)
    summary_ids = first_pass_model.generate(inputs["input_ids"], max_length=256)
    first_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Second-pass summarization (higher-quality model)
    inputs = tokenizer(first_summary, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    summary_ids = second_pass_model.generate(inputs["input_ids"], max_length=200)
    final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return final_summary
