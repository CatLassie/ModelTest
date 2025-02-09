def generate_summary(text, min_length=50, max_length=256, compression_ratio=0.3):
    """Generates a summary with adjustable compression ratio."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    summary_ids = model.generate(
        inputs["input_ids"], 
        min_length=int(len(text.split()) * compression_ratio * 0.5),  # Adaptive min_length
        max_length=int(len(text.split()) * compression_ratio),  # Adaptive max_length
        length_penalty=2.0,  # Encourages brevity
        num_beams=4  # Improves quality with beam search
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
