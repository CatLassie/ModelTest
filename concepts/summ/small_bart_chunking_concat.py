from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the model and tokenizer
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

def chunk_text(text, max_tokens=1024, overlap=200):
    """Splits text into chunks of max_tokens with optional overlap."""
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens - overlap):
        chunks.append(tokens[i : i + max_tokens])

        # Stop if the last chunk is processed
        if i + max_tokens >= len(tokens):
            break

    return chunks

def generate_summary(text, max_length=256):
    """Generates a summary for a given text input."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_document(text):
    """Performs multi-step summarization."""
    # Step 1: Chunk the document
    chunks = chunk_text(text)
    
    # Step 2: Summarize each chunk
    chunk_summaries = [generate_summary(tokenizer.decode(chunk)) for chunk in chunks]

    # Step 3: Combine chunk summaries and summarize again
    combined_summary_text = " ".join(chunk_summaries)
    final_summary = generate_summary(combined_summary_text, max_length=512)  # Increase length if needed

    return final_summary

# Example long document
long_document = "Your long article or text here..."
summary = summarize_document(long_document)
print(summary)