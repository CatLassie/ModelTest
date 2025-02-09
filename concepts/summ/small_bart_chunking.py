from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the model and tokenizer
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

def chunk_text(text, max_tokens=1024, overlap=200):
    """Splits text into chunks of max_tokens with optional overlap"""
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens - overlap):
        chunks.append(tokens[i : i + max_tokens])

        # Stop if the last chunk is processed
        if i + max_tokens >= len(tokens):
            break

    return chunks

def summarize(text):
    """Summarizes a long document by chunking"""
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        input_ids = torch.tensor([chunk]).to("cuda")
        summary_ids = model.generate(input_ids, max_length=256)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Combine all chunk summaries into a final summary
    return " ".join(summaries)

# Example long document
long_document = "Your long article or text here..."
summary = summarize(long_document)
print(summary)