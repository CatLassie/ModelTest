import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Determine the device: use GPU if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model name
model_name = "sshleifer/distilbart-cnn-12-6"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)

# Define a long document to summarize
document = (
    "The field of artificial intelligence has undergone significant advancements "
    "over the past decade. Researchers have developed models that can perform tasks "
    "ranging from natural language processing to computer vision. One notable breakthrough "
    "has been the development of transformer-based architectures, which have revolutionized "
    "the way machines understand and generate human language. These models are not only "
    "effective but also scalable, making them suitable for a variety of applications such as "
    "translation, summarization, and conversational agents. As the technology continues to evolve, "
    "new challenges and opportunities emerge, further pushing the boundaries of what artificial "
    "intelligence can achieve in both academic research and industry applications."
)

# For batch processing, wrap the document in a list.
documents = [document]  # Here, batch size = 1; add more documents to the list for larger batches.

# Tokenize the document(s) with appropriate truncation and maximum length.
# Adjust max_length if your input is longer; distilbart typically supports up to 1024 tokens.
inputs = tokenizer(documents, return_tensors="pt", max_length=1024, truncation=True, padding="longest")
inputs = {key: val.to(device) for key, val in inputs.items()}

# Set generation parameters:
#   - max_length: maximum length of the summary tokens.
#   - min_length: minimum length of the summary tokens.
#   - num_beams: for beam search (higher values may improve quality at the cost of speed).
#   - do_sample: whether to use sampling; False means deterministic (greedy/beam search).
generation_kwargs = {
    "max_length": 150,
    "min_length": 40,
    "num_beams": 4,
    "do_sample": False,
    "early_stopping": True,
}

# Generate the summary using the model.
# For a batch, this will return a tensor of shape (batch_size, sequence_length)
summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], **generation_kwargs)

# Decode the generated tokens to text.
summaries = [tokenizer.decode(sid, skip_special_tokens=True) for sid in summary_ids]

# Print the generated summary (or summaries, if batch size > 1)
print("Summary:")
print(summaries[0])
