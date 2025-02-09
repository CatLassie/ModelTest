import torch
from transformers import pipeline

# Determine whether to use the GPU (if available) or CPU
device = 0 if torch.cuda.is_available() else -1

# Initialize the summarization pipeline with the specified model.
# The "device" parameter tells the pipeline to run on GPU (0) or CPU (-1).
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device
)

# Define a sample long document to summarize.
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

# Set the summarization parameters:
# - max_length: maximum number of tokens in the summary.
# - min_length: minimum number of tokens in the summary.
# - do_sample: when False, uses deterministic (greedy or beam) decoding.
# - batch_size: number of documents processed simultaneously.
summary_params = {
    "max_length": 150,
    "min_length": 40,
    "do_sample": False,
    "batch_size": 1
}

# Run the summarization (note that we wrap the document in a list, since batch_size applies to a list of inputs)
summaries = summarizer([document], **summary_params)

# Print the resulting summary. Each element in the returned list is a dictionary.
print("Summary:")
print(summaries[0]['summary_text'])
