from keybert import KeyBERT
kw_model = KeyBERT()

def extract_relevant_sentences(text, top_n=5):
    """Extracts key sentences before summarization."""
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)
    relevant_sentences = [sentence for sentence in text.split(". ") if any(k[0] in sentence for k in keywords)]
    return ". ".join(relevant_sentences)
