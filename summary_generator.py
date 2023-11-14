import os
import torch
import nltk
from transformers import pipeline, AutoTokenizer

# model: https://huggingface.co/kabita-choudhary/finetuned-bart-for-conversation-summary
# code: https://huggingface.co/knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI

nltk.download('punkt')

model_file = "pytorch_model.bin"

transformer_model_path = "/root/.cache/summary/"
# check if model path exists
if not os.path.exists(transformer_model_path + model_file):
    transformer_model_path = "cache/summary/"

#summarizer = pipeline("summarization", model="kabita-choudhary/finetuned-bart-for-conversation-summary", device="cpu")
#summarizer = pipeline("summarization", model=transformer_model_path, device_map="auto", torch_dtype=torch.bfloat16)
summarizer = pipeline("summarization", model=transformer_model_path, device_map="cpu")

# Initialize the tokenizer based on the model
tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)

max_token_length = 512 # 512 for most BART models, 1024 for T5 or GPT-2 models

# Function to chunk text
def chunk_text(text, max_length):
    total_length = len(tokenizer.encode(text))

    if total_length <= max_length:
        return [text]

    sentences = nltk.sent_tokenize(text)
    chunks = []
    chunk = ""
    chunk_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence))

        # Special case: if a single sentence is too long
        if sentence_length > max_length:
            print("Warning: A sentence exceeded the model's maximum token limit and will be split.")
            sub_sentences = sentence.split(", ")
            
            for sub_sentence in sub_sentences:
                sub_sentence_length = len(tokenizer.encode(sub_sentence))

                if chunk_length + sub_sentence_length > max_length:
                    chunks.append(chunk.strip())
                    chunk = ""
                    chunk_length = 0

                if sub_sentence_length > max_length:
                    print("Warning: A sub-sentence still exceeds the model's maximum token limit and will be truncated.")
                    sub_sentence = tokenizer.decode(tokenizer.encode(sub_sentence)[:max_length])
                    sub_sentence_length = len(tokenizer.encode(sub_sentence))

                chunk += sub_sentence + ", "
                chunk_length += sub_sentence_length

            continue

        if chunk_length + sentence_length > max_length:
            chunks.append(chunk.strip())
            chunk = ""
            chunk_length = 0

        chunk += sentence + " "
        chunk_length += sentence_length

    if chunk:
        chunks.append(chunk.strip())

    return chunks


def summarize(text, max_length=142):
    # Chunk the text
    text_chunks = chunk_text(text, tokenizer.model_max_length)

    # Summarize each chunk
    summarizations = summarizer(text_chunks, max_length=max_length)
    
    # Combine the summaries
    summary_text = ". ".join([summ['summary_text'] for summ in summarizations])
    
    return summary_text.strip()
