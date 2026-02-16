import os
import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------
# Embedding Model
# ---------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------
# Load Local LLM (Stable Method)
# ---------------------------
print("Loading local LLM...")

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

tokenizer.pad_token = tokenizer.eos_token

# ---------------------------
# Load Documents
# ---------------------------
def load_docs(folder):

    texts = []
    names = []

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
            names.append(file)

    return texts, names


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_PATH = os.path.join(BASE_DIR, "docs")

docs, doc_names = load_docs(DOCS_PATH)



# ---------------------------
# Create FAISS Index
# ---------------------------
embeddings = embed_model.encode(docs)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("Indexed documents:", len(docs))


# ---------------------------
# Retrieval
# ---------------------------
def retrieve(query, k=3):

    q_embed = embed_model.encode([query])

    distances, indices = index.search(
        np.array(q_embed),
        k
    )

    results = []

    for i in indices[0]:
        results.append(docs[i])

    return results


# ---------------------------
# Generation
# ---------------------------
def generate_answer(query, contexts):

    context_text = " ".join(contexts)

    prompt = f"""
You are a warehouse assistant.

Use this documentation to answer clearly.

Documentation:
{context_text}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=400
    ).to(device)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Extract only text after "Answer:"
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    # Remove prompt repetition
    answer = answer.strip()

    return answer



# -------------------------------
# Optional Standalone Mode
# -------------------------------

def interactive_mode():

    print("\nRAG System Ready (Type 'exit' to quit)\n")

    while True:

        query = input("Ask: ")

        if query.lower() == "exit":
            break

        contexts = retrieve(query)
        answer = generate_answer(query, contexts)

        print("\nFinal Answer:\n")
        print(answer)
        print("\n" + "-" * 50)


# Run only if executed directly
if __name__ == "__main__":
    interactive_mode()


