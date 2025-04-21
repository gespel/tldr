import ollama
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import numpy as np
from bs4 import BeautifulSoup

def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    return soup.get_text(separator='\n')

def split_text_into_chunks(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def create_embeddings(text_chunks, model):
    embeddings = model.encode(text_chunks)
    return embeddings

def build_annoy_index(embeddings, dimension):
    annoy_index = AnnoyIndex(dimension, 'angular')  # 'angular' ist eine häufig verwendete Distanzmetrik
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)
    annoy_index.build(10)  # Baue den Index (10 Bäume)
    return annoy_index

def retrieve_relevant_sections(query, model, annoy_index, text_chunks, top_k=5):
    query_embedding = model.encode([query])[0]  # Erzeuge das Embedding der Anfrage
    nearest_neighbors = annoy_index.get_nns_by_vector(query_embedding, top_k)  # Top k Nachbarn finden
    relevant_sections = [text_chunks[i] for i in nearest_neighbors]  # Holen der Text-Chunks
    return relevant_sections

def ask_mistral_with_retrieved_context(model, query, relevant_sections):
    context = "\n".join(relevant_sections)  # Relevanteste Abschnitte kombinieren
    messages = [
        {"role": "system", "content": "You are a helpful assistant who answers questions based on the provided context."},
        {"role": "user", "content": f"Here is the context:\n{context}"},
        {"role": "user", "content": query}
    ]
    # Anfrage an das Modell senden
    stream = ollama.chat(model=model, messages=messages, stream=True)
    print("\nAntwort vom Modell (live):\n")
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

if __name__ == "__main__":
    html_text = extract_text_from_html("doca_docs_1.html")

    text_chunks = split_text_into_chunks(html_text)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = create_embeddings(text_chunks, model)

    annoy_index = build_annoy_index(embeddings, dimension=len(embeddings[0]))

    question = "Explain what a steering domain is?"

    relevant_sections = retrieve_relevant_sections(question, model, annoy_index, text_chunks)

    ask_mistral_with_retrieved_context("mistral", question, relevant_sections)