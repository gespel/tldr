import ollama
from bs4 import BeautifulSoup

def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    return soup.get_text()

def initialize_model_with_context(model, context):
    messages = [

    ]
    return messages

def ask_mistral(model, context, question):
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant who answers questions based on the manual you are give. try to stay very close to the manual."},
        {"role": "system", "content": f"Here is the manual:\n{context}"},
        {"role": "user", "content": question}
    ]

    stream = ollama.chat(model=model, messages=messages, stream=True)
    print("\nAntwort vom Modell (live):\n")
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

if __name__ == "__main__":
    html_text = extract_text_from_html("doca_docs_1.html")
    initialize_model_with_context("mistral", html_text)

    frage = "What is a pipe entry?"
    ask_mistral("mistral", frage, html_text)
