from vertexai.preview.generative_models import GenerativeModel
import json

def load_user_data(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def format_user_data(data: dict) -> str:
    return json.dumps(data, indent=2)

def start_chat_with_gemini():
    model = GenerativeModel("gemini-1.5-pro")
    chat = model.start_chat()
    return chat

def build_prompt(user_data: dict, question: str, history: list) -> str:
    memory = ""
    if history:
        memory = "\nRecent context:\n" + "\n".join([f"Q: {q}\nA: {a}" for q, a in history[-3:]])
    return f"""
You are a helpful assistant guiding the user through questions using their structured data below.

User Data:
{format_user_data(user_data)}
{memory}

User's Question:
{question}

Please answer in a clear, actionable way.
"""

def run_chat():
    user_data = load_user_data("user_data.json")
    chat = start_chat_with_gemini()
    history = []
    print("\U0001F537 FinMate Gemini Chat (type 'exit' to quit)\n")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break
        prompt = build_prompt(user_data, question, history)
        response = chat.send_message(prompt)
        answer = response.text.strip()
        print(f"AI: {answer}\n")
        history.append((question, answer))

if __name__ == "__main__":
    run_chat()
