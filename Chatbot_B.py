import os
import json
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load knowledge base from JSON file
with open("database.json") as f:
    knowledge_base = json.load(f)

# Initialize Gemini model
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize chat memory
chat_memory = [
    {"role": "system", "content": "You are a helpful assistant who answers based on the knowledge base provided."}
]

def run_search(query, top_k=2):
    results = []
    for title, entry in knowledge_base.items():
        description = entry["description"]
        score = fuzz.partial_ratio(query.lower(), description.lower())
        if query.lower() in description.lower():
            score += 30
        results.append((score, title, description))

    top_results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    return "\n".join([f"{title}: {desc}" for _, title, desc in top_results])

def summarize(memory):
    assistants = [msg["content"] for msg in memory if msg["role"] == "assistant"]
    return " | ".join(assistants[-2:]) if assistants else "No previous response to summarize."

def build_prompt(user, summary, search_results, question):
    return f"""
User Info: {user['name']} (Level: {user['level']})

Summary of Previous Discussion:
{summary}

Top Matching Knowledge Base Results:
{search_results}

User's Current Question:
{question}

Answer clearly and concisely based on the above.
"""

def chatbot_response(user_input, user_info):
    global chat_memory

    if user_input.lower() in ["continue", "explain again", "give example"]:
        last_reply = chat_memory[-1]["content"]
        if user_input.lower() == "continue":
            prompt_text = last_reply + "\n\nPlease continue."
        elif user_input.lower() == "explain again":
            prompt_text = last_reply + "\n\nCan you explain that again?"
        else:
            prompt_text = last_reply + "\n\nCan you give an example?"
    else:
        summary = summarize(chat_memory)
        search_result = run_search(user_input)
        prompt_text = build_prompt(user_info, summary, search_result, user_input)
        chat_memory.append({"role": "user", "content": user_input})

    try:
        response = gemini_model.generate_content(prompt_text)
        reply = response.text.strip()
    except Exception as e:
        reply = f"❌ Error calling Gemini API: {str(e)}"

    chat_memory.append({"role": "assistant", "content": reply})
    return reply

def main():
    user_info = {"name": "Alex", "level": "beginner"}
    print("🤖 Gemini Chatbot ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("👋 Exiting the chatbot. Goodbye!")
            break
        response = chatbot_response(user_input, user_info)
        print("Bot:", response)

if __name__ == "__main__":
    main()
