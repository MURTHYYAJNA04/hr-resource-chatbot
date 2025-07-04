# 💼 HR Resource Chatbot

An AI-powered chatbot built with FastAPI and Sentence Transformers to help HR teams find suitable candidates efficiently.

## 🚀 Features

- Natural language HR queries
- Semantic search using `all-MiniLM-L6-v2`
- FastAPI backend + HTML/JS frontend
- Optional GPT-3.5 Turbo integration

## 📦 Tech Stack

- FastAPI
- Sentence Transformers
- NumPy
- HTML + JavaScript
- OpenAI API (optional)

## 🔧 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/MURTHYYAJNA04/hr-resource-chatbot.git
cd hr-resource-chatbot

2. Install dependencies
pip install -r requirements.txt

3. Run the backend
python main.py
📍 It will start FastAPI at:
http://127.0.0.1:8000

4. Run the frontend
python -m http.server 8001
📍 Open in browser:
http://127.0.0.1:8001/chat.html

📁 Project Structure
hr-resource-chatbot/
├── main.py                 # FastAPI backend
├── chat.html               # Frontend UI
├── employees.json          # HR dataset
├── requirements.txt        # Dependencies
└── README.md               # This file
