## AI Sustainability Agent

An AI-powered voice-enabled assistant that helps estimate and reduce CO₂ emissions using natural language queries. It supports voice input/output, real-time API integrations (Climatiq), PDF-based RAG (Retrieval-Augmented Generation), and local LLM via Ollama.

---

## 📌 Features

- ✅ **Voice-enabled** question-answering (speech-to-text + text-to-speech)
- 🔁 **RAG**: Answer queries based on custom sustainability PDFs
- 🔍 **CO₂ Estimation Tools**:
  - Manual calculator using rule-based logic
  - Real-time emissions data via **Climatiq API**
- 🤖 **LLM Support**:
  - Default: **Ollama (Mistral model)**
  - [Optional] Hugging Face support (commented)

---

## 🧠 Tech Stack

- 🐍 Python
- 🧠 LangChain
- 🔊 SpeechRecognition + Pyttsx3
- 🦙 Ollama (local LLM inference)
- 📄 ChromaDB + PDFLoader (for RAG)
- 🌍 Climatiq API (CO₂ estimates)
- 📦 dotenv (.env secrets handling)


## 📁 Directory Structure

├── ai_sustainability_agent\_docs/ # PDF files for RAG

├── .env                           # API keys & config

├── requirements.txt               # Python dependencies

└── README.md                      # You're here!



## ⚙️ Setup Instructions

### 1️⃣ Clone the Repo

git clone https://github.com/yourusername/AI-Sustainability-agent.git
cd AI-Sustainability-agent

### 2️⃣ Create a Virtual Environment

python -
m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3️⃣ Install Dependencies
bash

pip install -r requirements.txt
```

### 4️⃣ Add Your Environment Variables

Create a `.env` file in the root folder with the following:

```
HF_TOKEN=your_huggingface_token   # Optional unless USE_OLLAMA=false
CLIMATIQ_API_KEY=your_climatiq_key
USE_OLLAMA=true
```

### 5️⃣ Run Ollama Locally (if using)

Make sure Ollama is running and the model is pulled:


ollama run mistral


## 🚀 Run the Agent


python main.py


You’ll be prompted to use voice (`y/n`) or type questions.

Example queries:

* `"How much CO2 is emitted from 100L of diesel?"`
* `"Is plastic worse than compostable packaging?"`
* `"What are the benefits of switching to solar energy?"`



## 🎤 Voice Input/Output

* `speech_recognition` listens for your question.
* `pyttsx3` reads out the final answer.
* If you select text input, type your query.

---

## 📄 RAG: PDF-Based Answers

Place your sustainability documents in:


sustainability_docs/

Default file: `example.pdf`

The app will use this to answer queries based on document content.


## 🧪 API Tools Available

| Tool Name                | Description                                              |
| ------------------------ | -------------------------------------------------------- |
| SustainabilityCalculator | Estimates CO₂ emissions manually based on query keywords |
| ClimatiqAPI              | Calls Climatiq for real-time emissions data              |


## 🛠️ Example .env File

env
HF_TOKEN=your_hf_token_if_needed
CLIMATIQ_API_KEY=your_climatiq_api_key
USE_OLLAMA=true


## 📦 Future Enhancements

* Web UI (Streamlit or React frontend)
* Logging and user analytics
* More RAG sources (YouTube transcripts, websites)
* Emission visualizations
* Integration with government sustainability data

---

## 🧑‍💻 Author

**Mehwish Umar**

 Data Science Explorer 



## ✅ Next Steps

1. Save this as `README.md` in your project root.
2. Add a `requirements.txt` with your used libraries (if not already).


