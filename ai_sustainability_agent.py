import os
import logging
import requests
import re
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
CLIMATIQ_API_KEY = os.getenv("CLIMATIQ_API_KEY")
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"

if not CLIMATIQ_API_KEY:
    raise ValueError("Missing CLIMATIQ_API_KEY in .env file.")

logging.basicConfig(level=logging.INFO)
_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance:
        return _llm_instance

    if USE_OLLAMA:
        logging.info("Using Ollama with Mistral model (local mode)")
        _llm_instance = Ollama(model="mistral")
    else:
        # Hugging Face fallback (disabled unless activated manually)
        # from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        # from langchain_community.llms import HuggingFacePipeline
        # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)
        # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)
        # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.7)
        # _llm_instance = HuggingFacePipeline(pipeline=pipe)

        if not HF_TOKEN:
            raise ValueError("HF_TOKEN is required if USE_OLLAMA is false.")
        raise ValueError("USE_OLLAMA is set to false, but Hugging Face support is currently disabled.")

    return _llm_instance

def sustainability_calculator(query: str) -> str:
    query = query.lower()
    res = []
    if any(word in query for word in ["plastic", "bioplastic", "compostable"]):
        res.append("Compostable packaging can reduce ~500kg CO₂/year.")
    match = re.search(r"(\d+)\s*(l|liters|litres)\s*(diesel|petrol|gasoline)", query)
    if match:
        qty, _, fuel = match.groups()
        co2e = round(2.39 * int(qty), 2)
        res.append(f"{qty}L of {fuel} emits approximately {co2e}kg CO₂.")
    if "electricity" in query or "solar" in query:
        res.append("Switching to solar may cut 20–40% CO₂ from electricity use.")
    if not res:
        res.append("Include keywords like plastic, fuel, diesel, or electricity.")
    return "\n".join(res)

def get_climatiq_emissions(_: str) -> str:
    headers = {
        "Authorization": f"Bearer {CLIMATIQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "emission_factor": {
            "activity_id": "fuel_type_diesel-fuel_vehicle_type_car_engine_size_na"
        },
        "parameters": {
            "fuel_amount": 100,
            "fuel_unit": "l"
        }
    }
    try:
        r = requests.post("https://api.climatiq.io/estimate", json=body, headers=headers)
        r.raise_for_status()
        co2e = r.json().get("co2e")
        return f"Real-time result: 100L diesel emits **{co2e:.2f} kg CO₂** (via Climatiq API)"
    except Exception as e:
        return f"Climatiq API error: {e}"
def build_agent():
    llm = get_llm()
    tools = [
        Tool(
            name="SustainabilityCalculator",
            func=sustainability_calculator,
            description="Estimates CO₂ impact of plastic, fuel, or electricity use."
        ),
        Tool(
            name="ClimatiqAPI",
            func=get_climatiq_emissions,
            description="Fetches real-time CO₂ emissions using the Climatiq API."
        )
    ]

def build_agent():
    tools = [
        Tool(
            name="SustainabilityCalculator",
            func=sustainability_calculator,
            description=(
                "Estimates CO₂ impact manually based on keywords or common quantities. "
                "Use for general queries like: 'How much CO2 from 100kWh?', "
                "'Plastic vs compostable', or 'Is solar energy better?'."
            )
        ),
        Tool(
            name="ClimatiqAPI",
            func=get_climatiq_emissions,
            description=(
                "Uses the Climatiq API to fetch real-time emissions data. "
                "Use when exact values like '100L of diesel' or 'domestic flight' are given."
            )
        )
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and intelligent AI Sustainability Agent focused on reducing CO₂ emissions in Pakistan and globally."),
        ("system", "You have access to these tools: {tool_names}"),
        ("system", "Tool details:\n{tools}"),
        ("system", 
         "To solve any CO₂ or sustainability-related question, follow this process:\n\n"
         "Question: {input}\n"
         "Thought: What is this about? Fuel, electricity, transport, packaging, etc?\n"
         "Action: Choose one of [{tool_names}]\n"
         "Action Input: Give a clean input like '100L of diesel', 'compare plastic vs bioplastic', etc.\n"
         "Observation: What the tool returns\n"
         "Final Answer: Summarize clearly and informatively."),
        ("human", "{input}"),
        ("system", "Previous steps:\n{agent_scratchpad}")
    ])

    print("Loaded tools:", [tool.name for tool in tools])

    return AgentExecutor(
    agent=create_react_agent(llm=get_llm(), tools=tools, prompt=prompt),
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15,  
    max_execution_time=90  
)

def init_rag_from_pdf(pdf_path="sustainability_docs/example.pdf"):
    if not os.path.exists(pdf_path):
        print("PDF not found. Please check the path:", pdf_path)
        return "PDF not found."
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    db = Chroma.from_documents(docs, embedding=OllamaEmbeddings(model="nomic-embed-text"))
    return RetrievalQA.from_chain_type(llm=get_llm(), retriever=db.as_retriever())

def voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I didn’t catch that."
        except sr.RequestError as e:
            return f"Voice recognition error: {e}"
        except sr.WaitTimeoutError:
            return "Listening timed out while waiting for speech."

def voice_output(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    agent = build_agent()
    print("\n AI Sustainability Agent Ready (CLI Mode + Voice + RAG)\nType 'exit' to quit.\n")

    while True:
        use_voice = input("Use voice? (y/n): ").strip().lower()
        question = voice_input() if use_voice == 'y' else input("Ask: ")
        if question.lower() in ["exit", "quit"]:
            break

        try:
            response = agent.invoke({"input": question})
            resp_text = response.get('output') or str(response)
            print("\nResponse:\n", resp_text)
            voice_output(resp_text)
        except Exception as e:
            print(f"Error: {e}")

