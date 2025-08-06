import os
import logging
import requests
import re
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, AgentExecutor, create_react_agent
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
        logging.info("Using OllamaLLM with Mistral model (local mode)")
        _llm_instance = OllamaLLM(model="mistral")
    else:
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
    tools = [
        Tool(
            name="SustainabilityCalculator",
            func=sustainability_calculator,
            description=(
                "Estimates CO₂ impact based on keywords or values like '100kWh', "
                "'plastic vs compostable', or 'diesel usage'."
            )
        ),
        Tool(
            name="ClimatiqAPI",
            func=get_climatiq_emissions,
            description="Provides real-time CO₂ emission data using the Climatiq API."
        )
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI Sustainability Agent focused on reducing CO₂ emissions."),
        ("system", "You can access the following tools: {tool_names}"),
        ("system", "Tool descriptions:\n{tools}"),
        ("system", 
         "To answer a question, use the following format:\n"
         "Question: {input}\n"
         "Thought: think about the question\n"
         "Action: <tool_name>\n"
         "Action Input: <tool input>\n"
         "Observation: result of the tool\n"
         "Final Answer: a helpful and clear reply to the user."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    print("Loaded tools:", [tool.name for tool in tools])

    return AgentExecutor(
        agent=create_react_agent(
            llm=OllamaLLM(model="mistral"),
            tools=tools,
            prompt=prompt
        ),
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        max_execution_time=60,
    )


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
    print("🎤 Speaking:", text)
