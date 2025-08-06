import os
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from agent_logic.sustainability_agent import build_agent
from langchain_core.messages import AIMessage, HumanMessage

app = Flask(__name__)
CORS(app)

# Initialize LangChain agent once
agent = build_agent()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No input provided"}), 400
    try:
        result = agent.invoke({"input": question})  
        print("Agent Output:", result)
        return jsonify({"response": result.get("output") or "No response"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# This should be outside of any function
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
