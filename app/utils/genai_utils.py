import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite") # gemini-2.0-flash-lite

def generate_explanation(row_dict):
    prompt = f"""
    Analyze the following customer data and explain in one sentence why the model have predicted churn:
    {row_dict}
    """

    try:
        response = model.generate_content(prompt)
        return response.strip()
    except Exception as e:
        return f"Failed to generate explanation: {str(e)}"
