import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def configure_genai():
    """Configures the Google Generative AI with the API key."""
    # Try getting the default key, or fallback to the numbered keys
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY1") or os.getenv("GEMINI_API_KEY2")
    if not api_key:
        raise ValueError("No valid GEMINI_API_KEY found in environment variables.")
    genai.configure(api_key=api_key)

def generate_content(prompt, model_name=None):
    """
    Generates content using the specified Gemini model.
    
    Args:
        prompt (str): The input prompt for the model.
        model_name (str): The name of the model to use. Defaults to "gemini-1.5-flash".
        
    Returns:
        str: The generated text content.
    """
    if model_name is None:
         model_name = "gemini-2.5-flash-lite"

    try:
        configure_genai()
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"
