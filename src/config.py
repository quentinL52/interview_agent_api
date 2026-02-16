import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def crew_openai():
    """Returns a ChatOpenAI instance configured for CrewAI feedback analysis."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )
