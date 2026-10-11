from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENAI_API_KEY")

if key:
    print("API Key loaded successfully ✅")
else:
    print("API Key not found ❌")