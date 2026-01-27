from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

print(model.invoke("Which model are you using?")) 