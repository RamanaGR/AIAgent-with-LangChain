
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
import os
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
messages = []
system_message = SystemMessage(
        content="You are a helpful assistant that can answer questions and help with tasks."
    )
messages.append(system_message)


while True:
    user_message = input("You: ")
    if user_message.lower() == "exit":
        break
    messages.append(HumanMessage(content=user_message))
    response = model.invoke(messages)
    messages.append(response)
    print(f"Assistant: {response.content}")