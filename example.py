"""
Example script demonstrating LangChain and OpenAI integration
Make sure to set your OPENAI_API_KEY in the .env file
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Example: Simple chat interaction
def simple_chat_example():
    """Simple example of using LangChain with OpenAI"""
    messages = [
        HumanMessage(content="Hello! Can you explain what LangChain is in one sentence?")
    ]
    
    response = llm.invoke(messages)
    print("Response:", response.content)

if __name__ == "__main__":
    print("LangChain + OpenAI Example")
    print("-" * 40)
    simple_chat_example()
