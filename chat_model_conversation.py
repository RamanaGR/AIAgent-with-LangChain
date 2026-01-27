from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
messages = [
    SystemMessage(
        content="You are a helpful assistant that can answer questions and help with tasks."
    ),
    HumanMessage(content="What is the capital of France?"),
]
print(model.invoke(messages))

print("\n--- Optional: switch providers (install extras first) ---")
try:
    from langchain_anthropic import ChatAnthropic  # type: ignore

    model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7)
    print(model.invoke(messages))
except ImportError:
    print("Anthropic not installed. Run: pip install langchain-anthropic")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    print(model.invoke(messages))
except ImportError:
    print("Google GenAI not installed. Run: pip install langchain-google-genai")


