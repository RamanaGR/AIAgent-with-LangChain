from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()

chatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

template = "Write a {tone} email to {company} expressing interest in the {position}, mentioning {skill} as a key strength. Keep it a 3 lines max"

prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke(
    {
        "tone": "formal",
        "company": "Google",
        "position": "Software Engineer",
        "skill": "Python",
    }
)

print(prompt)

# Example 2

messages = [("system", "You are a helpful assistant that can write emails."),
            ("user", "Write a formal email to Google expressing interest in the Software Engineer position, mentioning Python as a key strength. Keep it a 3 lines max")]

response = chatOpenAI.invoke(messages)
print(response.content)