# LangChain Tutorial (Beginner-Friendly, Detailed)
This guide teaches LangChain step-by-step using simple examples, then builds up to **RAG (Retrieval-Augmented Generation)** and basic **agent/tool** patterns.

It’s written for this repo’s setup (Python + `langchain` + `langchain-openai` + `python-dotenv`).

---

## What is LangChain?
LangChain is a framework that helps you build applications with LLMs (Large Language Models) in a structured way.

Instead of writing “one big prompt string” and hoping for the best, LangChain encourages you to combine:
- **Models** (chat/completions)
- **Prompts** (templates + variables)
- **Parsers** (convert LLM output to structured data)
- **Retrievers** (search your own data)
- **Chains** (connect components into a pipeline)
- **Tools & Agents** (LLM can decide what tool to call and when)

---

## Core mental model (how pieces fit together)
Think of a typical LLM app like this:

1. **Input** (user question)
2. **Context** (optional: retrieved docs, memory, system instructions)
3. **LLM call** (ChatOpenAI, etc.)
4. **Output** (plain text or structured JSON)

In LangChain, the same thing becomes composable building blocks:

```
User question
   ↓
PromptTemplate (inject variables)
   ↓
Chat Model (LLM)
   ↓
Output Parser (optional)
   ↓
Final answer / JSON / tool action
```

---

## Setup (local)

### 1) Create a virtual environment
From the repo root:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Add your OpenAI key
Create a `.env` file in the repo root:

```bash
OPENAI_API_KEY=your_key_here
```

Your Python scripts typically start with:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Part 1 — Your first chat model call

### Option A: simplest possible call (string in, string out)

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
result = llm.invoke("Explain LangChain in 1 sentence.")
print(result.content)
```

### Option B: message-based conversation
Chat models accept structured messages:
- **System** message: “rules” / role
- **Human** message: user input
- **AI** message: model output

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is RAG? Explain simply."),
]

response = llm.invoke(messages)
print(response.content)
```

### Important parameters
- **model**: which model to use
- **temperature**: creativity (0.0 = more deterministic, 0.7 = more creative)

---

## Part 2 — Prompts (templates with variables)
Instead of building prompt strings manually, use templates.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Summarize the topic: {topic} in {n} bullets."),
])

messages = prompt.format_messages(topic="LangChain", n=5)
```

Then pass those messages to a model:

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

response = llm.invoke(messages)
print(response.content)
```

---

## Part 3 — Chains (piping components together)
LangChain supports a simple “pipe” style called **LCEL**.

### Example: prompt → model → output text

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You explain things simply."),
    ("human", "Explain {thing} like I'm {age} years old."),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
parser = StrOutputParser()

chain = prompt | llm | parser

text = chain.invoke({"thing": "vector databases", "age": 12})
print(text)
```

Why this is good:
- You can swap **models**, **prompts**, or **parsers** without rewriting everything.

---

## Part 4 — Structured output (basic)
Many apps want JSON back (not just text). The simplest starting point is:
- ask for JSON in the prompt
- validate/parse it on your side

For production, you’d use structured output helpers or Pydantic-based parsing, but start simple.

```python
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = """Return ONLY valid JSON:
{
  "title": "...",
  "summary": "...",
  "tags": ["..."]
}
Topic: LangChain RAG
"""

resp = llm.invoke(prompt).content
data = json.loads(resp)
print(data["title"])
```

Tip: if JSON parsing fails, reduce creativity (`temperature=0`) and be strict in the prompt.

---

## Part 5 — RAG (Retrieval-Augmented Generation) basics
RAG = “LLM + your knowledge base”.

The standard RAG pipeline:

1. **Load documents** (files/web/db)
2. **Split documents** into chunks
3. **Embed** chunks into vectors
4. **Store vectors** in a vector DB
5. At query time: **retrieve** relevant chunks
6. Feed retrieved chunks + question to LLM

### Why RAG?
- LLMs can hallucinate
- Your company/project data isn’t in the model’s training set
- You want answers grounded in real documents

---

## Part 6 — Minimal RAG example (conceptual)
This tutorial is “dependency-light”, so the code below is a *template*.

For an actual working local RAG, you typically add:
- an embeddings model (often OpenAI embeddings)
- a vector store (e.g., FAISS, Chroma, Pinecone, etc.)
- a text splitter

### Pseudocode outline

```python
# 1) Load docs (files/web)
docs = load_documents(...)

# 2) Split docs
chunks = split(docs)

# 3) Create embeddings
vectors = embed(chunks)

# 4) Store in vector DB
vectorstore = store(vectors)

# 5) Retrieve for a question
retrieved_chunks = vectorstore.similarity_search(question)

# 6) Ask the LLM with retrieved context
answer = llm.invoke(
    f"Use this context:\n{retrieved_chunks}\n\nQuestion: {question}"
)
```

### What “good RAG” looks like
- You pass **only** the most relevant chunks (not all docs)
- You include **citations** (which chunk/source was used)
- You keep chunks small enough to be relevant, but large enough to carry meaning

---

## Part 7 — Tools and Agents (high-level)
An **agent** lets the model decide:
- “Do I answer directly?”
- “Do I call a tool (search, calculator, DB query)?”

### Tool idea examples
- Web search
- Query a vector store
- Read a local file
- Run calculations

Agent flow:

```
User → Agent
       ├─ (maybe calls Tool A)
       ├─ (maybe calls Tool B)
       └─ returns final answer
```

Start with tools you control and can validate.

---

## Part 8 — Good project habits

### Keep secrets out of git
- Put keys in `.env`
- Ensure `.env` is ignored in `.gitignore`

### Add a “single entrypoint”
As the project grows, create a folder structure like:

```
src/
  agent/
    main.py
  rag/
    ingest.py
    query.py
```

### Add logging / tracing later
Once you have multiple steps (RAG + tools + memory), add tracing (LangSmith) so you can see:
- which prompt was used
- which documents were retrieved
- latency and costs

---

## Suggested next steps for this repo
- Create a `src/` folder
- Implement “RAG v1” using a local vector store (FAISS/Chroma)
- Add ingestion script to index documents into the vector store
- Build a simple CLI chat app that:
  - uses retrieval when needed
  - uses normal chat when not

---

## Troubleshooting

### “OpenAI API key not found”
- Ensure `.env` exists in repo root
- Ensure your script calls `load_dotenv()`
- Restart your terminal/session

### “Module not found”
- Activate venv: `source venv/bin/activate`
- Reinstall deps: `pip install -r requirements.txt`

---

## Glossary (quick)
- **LLM**: Large Language Model (e.g., GPT-4 family)
- **Prompt**: instructions + inputs you send to the model
- **Embedding**: numeric vector representation of text for similarity search
- **Vector store**: database optimized for similarity search over vectors
- **Retriever**: component that fetches relevant docs/chunks for a query
- **RAG**: retrieval + generation for grounded answers

