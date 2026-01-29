# LangChain Tutorial (Beginner → Advanced)
This guide starts with **beginner-friendly** LangChain fundamentals, then gradually moves into **intermediate** and **advanced** patterns (RAG that actually runs, and agents built with LangGraph).

It’s written for this repo’s setup (Python + `langchain` + `langchain-openai` + `python-dotenv`).

## Who this is for
- **Beginner**: you’re new to LangChain/LLMs and want step-by-step examples.
- **Intermediate**: you can call a model already, but want better prompts/chains, streaming/async, and structured outputs.
- **Advanced**: you want **working RAG**, better retrieval strategies, and **tool-using agents** with memory/checkpointing.

## Versions used in this repo (important)
The code snippets aim to match what you have installed here:
- `langchain==1.2.6`
- `langchain-core==1.2.7`
- `langchain-openai==1.1.7`
- `langgraph-prebuilt==1.0.6`

If you install optional “RAG extras” later (vector stores, splitters), you’ll add a few more packages (explained below).

## Table of contents
- Part 1 — Your first chat model call
- Part 2 — Prompts (templates with variables)
- Part 3 — Chains (LCEL)
- Part 4 — Structured output (basic)
- Part 5 — RAG basics
- Part 6 — Minimal RAG (conceptual)
- Part 7 — Tools and Agents (high-level)
- Part 8 — Good project habits
- Part 9 — Intermediate: streaming, async, batching, retries, config
- Part 10 — Intermediate: structured outputs with Pydantic (reliable JSON)
- Part 11 — Advanced: working RAG (index → retrieve → answer) + citations
- Part 12 — Advanced: better retrieval (MMR, query rewriting, chunking strategy)
- Part 13 — Advanced: agents with LangGraph (tools + memory/checkpointing)
- Part 14 — Advanced: tracing/evaluation and production tips

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

If you want a **working** RAG implementation you can run locally, jump to **Part 11**.

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

In this repo/version setup, the most practical way to build agents is **LangGraph** (see **Part 13** for a working example using `create_react_agent`).

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

---

## Part 9 — Intermediate: streaming, async, batching, retries, config
Once “hello world” works, these are the features you’ll use constantly in real apps.

### Streaming (token-by-token output)
Useful for CLI chat apps and web UIs.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)

for chunk in llm.stream("Write a 2-line poem about RAG."):
    # chunk is a message-like object; content may arrive in pieces
    print(chunk.content, end="", flush=True)
print()
```

### Async (for web servers / concurrency)

```python
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

async def main():
    resp = await llm.ainvoke("Give me 3 ideas for an AI agent.")
    print(resp.content)

asyncio.run(main())
```

### Batching (many prompts at once)
Batching is a big cost/latency win for ingestion, evaluation, and bulk tasks.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompts = [
    "Summarize LangChain in 1 sentence.",
    "Summarize RAG in 1 sentence.",
    "Summarize agents in 1 sentence.",
]

results = llm.batch(prompts)
for r in results:
    print("-", r.content)
```

### Timeouts and basic reliability
Your `ChatOpenAI` supports a `timeout` parameter. In production, also add retries around transient failures.

```python
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)

@retry(wait=wait_exponential_jitter(initial=1, max=10), stop=stop_after_attempt(3))
def safe_invoke(prompt: str) -> str:
    return llm.invoke(prompt).content

print(safe_invoke("Return a single word: OK"))
```

### Passing metadata/tags (good for tracing later)
LangChain components support tags/metadata used by tracing/observability tools.

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

config = RunnableConfig(tags=["tutorial", "demo"], metadata={"feature": "tags"})
resp = llm.invoke("Say 'hi'", config=config)
print(resp.content)
```

---

## Part 10 — Intermediate: structured outputs with Pydantic (reliable JSON)
For real apps, “please output JSON” is often not enough. A better pattern:
- define a Pydantic model (the schema you want)
- tell the model the schema rules
- parse/validate automatically

```python
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

class StudyPlan(BaseModel):
    topic: str = Field(..., description="What the user wants to learn")
    days: int = Field(..., ge=1, le=30, description="Number of days")
    daily_tasks: list[str] = Field(..., description="Simple daily tasks")

parser = PydanticOutputParser(pydantic_object=StudyPlan)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a careful assistant. Output must follow the schema exactly."),
    ("human", "Create a {days}-day plan to learn: {topic}.\n{format_instructions}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

plan = chain.invoke({"topic": "LangChain + RAG", "days": 7})
print(plan.topic)
print(plan.days)
print(plan.daily_tasks[:2])
```

Why this matters:
- You can fail fast if the model output is invalid.
- Your downstream code can rely on the schema.

---

## Part 11 — Advanced: working RAG (index → retrieve → answer) + citations
This section gives you an end-to-end RAG pipeline you can actually run locally.

### Install RAG extras
These packages aren’t in this repo by default (to keep the base install small):

```bash
pip install langchain-community langchain-text-splitters faiss-cpu
```

If you prefer Chroma instead of FAISS:

```bash
pip install langchain-community langchain-text-splitters chromadb
```

### Example: index local markdown files and answer questions
This example:
- reads `.md` files
- splits them into chunks
- embeds chunks with OpenAI embeddings
- stores vectors in FAISS (in-memory)
- retrieves top-k chunks for a question
- asks the LLM using retrieved context
- prints **citations** (which files were used)

```python
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

def load_markdown_docs(root: str = ".") -> list[Document]:
    docs: list[Document] = []
    for path in Path(root).rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs

docs = load_markdown_docs(".")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[source={d.metadata.get('source','?')}]\n{d.page_content}"
        for d in docs
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the provided context. If missing, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "What is this repository building?"
answer = rag_chain.invoke(question)
print("Answer:\n", answer)

# Optional: show which sources were retrieved
retrieved = retriever.invoke(question)
print("\nCitations:")
for d in retrieved:
    print("-", d.metadata.get("source"))
```

### RAG “guardrails” (important)
- **Tell the model to stick to context** (and to say “I don’t know” when missing).
- **Keep chunks reasonable** (too big = irrelevant noise, too small = missing meaning).
- **Show sources** so you can debug retrieval.

---

## Part 12 — Advanced: better retrieval (MMR, query rewriting, chunking strategy)
Once basic RAG works, quality improvements come from retrieval strategy.

### Chunking strategy (big impact)
- Start with `chunk_size=800–1500`, `overlap=100–200`.
- Split on structure (headings/paragraphs) when possible.
- Store metadata: file path, section title, URL, timestamps.

### Retrieval strategy improvements
- **MMR retrieval**: tries to balance relevance + diversity (reduces duplicates).
- **Query rewriting**: rewrite user query into a better “search query”.
- **Multi-query retrieval**: generate 3–5 alternative queries, retrieve for each, merge results.
- **Hybrid retrieval**: combine keyword search + vector search.
- **Reranking**: retrieve \(k=20\), rerank to top \(k=4\).

These often require extra components/packages (and depend on your vector store), but the *concept* stays the same:
retrieve better context → the LLM answers better.

---

## Part 13 — Advanced: agents with LangGraph (tools + memory/checkpointing)
In your current setup, “agents” are best built using **LangGraph prebuilt agents**.

### A minimal tool-using agent (ReAct style)

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two integers.\"\"\"
    return a * b

@tool
def add(a: int, b: int) -> int:
    \"\"\"Add two integers.\"\"\"
    return a + b

tools = [multiply, add]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# MemorySaver stores conversation state in-memory (great for local dev)
checkpointer = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
)

# "thread_id" lets you keep multiple independent conversations
config = {"configurable": {"thread_id": "demo-thread"}}

result = agent.invoke(
    {"messages": [("user", "What is (12 * 7) + 5? Use tools.")]}
    , config=config
)

print(result["messages"][-1].content)
```

### Why LangGraph is useful
- You can add **memory/checkpointing** cleanly.
- You can add **interrupts** (approve tool calls, human-in-the-loop).
- You can build **multi-step graphs** (RAG node → tool node → answer node).

---

## Part 14 — Advanced: tracing/evaluation and production tips

### Tracing (recommended once you have >1 step)
When you have prompts + retrieval + tool calls, debugging becomes hard without traces.
You already have `langsmith` installed; you can enable tracing by setting env vars:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=ai-agent-with-langchain
```

### Evaluation (how to know your RAG is “good”)
Good evaluation is a whole topic, but a simple starter loop:
- Create 20–50 representative questions
- Define what a correct answer looks like (and which sources should be used)
- Run your pipeline and score:
  - answer correctness
  - citation correctness
  - “I don’t know” when appropriate

### Production tips (quick)
- **Never trust model output blindly** (validate schemas, sanitize tool inputs).
- **Limit tool permissions** (principle of least privilege).
- **Keep prompts versioned** (changes affect behavior).
- **Watch costs** (batching, caching, small models where possible).
- **Log retrieval** (k, scores, sources) for debugging.

