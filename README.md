# AI Agent with LangChain

## Project Overview

This project aims to build an intelligent AI agent using **LangChain** and **RAG (Retrieval-Augmented Generation)** technology. The agent will leverage LangChain's powerful framework to create sophisticated conversational AI capabilities with enhanced context understanding through retrieval-augmented generation.

## About LangChain

**LangChain** is an open-source framework designed to simplify the development of applications powered by large language models (LLMs). It provides a comprehensive toolkit for building AI applications that can reason, interact with data, and perform complex tasks.

### Key Features of LangChain

- **Chain Composition**: Build complex workflows by chaining together multiple LLM calls and other components
- **Memory Management**: Maintain conversation context and state across interactions
- **Agent Framework**: Create autonomous agents that can use tools, make decisions, and interact with external systems
- **Document Loaders**: Easily load and process documents from various sources (PDFs, web pages, databases, etc.)
- **Vector Stores**: Integrate with vector databases for semantic search and retrieval
- **Prompt Management**: Template and manage prompts efficiently
- **Output Parsers**: Structure and validate LLM outputs
- **Callbacks & Observability**: Monitor and debug LLM applications
- **Multi-Model Support**: Work with various LLM providers (OpenAI, Anthropic, Hugging Face, etc.)

## Project Goals

- Implement a RAG-based AI agent that can retrieve relevant information from knowledge bases
- Create an intelligent conversational interface using LangChain
- Build a system that combines retrieval mechanisms with generative AI for accurate, context-aware responses
- Explore advanced LangChain features including agents, chains, and memory management

## Technology Stack

- **LangChain**: Core framework for LLM application development
- **LangChain OpenAI**: OpenAI integration for LangChain
- **OpenAI**: Large Language Model API
- **RAG**: Retrieval-Augmented Generation for enhanced AI capabilities
- **Python**: Primary programming language
- **python-dotenv**: Environment variable management

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/RamanaGR/AIAgent-with-LangChain.git
   cd AIAgent-with-LangChain
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Start from the example file:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and replace the placeholder with your real OpenAI key:
     ```bash
     OPENAI_API_KEY=sk-REPLACE_ME_WITH_YOUR_KEY
     ```

### Running the Example

```bash
python example.py
```

### Project Structure (simplified)

```
AIAgent-with-LangChain/
├── venv/                  # Virtual environment (not tracked)
├── .env / .env.example    # Environment variables (local / template)
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
├── README.md              # Top-level project overview
├── documentation/         # Additional docs & tutorials
├── rag/                   # RAG demos (storage, retrieval, metadata)
├── Agents_basics.py       # Simple ReAct-style agent with a time tool
├── chains_*.py            # Chain composition examples
└── example.py             # Minimal LangChain example
```
