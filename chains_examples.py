from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompt_templates import prompt_template


load_dotenv()

chatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


prompt_template = ChatPromptTemplate.from_template(
    messages=[
        (
            "system",
            "You are a fact expert that can answer questions about the following : {animal}",
        ),
        ("human", "Tell me {fact_count} facts."),
    ]
)
# Using Pipe Syntax LCEL (LangChain Expression Language)
chain = prompt_template | chatOpenAI | StrOutputParser()
response = chain.invoke(
    {
        "animal": "dog",
        "fact_count": 1,
    }
)
print(response)


# Inner Chains

#creating individual runnable chains
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_prompt = RunnableLambda(lambda x: prompt_template.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_prompt], last=parse_output)

#Run the chain
response = chain.invoke(
    {
        "animal": "dog",
        "fact_count": 1,
    }
)
print(response)


# Create the sequential chain using LCEL syntax
# Define prompt templates
animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}."),
        ("human", "Tell me {count} facts."),
    ]
)

# Define a prompt template for translation to French
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)

# Define additional processing steps using RunnableLambda
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "french"})


# Create the combined chain using LangChain Expression Language (LCEL)
chain = animal_facts_template | chatOpenAI | StrOutputParser() | prepare_for_translation | translation_template | chatOpenAI | StrOutputParser() 

# Run the chain
result = chain.invoke({"animal": "cat", "count": 2})

# Output
print(result)



