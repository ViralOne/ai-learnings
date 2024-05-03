import argparse
import os
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

CHROMA_PATH = "docs/chroma"
os.environ["OPENAI_API_KEY"] = "no_need"

template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

def main():
    embedding_function = GPT4AllEmbeddings(device='gpu')
    model = ChatOpenAI(openai_api_base="http://localhost:3008/v1", model_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question"
    )

    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type='stuff',
        retriever=db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3},
        ),
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory
        }
    )
    prompt_1_details = qa.run({"query": "My name is Joe, an engineer and I am a freak, currently I am at Advanced level."})
    promp_2 = qa.run("What is my level and name and how can I get to the next Skill level and what is called the next skill level?")
    print(promp_2)

if __name__ == "__main__":
    main()
