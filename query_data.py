import argparse
import os
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

os.environ["OPENAI_API_KEY"] = "no_need"

CHROMA_PATH = "docs/chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
DO NOT give irelevant information that is not in the context.
Remember the correct word for the company is AgileFreaks.
---
Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = GPT4AllEmbeddings(device='gpu')
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Prepare the LLM.
    model = ChatOpenAI(openai_api_base="http://localhost:3008/v1", model_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF")

    # Prepare the memory.
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

    conversation = ConversationChain(
        llm=model,
        memory=memory,
        verbose=False
    )
    
    results = db.similarity_search_with_score(query_text, k=3)
    # results = db.similarity_search_with_relevance_scores(query_text, k=3) # it does not return anything
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = conversation.predict(input=prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
