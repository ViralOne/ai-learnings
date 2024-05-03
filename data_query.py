import argparse
import os
import time
import langchain
import langchain.llms
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.cache import SQLiteCache

os.environ["OPENAI_API_KEY"] = "no_need"

CHROMA_PATH = "docs/chroma"
LLM_CACHE_PATH = "docs/llm_cache.db"

PROMPT_TEMPLATE = """
<|start_header_id|>user<|end_header_id|>
You are an assistant for answering questions about the company HandBook.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
Answer the question: {question}
DO NOT give irelevant information that is not in the context.
Remember the correct word for the company is AgileFreaks.
Context: {context} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
---
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
    message_history = FileChatMessageHistory(file_path = 'conversation_history.txt')

    memory = ConversationSummaryMemory(
        llm=model,
        chat_memory=message_history,
        memory_key="history",
        return_messages=True
    )

    langchain.llm_cache = SQLiteCache(database_path=LLM_CACHE_PATH)

    conversation = ConversationChain(
        llm=model,
        memory=memory,
        verbose=False
    )

    start_time = time.time()
    results = db.similarity_search_with_score(query_text, k=5)
    relevance_score = results[0][1]
    if relevance_score < 0 or relevance_score > 1:
        print("Invalid relevance score. Please check your data.")
    else:
        if len(results) == 0 or results[0][1] <= 0.4:
            print(f"Unable to find matching results.")
            return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = conversation.predict(input=prompt)

    sources = {doc.metadata.get("source", None) for doc, _score in results}
    formatted_response = f"Response: {response_text}\nSources: {', '.join(sources)}"
    end_time = time.time()
    print(formatted_response)
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
