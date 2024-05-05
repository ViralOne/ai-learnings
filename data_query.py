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
Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic.
If you don't know the answer, just say "I do not know." Don't make up an answer.
Answer the questions as only one question: {question}
DO NOT give irelevant information that is not in the context.
Remember the correct word for the company is AgileFreaks.
Context: {context} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
---
"""

AUGUMENT_QUERY_TEMPLATE = """
You are a helpful expert research assistant. Your users are asking questions about the company.
Suggest a new question and make sure that the original question does not loose its meaning, with more details to the question. Suggest a similar slightly changed question that have the same meaning.
Question: {question}
Only output the new suggested question nothing more!
"""

# Constants
cache = SQLiteCache(database_path=LLM_CACHE_PATH)

def prepare_db(embedding_function):
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def prepare_llm(openai_api_base, model_name):
    return ChatOpenAI(openai_api_base=openai_api_base, model_name=model_name)

def prepare_memory(llm, chat_memory, memory_key, return_messages):
    return ConversationSummaryMemory(llm=llm, chat_memory=chat_memory, memory_key=memory_key, return_messages=return_messages)

def llm_chain(llm, cache, memory):
    langchain.llm_cache = cache
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
    return conversation

def augment_query_generated(query, llm, memory):
    # https://arxiv.org/abs/2305.03653
    prompt_template = ChatPromptTemplate.from_template(AUGUMENT_QUERY_TEMPLATE)
    prompt = prompt_template.format(question=query)

    response_text = llm_chain(llm, cache, memory).predict(input=prompt)

    return response_text

def query_db(query_text, db, llm, memory):
    # Retrieve search results
    results = db.similarity_search_with_score(query_text, k=5)

    # Create a set to store unique page content
    unique_texts = set()

    # Iterate through the results and add unique page content to the set
    for doc, _score in results:
        unique_texts.add(doc.page_content)

    # Join the unique page content with the specified separator
    context_text = "\n\n---\n\n".join(unique_texts)
    sources = {doc.metadata.get("source", None) for doc, _score in results}

    return context_text, sources

def process_query(query_text, context, sources, llm, memory):
    start_time = time.time()
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query_text)

    response_text = llm_chain(llm, cache, memory).predict(input=prompt)
    formatted_response = f"Response: {response_text}\nSources: {', '.join(sources)}"
    end_time = time.time()
    print(formatted_response)
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = GPT4AllEmbeddings(device='gpu')
    db = prepare_db(embedding_function)
    llm = prepare_llm(openai_api_base="http://localhost:3008/v1", model_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF")
    memory = prepare_memory(llm, FileChatMessageHistory(file_path='conversation_history.txt'), "history", True)

    # Expand the query
    query_text = augment_query_generated(query_text, llm, memory)

    # Query DB
    db_query, sources = query_db(query_text, db, llm, memory)

    process_query(query_text, db_query, sources, llm, memory)