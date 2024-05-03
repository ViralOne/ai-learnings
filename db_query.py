import argparse
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

chroma_dir = "docs/chroma"

def main(text):
    embedding_function = GPT4AllEmbeddings(device='gpu')
    db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embedding_function
    )
    query_text = db.similarity_search_with_score(text, k=3)
    return query_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = main(args.query_text)

    query_result = [doc.page_content for doc, _score in query_text]
    for result in query_result:
        for line in result.split('\n'):
            print(line)

    # Get sources
    sources = {doc.metadata.get("source", None) for doc, _score in query_text}

    print(f"Sources: {', '.join(sources)}")
