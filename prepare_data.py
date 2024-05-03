import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

source_dir = "handbook"
destination_dir = "docs/handbook_files"
chroma_dir = "docs/chroma"

def clone():
    os.system("git clone https://github.com/AgileFreaks/handbook.git")

def move_files():
    os.makedirs(destination_dir)  # Recursive directory creation
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".md"):
                # Construct the source and destination paths
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_dir, file)
                
                # Move the file to the destination directory
                shutil.move(source_path, destination_path)
                print(f"Moved '{source_path}' to '{destination_path}'")

def split_data():
    # Load entire directory
    loader = DirectoryLoader(destination_dir, glob="*.md", loader_cls=UnstructuredMarkdownLoader, use_multithreading=True)
    pages = loader.load()

    # Get Text from all pages
    txt = ' '.join([d.page_content for d in pages])

    print("Pages: ",len(pages))
    print("Text lenght: ", len(txt))

    # Split Text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n"
        ],
        keep_separator=True,
        chunk_size = 1000,
        chunk_overlap = 250,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(pages)
    print("Chunks: ",len(chunks))
    return chunks

def save_chunks(chunks):
    embedding_function = GPT4AllEmbeddings(device='gpu')
    Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function,
        persist_directory=chroma_dir
        )
    print(f"Chunks saved to {chroma_dir}")

def main():
    # Check if the cloned repo exists
    clone_repo = False
    if not os.path.exists(source_dir) and clone_repo == True:
        clone()

    # Check if the destination directory exists
    if not os.path.exists(destination_dir):
        move_files()

    # Split data
    chunks = split_data()

    # Save chunks
    save_chunks(chunks)

if __name__ == "__main__":
    main()
