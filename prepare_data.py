import os
import logging
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

logging.basicConfig(level=logging.INFO)

SOURCE_DIR = "docs/handbook"
DESNTINATION_DIR = "docs/handbook_files"
CHROMA_DIR = "docs/chroma"
FILE_PATERN = '---'

def clone():
    """Clone the Handbook repository."""
    os.system(f"git clone https://github.com/AgileFreaks/handbook.git {SOURCE_DIR}")

def move_files(source_dir, destination_dir):
    """Move Markdown files from source directory to destination directory.
    
    Args:
        source_dir (str): The directory to search for Markdown files.
        destination_dir (str): The directory to move the files to.

    Returns:
        None
    """
    # Check if the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        logging.info(f"Created directory '{destination_dir}'")

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".md"):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_dir, file)

                try:
                    os.rename(source_path, destination_path)
                    logging.info(f"Moved '{source_path}' to '{destination_path}'")
                except Exception as e:
                    logging.error(f"Error moving file: {e}")

def clean_files(directory, start_pattern):
    """
    Remove lines starting with the specified pattern from Markdown files in the specified directory.

    Args:
        directory (str): The directory containing Markdown files to process.
        start_pattern (str): The pattern to search for in the first line of the file.

    Returns:
        None
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Check if the first line starts with the specified pattern
            if lines and lines[0].startswith(start_pattern):
                del lines[:7]  # Delete lines 1 to 7

                # Write the modified content back to the file
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.writelines(lines)
                logging.info(f"Lines deleted successfully in {filename}.")

def split_data(directory):
    """
    Split Markdown files in the specified directory into chunks.

    Args:
        directory (str): The directory containing Markdown files to process.

    Returns:
        list: A list of chunked text data.
    """

    loader = DirectoryLoader(
        directory, 
        glob="*.md", 
        loader_cls=UnstructuredMarkdownLoader, 
        use_multithreading=True
        )
    pages = loader.load()

    logging.info(f"Loaded {len(pages)} pages")

    txt = ' '.join([d.page_content for d in pages])
    logging.info(f"Text length: {len(txt)}")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n#{1,6}", "\n\n", "\n", "__"],
        keep_separator=True,
        chunk_size=1000,
        chunk_overlap=250,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(pages)
    logging.info(f"Split into {len(chunks)} chunks")

    return chunks

def save_chunks(chunks, chroma_dir):
    """Process chunks into vectors and save them to directory."""
    embedding_function = GPT4AllEmbeddings(device='gpu')
    try:
        Chroma.from_documents(
            documents=chunks, 
            embedding=embedding_function, 
            persist_directory=chroma_dir
        )
        logging.info(f"Chunks saved to {chroma_dir}")
    except Exception as e:
        logging.error(f"Error saving chunks: {e}")

def main():
    # # Check if the cloned repo exists
    clone_repo = True
    if not os.path.exists(SOURCE_DIR) and clone_repo == True:
        clone()

    # Check if the destination directory exists
    if not os.path.exists(DESNTINATION_DIR):
        move_files(SOURCE_DIR, DESNTINATION_DIR)

    # Clean data
    clean_files(DESNTINATION_DIR, FILE_PATERN)

    # Split data
    chunks = split_data(DESNTINATION_DIR)

    # Save chunks
    save_chunks(chunks, CHROMA_DIR)

if __name__ == "__main__":
    main()
