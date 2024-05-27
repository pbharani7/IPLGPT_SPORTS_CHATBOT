import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


def setup_environment():
    load_dotenv()

def generate_embeddings(files, pinecone_index, chunk_size, chunk_overlap):

    processed_dir = os.path.join(os.getcwd(), 'processed')  # Define the processed directory path

    # Create the processed directory if it doesn't exist
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for file in files:
        loader = TextLoader(file)
        text_documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(text_documents)

        embeddings = OpenAIEmbeddings()

        pinecone = PineconeVectorStore.from_documents(documents, embeddings, index_name=pinecone_index)
        
        # Move the processed file to the processed directory
        processed_file_path = os.path.join(processed_dir, os.path.basename(file))

        # Rename or move the file using os.rename or os.replace
        try:
            os.rename(file, processed_file_path)  # Use os.rename for Python 3.3+
        except OSError:
            os.replace(file, processed_file_path)  # Use os.replace for Python 3.4+

        print(f"File {file} moved to 'processed' directory.")

FILES = glob.glob(os.path.join(os.getcwd(), 'Cric*.txt'))
print(FILES)
setup_environment()
generate_embeddings(FILES, os.environ["PINECONE_INDEX"], 6000, 100)