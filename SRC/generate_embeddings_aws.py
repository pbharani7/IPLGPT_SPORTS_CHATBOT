import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_pinecone import PineconeVectorStore
import os
#pinecone_index = os.environ["PINECONE_INDEX"]
##="ipl-2022-23-24"

PINECONE_API_KEY = "b170bcfa-6542-4672-b5da-d85349a77519"
PINECONE_INDEX = "ipl-2022-23-24"

'''
def download_objects_in_folder(bucket_name, folder_name):
    s3 = boto3.client('s3')

    # Define the prefix to list objects only from the specified folder
    prefix = folder_name + '/'

    object_keys = []  # Initialize an empty list to store object keys

    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        print(response)
        # Check if objects were found
        if 'Contents' in response:
            objects = response['Contents']
            for obj in objects:
                if obj != "cleaned/":  # Check if the key is not just the folder name
                    object_keys.append(obj)  # Extracting file name from the object key
        else:
            print(f"No objects found in folder '{folder_name}' in bucket '{bucket_name}'")
    except Exception as e:
        print(f"Error listing objects in folder '{folder_name}' in bucket '{bucket_name}': {e}")

    for file_name in object_keys:
        try:
            s3.download_file(bucket_name, file_name, file_name)
            print(f"File downloaded successfully to: {file_name}")
        except Exception as e:
            print(f"Error downloading file '{file_name}' from bucket '{bucket_name}': {e}")
    
    return object_keys
'''


def download_s3_files(bucket_name, folder_name, local_directory):
    """
    Downloads files from Amazon S3 to the local machine.

    Args:
        bucket_name (str): The name of the S3 bucket.
        folder_name (str): The name of the folder in the bucket.
        local_directory (str): The local directory where the files should be downloaded.
    """
    s3 = boto3.client('s3')
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)['Contents']
    print(objects)
    for obj in objects:
        object_key = obj['Key']
        if object_key != f"{folder_name}/":
            local_file_path = f"{object_key}"
            print("#########################",local_file_path)
            s3.download_file(bucket_name, object_key, local_file_path)
            print(f"File '{object_key}' downloaded successfully to: {local_file_path}")




def generate_embeddings(directory, pinecone_index, chunk_size, chunk_overlap):
            
            loader = DirectoryLoader('cleaned/', glob="*.txt")
            text_documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents = text_splitter.split_documents(text_documents)

            embeddings = OpenAIEmbeddings()

            pinecone = PineconeVectorStore.from_documents(documents, embeddings, index_name=pinecone_index)



# Example usage:
bucket_name = "iplbucketmy"
folder_name = "cleaned"
download_s3_files(bucket_name, folder_name,folder_name)

generate_embeddings(folder_name, os.environ["PINECONE_INDEX"], 1000, 50)
