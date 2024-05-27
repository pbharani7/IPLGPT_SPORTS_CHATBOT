import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Pinecone
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def setup_environment():
    load_dotenv()

def get_vectorstore(index_name):
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=OpenAIEmbeddings())
    return vectorstore

def get_context_retriever(vectorstore):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    #Generating the prompt template
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}. If you can't answer the question, reply 'I don't know'"), # This is a system message we are telling the AI what it is supposed to do
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

    stuff_doc_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain,stuff_doc_chain)

def get_answer(user_input):
    
    retriever_chain = get_context_retriever(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response =  conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response["answer"]

#The way streamlit works is it initialise the entire code after an event.
setup_environment()  

st.set_page_config(page_title="IplGPT", page_icon="🤖")
st.title("🏏 IplGPT")

# Such initialisation gives us a problem because everytime an event happens the variable is re-initialised and we lose the session state.
#chat_history = [AIMessage(content = "Hello, I am a bot. How can I help you ?")]

#Solution is to use session.state which is persistent throughout the session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello,How can I help you ?")]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore(os.environ["PINECONE_INDEX"]) 


#     st.header("Settings")
#     website_url = st.text_input("Website URL")
    
user_query = st.chat_input("Ask me a question about IPL...")

if user_query is not None and user_query != "":
    
    response = get_answer(user_query)

    retriever_chain = get_context_retriever(st.session_state.vector_store)
    #st.write(response)
    
    st.session_state.chat_history.append(HumanMessage(content = user_query))
    st.session_state.chat_history.append(AIMessage(content = response))

    # with st.chat_message("Human"):
    #     st.write(user_query)
    
    # with st.chat_message("AI"):
    #     st.write(response)

    retrieved_documents = retriever_chain.invoke({
        "chat_history":st.session_state.chat_history,
         "input": user_query
     })

    #st.write(retrieved_documents)  #Very Important

    

for message in st.session_state.chat_history:  
    if isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)       

#For Debugging
# with st.sidebar:
#     st.write(st.session_state.chat_history)
#