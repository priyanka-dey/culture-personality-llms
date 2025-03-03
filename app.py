from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")
os.environ["OPEN_AI_API"] = os.getenv("OPEN_AI_API")
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
open_ai_api_key = os.getenv("OPEN_AI_API")

# Paths and API keys
root_directory = ".\Documents"
documents = list()

# Function to initialize chatbot components
@st.cache_resource
def initialize_chatbot():
    # Load documents
    for folder, _, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(folder, file)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
    chunked_documents = []
    
    for doc in tqdm(documents, desc="Splitting Documents"):
        chunks = text_splitter.split_documents([doc])
        chunked_documents.extend(chunks)
    
    # Generate embeddings using OpenAI's model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=open_ai_api_key)
    
    # Create FAISS vector store
    vector_db = FAISS.from_documents(chunked_documents, embedding_model)

    # Initialize LLM (Llama with Groq)
    llm = ChatGroq(model='llama-3.2-11b-text-preview', groq_api_key=groq_api_key)

    # 5. Memory setup
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # Key used to store conversation history
        return_messages=True        # Allows memory to be added to chain responses
    )

    # 6. Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=memory,
        chain_type="stuff"
    )
    
    return qa_chain

# Initialize the chatbot
qa_chain = initialize_chatbot()

# Streamlit app setup
st.title("Conversational Chatbot on Cultural Norms")
st.markdown("A minimalistic chatbot for querying cultural values and aspects. Ask your questions below!")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Chat interface
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:", placeholder="Ask something about a country's cultural norms...")
    submit = st.form_submit_button("Send")

# Handle user input and chatbot response
if submit and user_input:
    # Append user input to chat history
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    # Get chatbot response
    bot_response = qa_chain.run(user_input)
    st.session_state["chat_history"].append({"role": "bot", "content": bot_response})

# Display chat history
st.divider()  # Add a visual separator
st.subheader("Chat History")
for message in st.session_state["chat_history"]:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Chatbot:** {message['content']}")
