import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils.pdf_loader import load_pdf

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading and processing..."):
        # Load and split
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        docs = load_pdf(tmp_path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Embeddings and Vector Store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        vectordb = Chroma.from_documents(chunks, embeddings)

        # Conversation Chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(openai_api_key=openai_key),
            retriever=vectordb.as_retriever(),
            memory=memory
        )

        # Chat interface
        st.subheader("💬 Ask your questions")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_query = st.text_input("Your question:")
        if user_query:
            result = chain({"question": user_query})
            st.session_state.chat_history.append((user_query, result["answer"]))

        for q, a in st.session_state.chat_history[::-1]:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
