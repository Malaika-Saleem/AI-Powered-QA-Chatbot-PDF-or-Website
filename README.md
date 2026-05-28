# 📄 PDF Chatbot with Streamlit + LangChain

A conversational AI chatbot that allows users to upload PDF documents and ask questions about their content using **LangChain**, **Streamlit**, **Hugging Face Embeddings**, **ChromaDB**, and **OpenRouter/OpenAI models**.

---

## 🚀 Features

* Upload and analyze PDF documents
* Chat with your PDF in natural language
* Conversational memory for context-aware responses
* Semantic search using vector embeddings
* Local vector database with Chroma
* Free Hugging Face embedding model support
* OpenRouter API integration for LLM access
* Simple and interactive Streamlit UI

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **LLM Framework:** LangChain
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Database:** ChromaDB
* **LLM Provider:** OpenRouter
* **PDF Processing:** PyPDF Loader (custom utility)

---

## 📂 Project Structure

```bash
project/
│
├── app.py
├── .env
├── requirements.txt
│
├── utils/
│   └── pdf_loader.py
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

---

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Get your API key from:

* OpenRouter: https://openrouter.ai/

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. User uploads a PDF document.
2. The PDF text is extracted and split into chunks.
3. Chunks are converted into embeddings using Hugging Face.
4. Embeddings are stored in ChromaDB.
5. User asks questions.
6. LangChain retrieves relevant chunks.
7. OpenRouter LLM generates contextual answers.

---

## 🧠 Core Components

### PDF Loading

```python
docs = load_pdf(tmp_path)
```

### Text Splitting

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

### Embeddings

```python
HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
```

### Vector Store

```python
Chroma.from_documents(chunks, embeddings)
```

### Conversational Retrieval Chain

```python
ConversationalRetrievalChain.from_llm(...)
```

---

## 📸 UI Preview

* Upload PDF
* Ask questions
* Receive AI-generated answers
* Persistent chat history during session

---

## 📋 Requirements

Example `requirements.txt`:

```txt
streamlit
python-dotenv
langchain
langchain-community
chromadb
sentence-transformers
openai
pypdf
```

---

## 🔒 Notes

* Ensure your OpenRouter API key is valid.
* Large PDFs may take longer to process.
* Internet connection is required for LLM responses.

---

## 🚧 Future Improvements

* Multi-PDF support
* Persistent vector database
* Chat export feature
* Streaming responses
* Source citations for answers
* Better UI/UX enhancements

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Developed using Streamlit + LangChain + OpenRouter.

