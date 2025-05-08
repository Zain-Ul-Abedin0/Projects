import os
import streamlit as st
import Api_key from key
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = Api_key

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.6, max_tokens=500)

# Streamlit UI setup
st.title("Internet Research Tool ")
st.sidebar.title("News Article URLs")

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

index_dir = "faiss_index_store"
main_placeholder = st.empty()

# When Process Button is Clicked
if process_url_clicked:
    # Load URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading Data.......")
    data = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Splitting Text.......")
    docs = text_splitter.split_documents(data)

    # Create embeddings and FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Embedding Vector Building.......")
    time.sleep(2)

    # Save FAISS index locally
    vectorstore.save_local(index_dir)

# Handle Query
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_dir):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display result
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    else:
        st.warning("Please process the URLs first!")
