import asyncio
import shutil
import time
from pathlib import Path 
from fastapi import FastAPI, Form, UploadFile, BackgroundTasks, File
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore,Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import qdrant_client

load_dotenv()
app = FastAPI()
Path("docs").mkdir(exist_ok=True)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model
)

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def process_and_index_pdf(save_path: Path):
    """
    This is the long-running function that will be executed in the background.
    Note: This is a regular 'def' function, not 'async def'.
    """
    print(f"BACKGROUND TASK: Starting indexing for {save_path.name}...")

    # 1. Load the PDF
    loader = PyPDFLoader(file_path=str(save_path))
    docs = loader.load()

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400
    )
    chunks = text_splitter.split_documents(documents=docs)
    print(f"BACKGROUND TASK: PDF split into {len(chunks)} chunks.")

    # 3. Vector Embeddings Model
    # Using the same model as the main app to ensure consistency and fix the 404 error.
    embedding_model_bg = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # 4. Initialize Qdrant client and vector store
    qdrant_client_bg = qdrant_client.QdrantClient(url="http://localhost:6333")
    vector_store = Qdrant(
        client=qdrant_client_bg,
        collection_name="learning_rag",
        embeddings=embedding_model_bg,
    )

    # 5. Batch processing
    batch_size = 10
    delay_in_seconds = 60

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"BACKGROUND TASK: Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}...")
        
        vector_store.add_documents(batch)
        
        if i + batch_size < len(chunks):
            print(f"BACKGROUND TASK: Batch added. Waiting for {delay_in_seconds} seconds...")
            time.sleep(delay_in_seconds)

    print(f"BACKGROUND TASK: Indexing of {save_path.name} is complete.")


@app.post("/upload")
async def handle_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Receives a file, saves it, and schedules the indexing to run in the background.
    Returns an immediate response to the user.
    """
    save_path = Path("docs") / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(process_and_index_pdf, save_path)

    return {
        "filename": file.filename,
        "status": "File received. Indexing has started in the background."
    }

@app.post("/askQuery")
async def home(query: str = Form(...)):
    """
    This is an asynchronous route in FastAPI.
    - It's async by default.
    - FastAPI uses Python type hints for data validation.
      `query: str = Form(...)` tells FastAPI to expect a form field named 'query'
      and ensures it's a string.
    """
    print(f"User Query is: {query}")

    search_results = vector_db.similarity_search(query=query)
    context = "\n\n\n".join([f"Page Content: {results.page_content}\nPage Number: {results.metadata['page_label']}\nFile Location: {results.metadata['source']}" for results in search_results])

    SYSTEM_PROMPT = f"""
    You are a helpul AI Assistance who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

    You should only answer the user based on the following context and navigate the user to open the right page number to know more.

    Context: {context}
    """

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": query
            }
        ]
    )
    return {'Response': response.choices[0].message.content}