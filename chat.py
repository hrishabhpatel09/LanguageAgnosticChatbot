from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

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

# take user input
user_query = input("Ask Something..\n")

#Relevant chunks
search_results = vector_db.similarity_search(query=user_query)

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
            "content": user_query
        }
    ]
)

print(response.choices[0].message.content)