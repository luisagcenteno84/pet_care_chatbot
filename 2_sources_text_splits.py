from langchain_community.vectorstores.chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
#from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma("pet_care_rag", embeddings)

sources = [
    'https://www.petsmart.com/learning-center/dog-care/risks-and-benefits-of-raw-dog-food/A0286.html?fdid=dog',
    'https://www.petsmart.com/learning-center/dog-care/a-guide-to-supporting-your-dogs-hip-and-joint-health/A0148.html?fdid=dog',
    'https://www.petsmart.com/learning-center/dog-care/what-should-i-feed-my-dog/A0209.html?fdid=dog'
]

loader = WebBaseLoader(sources)
#loader = PyPDFLoader(file_path="resume.pdf")
data = loader.load()
#print(data)

text_splitters = RecursiveCharacterTextSplitter(chunk_size=250,chunk_overlap=50)
all_splits = text_splitters.split_documents(data)

print(all_splits)

#vector_store.add_documents(all_splits)

#print(vector_store.similarity_search("Databricks"))
