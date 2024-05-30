from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
#from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


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

#print(all_splits)


vector_store = FAISS.from_texts([""], embeddings)

batch_size = 100;
total_docs = len(all_splits);

for i in range(0, total_docs, batch_size):
    batch = all_splits[i:i + batch_size]
    #print(batch)
    vector_store.add_documents(batch)




results = vector_store.similarity_search("questions")

print(results)
print("all splits:"+str(len(all_splits)))
print("results:"+str(len(results)))


