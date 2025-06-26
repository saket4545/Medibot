from langchain_community.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS





DATA_PATH='data/'
def load_pdf(data):
    loader= DirectoryLoader(data,
                            glob='*.pdf',
                            loader_cls=PyMuPDFLoader)

    doucument=loader.lazy_load()
    return doucument

documents=load_pdf(data=DATA_PATH)
print("The length of documents is",len(documents))

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("length of chunks",len(text_chunks))

    
def get_embedding():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model

embedding_model= get_embedding()
    
DB_FAISS_PATH='vectorstore/db_faiss'
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)
