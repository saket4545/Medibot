from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN= os.environ.get('HF_TOKEN')
hugging_face_repo_id='mistralai/Mistral-7B-Instruct-v0.3'

def load_llm(hugging_face_repo_id):
    llm= HuggingFaceEndpoint(
        repo_id=hugging_face_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={'token':HF_TOKEN,
                      'max_length':"512"}
    )
    return llm


custom_promt_template='''
use the pieces of informaton provided in the context to answer user's question
if you dont know the answer, just say that you dont know in polite way,dont try
to make up answers. Dont provide anything out of the given context.

context:{context}
Question:{question}


Start the answer directly. No small talks please
'''
def set_custom_promt(custom_promt_template):
    promt=PromptTemplate(template=custom_promt_template,input_variables={'context','question '})
    return promt
DB_FAISS_PATH='vectorstore/db_faiss'
embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db=FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)

qa_chain=RetrievalQA.from_chain_type(
    llm= load_llm(hugging_face_repo_id=hugging_face_repo_id),
    chain_type="stuff",
    retriever = db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_promt(custom_promt_template)}
)

user_query=input("write query here : ")
response=qa_chain.invoke({'query':user_query})
print('Result: ',response['result'])
print('Source Documents: ',response['source_documents'])