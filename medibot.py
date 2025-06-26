import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import os
import sys
import types




 
DB_FAISS_PATH='vectorstore/db_faiss'
@st.cache_resource
def get_vector_store():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)
    return db

def set_custom_promt(custom_promt_template):
    promt=PromptTemplate(template=custom_promt_template,input_variables={'context','question '})
    return promt

def load_llm(hugging_face_repo_id,HF_TOKEN):
    llm= HuggingFaceEndpoint(
        repo_id=hugging_face_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={'token':HF_TOKEN,
                      'max_length':"512"}
    )
    return llm

def main():
    class FakeTorchClasses(types.SimpleNamespace):
        __path__ = types.SimpleNamespace(_path=[])

    sys.modules["torch.classes"] = FakeTorchClasses()
    st.title("🤖 Welcome to Medibot - Your AI Health Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("💬 Ask me anything about internal medicine...")

    if prompt:
        st.chat_message('user').markdown(f"🧑‍💻 **You:** {prompt}")
        st.session_state.messages.append({'role': 'user', 'content': prompt})

    custom_promt_template = '''
    Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, just say you don't know in a polite way. 
    Don't try to make up answers or provide anything outside the given context.

    context: {context}
    question: {question}

    Start the answer directly. No small talk, please.
    '''

    hugging_face_repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    HF_TOKEN = os.environ.get('HF_TOKEN')

    try:
        vectorstore = get_vector_store()
        if vectorstore is None:
            st.error("❌ Failed to load database.")
            return

        llm = load_llm(hugging_face_repo_id=hugging_face_repo_id, HF_TOKEN=HF_TOKEN)

        qa_chain = RetrievalQA.from_chain_type( 
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_promt(custom_promt_template)}
        )

        response = qa_chain.invoke({'query': prompt})

        result = response.get('result', "❌ Sorry, no result found.")
        result_to_show = f"💡 **Medibot:** {result.strip()}"

        st.chat_message('assistant').markdown(result_to_show)
        st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

    except Exception as e:
        st.error(f"⚠️ Oops! Something went wrong: `{str(e)}`")


if __name__ =='__main__':
    main()
    