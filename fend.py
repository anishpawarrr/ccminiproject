from haystack.nodes import PreProcessor, PromptModel, PromptTemplate, PromptNode
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack import Pipeline
from haystack.nodes import BM25Retriever
from json import loads, dumps
import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from huggingface_hub import login
st.session_state['HF_TOKEN'] = "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"


st.header("Talk with PDF")
st.write('---')
st.subheader('Upload a PDF file and ask questions about it. The model will answer your questions based on the content of the PDF.')
uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

if 'is_uploaded' not in st.session_state:
    st.session_state['is_uploaded'] = False 
if 'notrerun' not in st.session_state:
    st.session_state['notrerun'] = False
if uploaded_file is not None:
    st.success("File has been uploaded successfully!")
    st.session_state['is_uploaded'] = True
    st.session_state['notrerun'] = True

if st.session_state['is_uploaded'] and st.session_state['notrerun']:
    login(token = st.session_state['HF_TOKEN'])
    uploaded_file_io = BytesIO(uploaded_file.read())
    st.session_state['notrerun'] = False

    # Create a PDF file reader object
    pdf_reader = PdfReader(uploaded_file_io)

    st.session_state['docs'] = []

    # Read each page of the PDF and append its text to the string
    for page in pdf_reader.pages:
        st.session_state['docs'].append(Document(content = page.extract_text()))

    processor = PreProcessor()
    ppdocs = processor.process(st.session_state['docs'])


    docu_store = InMemoryDocumentStore(use_bm25=True)
    docu_store.write_documents(ppdocs)

    st.session_state['retriever'] = BM25Retriever(docu_store, top_k = 3)

    qa_template = PromptTemplate(
        prompt =
        '''
        PROVIDE ANSWERS FROM CONTEXT ONLY.
        DON'T PROVIDE IRRELEVANT INFORMATION.
        IF YOU DON'T KNOW THE ANSWER, JUST REPLY THAT YOU DON'T KNOW.
        Context: {join(documents)};
        Prompt: {query}
        '''
    )
    st.session_state['prompt_node'] = PromptNode(
        model_name_or_path = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        api_key = st.session_state['HF_TOKEN'],
        default_prompt_template=qa_template,
        max_length = 29000,
        model_kwargs={"model_max_length":500000}
    )

    st.session_state['rag_pipeline'] = Pipeline()
    st.session_state['rag_pipeline'].add_node(component=st.session_state['retriever'], name = 'retriever', inputs=['Query'])
    st.session_state['rag_pipeline'].add_node(component=st.session_state['prompt_node'], name = 'prompt_node', inputs=['retriever'])


if st.session_state['is_uploaded']:
    query = st.text_input("Enter your query: ")
    if st.button("Submit"):
        response = st.session_state['rag_pipeline'].run(query = query)
        response = response['results'][0].strip()
        st.write(response)
