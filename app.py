### Import Section ###
"""
IMPORTS HERE
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chainlit as cl
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache
from operator import itemgetter
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from chainlit.types import AskFileResponse
from langchain.chains import (
    ConversationalRetrievalChain,
)
import os
import uuid

### Global Section ###
"""
GLOBAL CODE HERE
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
Loader = PyMuPDFLoader
set_llm_cache(InMemoryCache())


core_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context.
"""

rag_message_list = [
    {"role" : "system", "content" : rag_system_prompt_template},
]

rag_user_prompt_template = """\
Question:
{question}
Context:
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt_template),
    ("human", rag_user_prompt_template)
])
chat_model = ChatOpenAI(model="gpt-4o-mini")

def process_file(file: AskFileResponse):
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tempfile:
        with open(tempfile.name, "wb") as f:
            f.write(file.content)
    loader = Loader(tempfile.name)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs


### On Chat Start (Session Start) Section ###
@cl.on_chat_start
async def on_chat_start():
    """ SESSION SPECIFIC CODE HERE """
    #file_path = "https://arxiv.org/pdf/2106.09685"
    #loader = Loader(file_path)
    #documents = loader.load()
    #docs = text_splitter.split_documents(documents)
    #for i, doc in enumerate(docs):
        #doc.metadata["source"] = f"source_{i}"

    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # load the file
    docs = process_file(file)
    
    # Create a unique cache for each user
    user_id = str(uuid.uuid4())  # Unique ID per user
    cache_path = f"./cache/user_{user_id}/"
    os.makedirs(cache_path, exist_ok=True)

    store = LocalFileStore(cache_path)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings, store, namespace=f"user_{user_id}"
    )

    # Typical QDrant Vector Store Set-up
    collection_name = f"pdf_to_parse_{user_id}"
    client = QdrantClient(":memory:")
    client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=cached_embedder)
    vectorstore.add_documents(docs)
    rv = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    # Let the user know that the system is ready
    # msg = cl.Message(
    #     content=f"Welcome to the AI Legal Chatbot! Ask me anything about the AI policy", disable_human_feedback=True, author="Chat AI"
    #     )
    # await msg.send()

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Qdrant vector store
    retrieval_augmented_qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=rv,
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    # retrieval_augmented_qa_chain = (
    #     {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    #     | RunnablePassthrough.assign(context=itemgetter("context"))
    #     | chat_prompt | chat_model
    # )
    cl.user_session.set("chain", retrieval_augmented_qa_chain)

### Rename Chains ###
@cl.author_rename
def rename(orig_author: str):
    """ RENAME CODE HERE """
    user_id = cl.user_session.get("user_id")  # Retrieve the user_id from the session
    if not user_id:
        # In case the user_id is not stored yet, generate one
        user_id = str(uuid.uuid4())
        cl.user_session.set("user_id", user_id)
        
    # Append or modify the original author name with the user-specific ID
    new_author_name = f"{orig_author}_user_{user_id}"
    return new_author_name

### On Message Section ###
@cl.on_message
async def main(message: cl.Message):
    """
    MESSAGE CODE HERE
    """
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    #res = await chain.acall(message.content, callbacks=[cb])
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    
    # Send the response to the user
    await cl.Message(content=answer, elements=text_elements, author="bot_for").send()
