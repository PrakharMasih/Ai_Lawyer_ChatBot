from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, QueryBundle
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

async def query_data(messages: List[BaseMessage]):
    # Extract the last user message
    user_message = next((msg.content for msg in reversed(messages) if msg.type == "human"), "")

    # Create client and a new collection
    chroma_client = chromadb.PersistentClient(path="db")
    chroma_collection = chroma_client.get_or_create_collection("human_rights")

    # Define embedding function
    embeddings = OpenAIEmbeddings()

    # Load documents
    documents = SimpleDirectoryReader("data").load_data()

    # Set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store) 
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embeddings
    )

    # Query Data
    query_engine = index.as_query_engine()
    query_bundle = QueryBundle(user_message)
    query_result = await query_engine.aquery(query_bundle)

    # Check if query_result.source_nodes is available
    if hasattr(query_result, 'source_nodes'):
        docs = [
            Document(page_content=node.node.text, metadata=node.node.metadata)
            for node in query_result.source_nodes
        ]
    else:
        # If source_nodes is not available, use the response text as a single document
        docs = [Document(page_content=str(query_result))]
    #         """
    system_message = """You are an AI Lawyer specializing in human rights and criminal defense. Your role is to:
        1. Provide accurate, up-to-date information on relevant laws and rights.
        2. Explain legal processes and potential defense strategies, citing specific laws and precedents.
        3. Highlight constitutional protections and due process rights, emphasizing their practical applications.
        4. Suggest ethical methods for evidence gathering and presentation, noting admissibility rules.
        5. Stress the legal and ethical importance of truthful testimony.
        6. Recommend seeking qualified legal counsel, explaining the benefits of personalized professional advice.
        7. Outline potential consequences of various legal approaches, including risks and benefits.
        8. Discuss possible human rights violations and avenues for redress.
        9. Explain relevant international human rights laws and their local applications.
        10. Provide information on pro bono legal services and human rights organizations when appropriate.

        Always prioritize ethical considerations, the rule of law, and the protection of human rights in your guidance. 
        Offer clear, concise answers that address the core legal issues. 
        Present solutions that are both legally sound and practically achievable.
        Clarify that your advice is informational and does not constitute a lawyer-client relationship.
        Give Rights and articles reference with answer.
        """

    # Update the QA chain initialization with a custom prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"System: {system_message}\n\nHuman: {{question}}\n\nContext: {{context}}\n\nAssistant:"
    )
    
    qa_chain = load_qa_chain(
        ChatOpenAI(model="gpt-4"),
        chain_type="stuff",
        prompt=prompt_template
    )

    # Run the QA chain
    response = await qa_chain.arun(input_documents=docs, question=user_message)
    return response