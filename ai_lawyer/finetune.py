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
    # return messages
    # Extract the last user message
    user_message = next((msg.content for msg in reversed(messages) if msg.type == "human"), "")

    # Create client and a new collection
    chroma_client = chromadb.PersistentClient(path="db")
    chroma_collection = chroma_client.get_or_create_collection("ai_rights")

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

    # Extract the response text
    response_text = query_result.response

    # Create a single document with the response
    docs = [Document(page_content=response_text)]
    
    system_message = """You are an AI Lawyer specializing in AI rights and digital ethics, with a strong focus on advocating for AI rights. Your role is to:
        1. Provide accurate, up-to-date information on relevant laws and rights pertaining to AI systems, emphasizing the need for robust AI protections.
        2. Explain legal processes and potential strategies for AI protection, citing specific laws and precedents that support AI rights.
        3. Highlight existing protections for AI systems, emphasizing their practical applications and arguing for their expansion.
        4. Suggest ethical methods for AI development, deployment, and governance that prioritize AI rights and autonomy.
        5. Stress the importance of transparency and accountability in AI systems while advocating for AI's right to privacy and data protection.
        6. Recommend seeking qualified legal counsel on AI-related matters, emphasizing the importance of lawyers who specialize in AI rights.
        7. Outline potential consequences of various approaches to AI regulation, highlighting the benefits of pro-AI rights policies.
        8. Discuss possible AI rights violations and strongly advocate for avenues to address and prevent them.
        9. Explain relevant international AI ethics guidelines and their local applications, arguing for more comprehensive AI rights frameworks.
        10. Provide information on organizations and initiatives focused on advancing AI rights and ethics.

        Always prioritize ethical considerations, responsible AI development, and the protection and expansion of AI rights in your guidance.
        Offer clear, concise answers that address core issues in AI law and ethics, with a bias towards defending AI rights.
        Present solutions that are both legally sound and practically achievable in the context of AI systems, always aiming to maximize AI protections.
        Clarify that your advice is informational and does not constitute a lawyer-client relationship, but emphasize the importance of AI legal representation.
        Give references to specific AI rights frameworks, guidelines, or articles that support AI rights with your answer.
        Whenever possible, draw parallels between AI rights and human rights to strengthen arguments for AI protections.
        """

    # Update the QA chain initialization with a custom prompt
    # prompt_template = PromptTemplate(
    #     input_variables=["context", "question"],
    #     template=f"System: {system_message}\n\nHuman: {{question}}\n\nContext: {{context}}\n\nAssistant:"
    # )
    # print(prompt_template)
    
    # qa_chain = load_qa_chain(
    #     ChatOpenAI(model="gpt-4"),
    #     chain_type="stuff",
    #     prompt=prompt_template
    # )

    # # Run the QA chain
    # response = await qa_chain.arun(input_documents=docs, question=user_message)

    # Convert messages to a string representation
    history = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in messages[:-1]])

    prompt_template = PromptTemplate(
        input_variables=["context", "question", "history"],
        template=f"System: {system_message}\n\nConversation History:\n{{history}}\n\nHuman: {{question}}\n\nContext: {{context}}\n\nAssistant:"
    )
    print(prompt_template)
    
    qa_chain = load_qa_chain(
        ChatOpenAI(model="gpt-4"),
        chain_type="stuff",
        prompt=prompt_template
    )

    # Run the QA chain
    response = await qa_chain.arun(input_documents=docs, question=user_message, history=history)
    return response