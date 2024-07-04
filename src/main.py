import os
from langchain import hub
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import requests
import bs4

# Load environment variables
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY environment variable not set. Please set it for Google Generative AI.")

# Configure Google API key for generative AI
genai.configure(api_key=google_api_key)

# Define model names
Generative_model = 'gemini-1.5-pro'
embedding_model_name = "models/embedding-001"
embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model_name, task_type="retrieval_document")

# Initialize the generative AI model for conversation
llm = ChatGoogleGenerativeAI(model=Generative_model, temperature=0.7)



# Define the URL to load
urls = ["https://kenyanwallstreet.com/safaricom-and-pezesha-introduce-mkopo-wa-pochi-to-lend-small-businesses/",
        "https://kenyanwallstreet.com/nses-next-listing-expected-to-be-marula-mining/", "https://kenyanwallstreet.com/value-of-public-assets-traced-by-eacc-declines-to-ksh-6-6bn-in-2022-23/",
        "https://kenyanwallstreet.com/sgr-passenger-traffic-drops-by-11-per-cent-in-q1-2024/S"]

# Initialize WebBaseLoader with BeautifulSoup parsing options
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            ("body")
        )
    ),
)

# Load text from URLs
documents = loader.load()

# Extract text content from documents
split_text = [doc.page_content for doc in documents]

# Initialize the vector database with the embedding model
vectordb = Chroma(embedding_function=embedding_model)

# Add documents to the vector database
vectordb.add_texts(texts=split_text)

# print("Documents have been successfully added to the vector database.")

retriever = vectordb.as_retriever()

print(retriever)


# Create prompt template for RAG
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant with access to a document containing information about investing in Kenya. 
    Your task is to provide relevant information based on this document. If the document does not contain 
    the relevant information, provide a generic answer but mention this is a generic answer since the document does not have
    that information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("I need to understand what investments are withing the Kenyan context. Which areas should I focus on as an investor?")

print(response)

