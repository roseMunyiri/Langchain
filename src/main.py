import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai
import os

# Load environment variables
load_dotenv()

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Load document
loader = WebBaseLoader(["https://kenyanwallstreet.com/safaricom-and-pezesha-introduce-mkopo-wa-pochi-to-lend-small-businesses/"])
data = loader.load()
# print(data)

# Extract text content from the loaded data
content = "\n\n".join(doc.page_content for doc in data)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
text_chunks = text_splitter.split_text(content)
print(text_chunks)
# print(content)

# text = text_splitter.split_text(content)

# print(text[700])

# # Create embeddings for text chunks

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# vectore_store = Chroma.from_texts(text, embeddings).as_retriever()

# # Create prompt

# prompt_template = """" 
# You are an AI assistant with access to a document containing information about using Django REST 
# Framework 
# (DRF) in the context of IoT. Your task is to provide relevant information based on this document. 
# If the document does not contain the relevant information, provide a generic answer and prompt the 
# user for more specific content.
# \n\n

# context :\n {context}?\n
# question :\n {question}?\n

# Answer:

# """

# prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
# chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# question = input("Enter your question: ")
# docs = vectore_store.invoke({"query": question})

# response = chain.invoke({"input_documents":docs, "question":question},
#                  return_only_outputs=True)
# print(response)

