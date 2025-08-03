import os
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

loader = PyPDFLoader("test.pdf")

pdf = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

texts = text_splitter.split_documents(pdf)

embeddings = OpenAIEmbeddings( 
    model="text-embedding-3-large" 
)

prompts= PromptTemplate(
    input_variables=["question", "context"],
    template="You are the greatest teacher like Andrew ng. Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {question} \n\n if question is not related to contex then just say I dont know. :"
)

get_input=RunnableLambda(lambda x: {"question": input("Enter your question: ")})

vectorstore = Chroma.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def get_context(question):
    question = question["question"]
    context = retriever.invoke(question)
    full_context = "\n".join([doc.page_content for doc in context])
    return {"question": question, "context": full_context}

get_context = RunnableLambda(get_context)

def parallel_chain(inputs):
    return {
        "question": inputs["question"],
        "context": inputs["context"]
    }
parralel_chain = RunnableLambda(parallel_chain)

output_parser = StrOutputParser()

inputs_for_prompt = get_input | get_context | parralel_chain | prompts | model | output_parser

result=inputs_for_prompt.invoke({})

print(result)

