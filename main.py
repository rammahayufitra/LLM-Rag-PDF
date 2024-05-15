from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import hub
from transformers import BitsAndBytesConfig

import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)

# Step 1: Load PDF data
loader = PyPDFLoader("/home/ramma/Downloads/a.pdf")
data = loader.load()

# Step 2: Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", ".", "\n", " ", ""])
all_splits = text_splitter.split_documents(data)

# Step 3: Initialize HuggingFaceEmbeddings
model_path = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Step 4: Create the vector store
vectorstore = FAISS.from_documents(all_splits, hf)

# Step 5: Convert vector store to retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Step 6: Initialize text generation pipeline
model_id = "pankajmathur/orca_mini_3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_threshold=200.0)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
)
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024
)

llm = HuggingFacePipeline(pipeline=pipe)

# Step 7: Load RAG prompt
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

# Step 8: Create RetrievalQA instance
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}, chain_type="stuff")

while True:
    # Ambil pertanyaan dari terminal
    question = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ")
    if question.lower() == 'exit':
        break
    
    # Start time measurement
    start_time = time.time()

    response = qa_chain.invoke(question)

    # End time measurement
    end_time = time.time()

    # Extract the relevant part of the response containing the conclusion
    conclusion_part = response["result"].split("Answer:")[1]

    # Remove any leading or trailing whitespace
    conclusion_part = conclusion_part.strip()

    # Calculate the number of tokens in the question
    question_tokens = len(tokenizer(question)["input_ids"])

    # Calculate the time taken in seconds
    elapsed_time = end_time - start_time

    # Calculate tokens per second
    tokens_per_second = question_tokens / elapsed_time

    # Print the results
    print("-" * 50)
    print("Response:", conclusion_part)
    print("Tokens per second:", tokens_per_second)
