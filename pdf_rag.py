import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PERSIST_DIR = "faiss_index"
PDF_PATHS = ["PDF/healthy-cookbook.pdf"]
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)


def ensure_vectorstore(persist_dir: str = PERSIST_DIR):
    if os.path.exists(persist_dir):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

    print("⚡ FAISS index not found. Building from PDF...")
    docs = []

    def is_recipe_page(page_text: str):
        text = page_text.strip().lower()
        # Accept pages starting with "serves" or "ingredients:"
        return text.startswith("serves") or text.startswith("ingredients:")

    for pdf in PDF_PATHS:
        if not os.path.exists(pdf):
            raise FileNotFoundError(f"PDF not found: {pdf}")
        loader = PyPDFLoader(pdf)
        all_pages = loader.load()
        recipe_pages = [p for p in all_pages if is_recipe_page(p.page_content)]
        docs.extend(recipe_pages)

    # Use a large chunk size to avoid cutting instructions
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(persist_dir)
    return vectordb


vectorstore = ensure_vectorstore()


def query_pdf_structured(query: str) -> list:
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        docs = retriever.get_relevant_documents(query)

        if not docs:
            print("DEBUG: No PDF chunks retrieved")
            return []

        # Concatenate chunks to send to LLM
        text = "\n".join([d.page_content for d in docs if hasattr(d, "page_content")]).strip()

        prompt = PromptTemplate(
            input_variables=["recipe_text"],
            template="""
You are a recipe assistant. Extract all recipes from the text. 
Each recipe should include:
- name (title)
- ingredients (list)
- instructions (step-by-step)
- calories (if available, else null)

Return a JSON array exactly like:
[
    {{
    "name": "Recipe 1",
    "ingredients": ["ingredient1", "ingredient2"],
    "instructions": "Step 1 ... Step 2 ...",
    "calories": 123
    }},
    {{
    "name": "Recipe 2",
    "ingredients": ["ingredientA", "ingredientB"],
    "instructions": "Step 1 ... Step 2 ...",
    "calories": 456
    }}
]

ONLY include actual recipes. Exclude TOC, headings, or non-recipe text. All recipes start with "Serves" or "Ingredients:".

Recipe text:
\"\"\"{recipe_text}\"\"\"
"""
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        response = llm_chain.run(recipe_text=text)

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            print("DEBUG: JSON parse error, returning empty list")
            parsed = []

        return parsed

    except Exception as e:
        print(f"PDF extraction error: {e}")
        return []
