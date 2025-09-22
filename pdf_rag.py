import os
import json
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Config
# -------------------------
PERSIST_DIR = "faiss_index"
PDF_PATHS = ["PDF/healthy-cookbook.pdf"]
llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)

# -------------------------
# Helpers
# -------------------------
def split_recipes_from_text(text: str) -> list[str]:
    """
    Split PDF text into full recipes based on 'Serves' or 'INGREDIENTS:' markers.
    Keeps the full ingredient and instruction blocks intact.
    """
    # Use a positive lookahead to keep the delimiter
    pattern = r"(?=^[A-Z][A-Z0-9 &\-']+\s*(?:Serves:|INGREDIENTS:))"
    parts = re.split(pattern, text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]

def load_and_split_pdfs(pdf_paths: list[str]) -> list[str]:
    """
    Load PDFs and split them into recipe-level chunks.
    """
    all_recipes = []
    for pdf in pdf_paths:
        if not os.path.exists(pdf):
            raise FileNotFoundError(f"PDF not found: {pdf}")
        loader = PyPDFLoader(pdf)
        pages = loader.load()
        raw_text = "\n".join([p.page_content for p in pages])
        recipes = split_recipes_from_text(raw_text)
        all_recipes.extend(recipes)
    return all_recipes

# -------------------------
# Vectorstore
# -------------------------
def ensure_vectorstore(persist_dir: str = PERSIST_DIR):
    if os.path.exists(persist_dir):
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

    print("⚡ FAISS index not found. Building from PDF...")
    recipes = load_and_split_pdfs(PDF_PATHS)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Each recipe is a "document"
    from langchain.schema import Document
    docs = [Document(page_content=recipe) for recipe in recipes]

    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(persist_dir)
    return vectordb

vectorstore = ensure_vectorstore()

# -------------------------
# Query PDF
# -------------------------
def query_pdf_structured(query: str) -> list[dict]:
    """
    Retrieve relevant PDF chunks and extract structured recipes.
    Each recipe dict includes: {name, serves, ingredients, instructions}
    """
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)

        if not docs:
            print("DEBUG: No PDF chunks retrieved")
            return []

        # Debug: show first 300 chars of each retrieved chunk
        for i, doc in enumerate(docs):
            preview = doc.page_content[:300].replace("\n", " ")
            print(f"DEBUG: Chunk {i} preview: {preview}...\n")

        text = "\n\n".join([d.page_content for d in docs]).strip()

        # LLM prompt: force line-by-line ingredients and full instructions
        recipe_prompt = PromptTemplate(
            input_variables=["recipe_text"],
            template="""
You are a helpful recipe assistant.
Extract COMPLETE recipes from the following text, without omitting any ingredients or instructions.
Return as a JSON array with this format:
[
    {{
    "name": "Recipe Name",
    "serves": "X" or null,
    "ingredients": ["ingredient 1", "ingredient 2", "ingredient 3"],
    "instructions": "step 1. step 2. step 3."
    }}
]
Instructions:
- Keep each ingredient as one item in the list (do not merge multiple ingredients into one line).
- Preserve all steps of the instructions, including multi-line notes.
Text:
\"\"\"{recipe_text}\"\"\"
"""
        )

        chain: RunnableSequence = recipe_prompt | llm
        response = chain.invoke({"recipe_text": text}).content
        print(f"DEBUG: LLM response:\n{response}\n")

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            print("DEBUG: JSON parse error, returning empty list")
            parsed = []

        return parsed

    except Exception as e:
        print(f"PDF extraction error: {e}")
        return []
