# Healthy Recipe Agent

## Overview
The Healthy Recipe Agent is an AI-powered assistant that helps users discover recipes from a PDF cookbook. It supports filtering by diet, allergies, and calories, and provides detailed structured recipe information.

The project uses:
- Python 3.13
- LangChain + OpenAI GPT-4
- FAISS vectorstore for efficient PDF retrieval
- Streamlit for the interactive UI

---

## Features
- Query a PDF cookbook and retrieve structured recipe information.
- Filters recipes based on:
  - Dietary preferences (vegetarian, vegan, etc.)
  - Allergies (gluten, lactose, nuts, etc.)
  - Calories
- Returns JSON output including:
  - Recipe name
  - Ingredients
  - Instructions
  - Calories (if available)
- Debug mode prints full recipe text for inspection.

---

## Project Structure
```
CAPSTONE/
│
├─ app.py                # Streamlit front-end
├─ pdf_rag.py            # PDF RAG pipeline
├─ agent_tools.py        # Filtering, Spoonacular API integration
├─ graph.py              # LangGraph agent setup
├─ PDF/                  # PDF cookbook(s)
├─ faiss_index/          # Persisted FAISS index
├─ README.md             # This file
└─ pyproject.toml        # Poetry config
```

---

## LangGraph Agent Decision Flow
```text
[User Query]
       |
       v
[PDF RAG Retriever] -- retrieves relevant chunks from FAISS index
       |
       v
[LLM Parser] -- extracts structured recipes (name, ingredients, instructions, calories)
       |
       +----------------------------+
       |                            |
       v                            v
[Filter Diet]                  [Filter Allergies]
       |                            |
       +-------------+--------------+
                     v
              [Filter Calories]
                     |
                     v
              [JSON Response]
                     |
                     v
            [Streamlit UI Display]
```

- Each step can be debugged independently.
- The PDF retriever ensures only recipe pages are processed.
- Filters are optional and can be combined.

---

## Setup Instructions

1. Clone the repository:
```bash
git clone <repo_url>
cd CAPSTONE
```

2. Install dependencies with Poetry:
```bash
poetry install
```

3. Create a `.env` file and set your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
RAPIDAPI_KEY=your_rapidapi_key_here

How to get a RapidAPI key
Go to RapidAPI.
Sign up for a free account.
Search for Spoonacular in the marketplace.
Subscribe to the API (the free tier is enough for testing).
Go to your Dashboard → Security → API Keys, and copy your key.
Paste it into your .env file as RAPIDAPI_KEY.
```

4. Place your cookbook PDF(s) in the `PDF/` folder.

5. Run the Streamlit app:
```bash
poetry run streamlit run app.py
```

---

## Notes

- FAISS index is automatically built on first run.
- Debug prints show full recipe text if enabled.


