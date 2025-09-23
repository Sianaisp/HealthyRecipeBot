from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel
from langchain_openai import ChatOpenAI 
from langgraph.graph import StateGraph, START, END
from agent_tools import search_recipes_spoonacular, filter_allergies, filter_diet
from pdf_rag import query_pdf_structured

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPENAI_API_KEY)

# -------------------------
# State model
# -------------------------
class RecipeState(BaseModel):
    query: str
    intent: str = ""
    meal_type: str | None = None
    diet: str | None = None
    ingredients: list[str] = []
    allergies: list[str] = []
    results: list[dict] = []


# -------------------------
# Nodes
# -------------------------
def classify_intent(state: RecipeState) -> RecipeState:
    """
    Determine intent, meal type, and diet from user query.
    """
    prompt = f"""
You are a smart recipe assistant.
User query: "{state.query}"

1️⃣ Determine intent:
    - "ingredients" if user specifies ingredients
    - "profile" if user asks generally (like "I want a healthy lunch")

2️⃣ Detect meal type (breakfast, lunch, dinner, snack), or leave empty if not clear.

3️⃣ Detect diet if mentioned (vegetarian, vegan, pescetarian, gluten-free), or leave empty.

Respond in JSON like:
{{
    "intent": "profile",
    "meal_type": "lunch",
    "diet": "vegetarian"
}}
"""
    try:
        response = llm.predict(prompt).strip()
        parsed = json.loads(response)
        intent = parsed.get("intent", "profile").lower()
        meal_type = parsed.get("meal_type")
        diet = parsed.get("diet")
        if intent not in ["ingredients", "profile"]:
            intent = "profile"
    except Exception as e:
        print(f"⚠️ classify_intent failed: {e}")
        intent = "profile"
        meal_type = None
        diet = None

    state.intent = intent
    state.meal_type = meal_type

    # <-- Only overwrite diet if not already set (Streamlit input takes priority)
    if not state.diet:
        state.diet = diet

    print(f"🧭 Classified intent='{state.intent}', meal_type='{state.meal_type}', diet='{state.diet}'")
    return state


def ingredients_flow(state: RecipeState) -> RecipeState:
    """
    Search recipes by ingredients via Spoonacular API,
    apply diet/allergy filters, and include structured PDF recipes.
    """
    print(f"🥕 Entering ingredients_flow with query='{state.query}'")

    # Spoonacular search
    results = search_recipes_spoonacular(
        ingredients=[state.query],
        meal_type=state.meal_type,
        diet=None,  # We filter manually via LLM
        number=5
    )

    print(f"🔍 Found {len(results)} Spoonacular results before filtering. Diet={state.diet}, Allergies={state.allergies}")

    # Filter by allergies 
    if state.allergies:
        results = filter_allergies(results, state.allergies)
        print(f"🚫 After allergy filter: {len(results)} recipes remain")

    # Filter by diet 
    if state.diet:
        print(f"🥗 Applying diet filter: {state.diet}")
        results = filter_diet(results, state.diet)
        print(f"✅ After diet filter: {len(results)} recipes remain")

    # Tag Spoonacular results 
    for r in results:
        r["source"] = "Spoonacular"
        r["sourceUrl"] = r.get("sourceUrl")
        r["image"] = r.get("image", None)

    # Add PDF results 
    parsed = query_pdf_structured(state.query)
    if parsed:
        print(f"📖 Adding {len(parsed)} PDF recipes")
        pdf_recipes = parsed if isinstance(parsed, list) else [parsed]
        for r in pdf_recipes:
            # Filter PDF by diet if needed
            if state.diet:
                allowed = filter_diet([r], state.diet)
                if not allowed:
                    continue
            r["source"] = "PDF"
            r["sourceUrl"] = None
            r["image"] = None
            results.append(r)

    state.results = results
    print(f"🏁 ingredients_flow finished with {len(state.results)} recipes")
    return state


def profile_flow(state: RecipeState) -> RecipeState:
    """
    Search recipes based on general profile queries,
    apply diet/allergy filters, and include structured PDF recipes.
    """
    print(f"🍽️ Entering profile_flow with query='{state.query}'")

    # Spoonacular search 
    results = search_recipes_spoonacular(
        ingredients=[],
        meal_type=state.meal_type,
        diet=None,  # We filter manually via LLM
        number=5
    )

    print(f"🔍 Found {len(results)} Spoonacular results before filtering. Diet={state.diet}, Allergies={state.allergies}")

    # Filter by allergies 
    if state.allergies:
        results = filter_allergies(results, state.allergies)
        print(f"🚫 After allergy filter: {len(results)} recipes remain")

    # Filter by diet
    if state.diet:
        print(f"🥗 Applying diet filter: {state.diet}")
        results = filter_diet(results, state.diet)
        print(f"✅ After diet filter: {len(results)} recipes remain")

    # Tag Spoonacular results
    for r in results:
        r["source"] = "Spoonacular"
        r["sourceUrl"] = r.get("sourceUrl")
        r["image"] = r.get("image", None)

    # Add PDF results 
    parsed = query_pdf_structured(state.query)
    if parsed:
        print(f"📖 Adding {len(parsed)} PDF recipes")
        pdf_recipes = parsed if isinstance(parsed, list) else [parsed]
        for r in pdf_recipes:
            # Filter PDF by diet if needed
            if state.diet:
                allowed = filter_diet([r], state.diet)
                if not allowed:
                    continue
            r["source"] = "PDF"
            r["sourceUrl"] = None
            r["image"] = None
            results.append(r)

    state.results = results
    print(f"🏁 profile_flow finished with {len(state.results)} recipes")
    return state


# -------------------------
# Routing
# -------------------------
def route_by_intent(state: RecipeState):
    """
    Route to the correct flow based on intent.
    PDF recipes are included automatically inside each flow.
    """
    next_node = "ingredients_flow" if state.intent == "ingredients" else "profile_flow"
    print(f"🚦 Routing intent='{state.intent}' → {next_node}")
    return next_node


# -------------------------
# Build graph
# -------------------------
def build_graph():
    graph = StateGraph(RecipeState)

    # Add nodes
    graph.add_node(classify_intent)
    graph.add_node(profile_flow)
    graph.add_node(ingredients_flow)

    # Start edge
    graph.add_edge(START, "classify_intent")

    # Conditional routing
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "profile_flow": "profile_flow",
            "ingredients_flow": "ingredients_flow",
        },
    )

    # End edges
    graph.add_edge("profile_flow", END)
    graph.add_edge("ingredients_flow", END)

    print("⚡ Graph built and compiled.")
    return graph.compile()
