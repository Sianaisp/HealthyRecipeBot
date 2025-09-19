import os
import time
import requests
import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI  # updated import

# Load environment variables
load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# --- Caches ---
RECIPE_CACHE = {}       # Spoonacular full recipe info cache
ALLERGEN_CACHE = {}     # LLM allergen cache


# --- LLM-based helpers ---
def is_allergen(ingredient: str, allergy: str) -> bool:
    """Check if an ingredient contains the specified allergen using LLM with caching."""
    key = (ingredient.lower(), allergy.lower())
    if key in ALLERGEN_CACHE:
        return ALLERGEN_CACHE[key]

    prompt = f'Is the ingredient "{ingredient}" considered "{allergy}"? Answer only "yes" or "no".'
    try:
        response = llm.predict(prompt).strip().lower()
        result = response == "yes"
    except Exception:
        result = False

    ALLERGEN_CACHE[key] = result
    return result


def filter_allergies(recipes: list, allergies: list) -> list:
    """Filter out recipes containing allergens."""
    safe_recipes = []
    for recipe in recipes:
        if any(is_allergen(ing, allergy) for ing in recipe.get("ingredients", []) for allergy in allergies):
            continue
        safe_recipes.append(recipe)
    return safe_recipes


def filter_diet(recipes: list, diet: str) -> list:
    """Use LLM to filter recipes by diet."""
    if not recipes:
        return []

    recipe_names = [r["name"] for r in recipes]
    prompt = f"""
    You are a diet filter. User diet: {diet}.
    From this list of recipes, return ONLY the names that match the diet:
    {recipe_names}

    Return as a JSON list of strings.
    """
    try:
        response = llm.predict(prompt)
        allowed = json.loads(response)
        return [r for r in recipes if r["name"] in allowed]
    except Exception:
        # Fallback for vegetarian
        if diet == "vegetarian":
            return [r for r in recipes if not any(
                meat in " ".join(r["ingredients"]).lower()
                for meat in ["chicken", "beef", "pork", "fish", "lamb", "turkey"]
            )]
        return recipes


# --- Spoonacular API helpers ---
def get_recipe_info(recipe_id: int, retries=5) -> dict:
    """Fetch full recipe info from Spoonacular with caching."""
    if recipe_id in RECIPE_CACHE:
        return RECIPE_CACHE[recipe_id]

    url = f"https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/{recipe_id}/information"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 429:  # rate limit
                sleep_time = 2 ** attempt
                print(f"429 rate limit, sleeping {sleep_time}s")
                time.sleep(sleep_time)
                continue
            response.raise_for_status()
            data = response.json()
            recipe = {
                "id": recipe_id,
                "name": data.get("title"),
                "ingredients": [i['name'] for i in data.get('extendedIngredients', [])],
                "calories": None,  # optional: calculate from nutrition info if needed
                "sourceUrl": data.get("sourceUrl")
            }
            RECIPE_CACHE[recipe_id] = recipe
            return recipe
        except requests.RequestException as e:
            sleep_time = 2 ** attempt
            print(f"Fetch failed (attempt {attempt+1}/{retries}): {e}, sleeping {sleep_time}s")
            time.sleep(sleep_time)
            continue

    return {"id": recipe_id, "name": "Unknown", "ingredients": [], "calories": None, "sourceUrl": None}


def search_recipes_spoonacular(
    ingredients=None,
    meal_type=None,
    diet=None,
    number=5,
    retries=5
) -> list:
    """
    Search recipes via Spoonacular and fetch full info for filtering.
    """
    url = "https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/complexSearch"
    querystring = {
        "includeIngredients": ",".join(ingredients) if ingredients else None,
        "type": meal_type,
        "diet": diet,
        "number": str(number),
        "addRecipeInformation": "false"  # fetch info separately
    }
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "spoonacular-recipe-food-nutrition-v1.p.rapidapi.com"
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=querystring)
            print(f"DEBUG: status {response.status_code}, attempt {attempt+1}")
            if response.status_code == 429:
                wait_time = 5 * (attempt + 1)
                print(f"⚠️ Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()
            results = [get_recipe_info(r["id"]) for r in data.get("results", [])]
            return results
        except requests.RequestException as e:
            wait_time = 5 * (attempt + 1)
            print(f"⚠️ Request failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            continue

    return []


# --- PDF helper ---
def extract_recipes_from_pdf(parsed_chunks: list) -> list:
    """
    Use LLM to extract structured recipes from PDF text chunks.
    Each recipe should have name, ingredients list, and optionally calories.
    """
    recipes = []
    for chunk in parsed_chunks:
        prompt = f"""
        Extract all recipes from the following text.
        Return JSON list of objects with 'name', 'ingredients' (list), 'calories' (if available):
        {chunk}
        """
        try:
            llm_chain = llm  # can just use llm.predict here
            response = llm_chain.predict(prompt)
            extracted = json.loads(response)
            recipes.extend(extracted)
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            continue
    return recipes
