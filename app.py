import streamlit as st
from dotenv import load_dotenv
import os
from graph import build_graph

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    st.set_page_config(page_title="Healthy Recipe Copilot", page_icon="🥗")
    st.title("🥗 Healthy Recipe Copilot")
    st.write("Find healthy recipes based on ingredients, calories, allergies, and diet preferences.")

    # --- User Inputs ---
    query = st.text_input("What are you looking for?")
    allergies_text = st.text_input("Food allergies (comma-separated, optional)")
    diet_options = ["None", "Vegetarian", "Vegan", "Pescatarian", "Gluten-Free", "Lactose-Free"]
    diet_choice = st.selectbox("Diet Preference", diet_options)

    if st.button("Find Recipes"):
        if not query:
            st.warning("Please enter a query.")
            return

        # --- Parse allergies ---
        allergies = [a.strip().lower() for a in allergies_text.split(",") if a.strip()]

        # --- Parse diet ---
        diet = diet_choice.lower() if diet_choice != "None" else None

        # --- Initialize state ---
        state = {
            "query": query,
            "intent": "",
            "ingredients": [],
            "allergies": allergies,
            "diet": diet,
            "results": []
        }

        # --- Build and run the graph ---
        graph = build_graph()
        final_state = graph.invoke(state)
        results = final_state.get("results", [])

        # --- Display results ---
        if not results:
            st.error("No recipes found matching your filters.")
        else:
            st.subheader("🍴 Recommended Recipes")
            for r in results:
                # Recipe title + link
                if r.get("sourceUrl"):
                    st.markdown(f"### [{r['name']}]({r['sourceUrl']}) ({r.get('source', 'Unknown')})")
                else:
                    st.markdown(f"### {r['name']} ({r.get('source', 'Unknown')})")

                # Recipe image
                if r.get("image"):
                    st.image(r["image"], width=300)

                if r.get("description"):
                    st.write(r["description"])

                if r.get("ingredients"):
                    st.write("**Ingredients:**")
                    for ing in r["ingredients"]:
                        st.write(f"- {ing}")

                st.divider()


if __name__ == "__main__":
    main()
