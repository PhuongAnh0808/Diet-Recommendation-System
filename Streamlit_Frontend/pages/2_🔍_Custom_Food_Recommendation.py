import streamlit as st
from Generate_Recommendations import Generator
from ImageFinder.ImageFinder import get_images_links as find_image
import pandas as pd
from streamlit_echarts import st_echarts

st.set_page_config(page_title="Custom Food Recommendation", page_icon="🔍", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/style.css")

nutrition_values = [
    'Calories','FatContent','SaturatedFatContent','CholesterolContent','SodiumContent',
    'CarbohydrateContent','FiberContent','SugarContent','ProteinContent'
]

if 'custom_generated' not in st.session_state:
    st.session_state.custom_generated = False
    st.session_state.custom_recommendations = None

_recs = st.session_state.get('custom_recommendations')
if isinstance(_recs, list) and _recs and not isinstance(_recs[0], dict):
    st.session_state.custom_generated = False
    st.session_state.custom_recommendations = None

class Recommendation:
    def __init__(self, nutrition_list, nb_recommendations, ingredient_txt):
        self.nutrition_list = nutrition_list
        self.nb_recommendations = nb_recommendations
        self.ingredient_txt = ingredient_txt

    def generate(self):
        raw = (self.ingredient_txt or "").strip()
        raw = raw.replace("；", ";")  # full-width ; -> ASCII ;
        ingredients = [i.strip() for i in raw.split(";") if i.strip()]

        params = {'n_neighbors': self.nb_recommendations, 'return_distance': False}
        generator = Generator(self.nutrition_list, ingredients, params)

        response = generator.generate()
        if not response.ok:
            st.error("Backend trả lỗi. Kiểm tra FastAPI đang chạy http://localhost:8000", icon="⚠️")
            return None
        try:
            data = response.json()
        except Exception:
            st.error("Backend không trả JSON hợp lệ.", icon="⚠️")
            return None

        recommendations = data.get('output') or []
        if not recommendations:
            return []

        for recipe in recommendations:
            recipe['image_link'] = find_image(recipe.get('Name', ''))
        return recommendations


class Display:
    def __init__(self):
        self.nutrition_values = nutrition_values

    def display_recommendation(self, recommendations):
        st.subheader('Recommended recipes:')
        if not recommendations:
            st.info("Couldn't find any recipes with the specified ingredients.", icon="🙁")
            return

        n = len(recommendations)
        cols = st.columns(5)
        rows = (n + 4) // 5
        for c_idx, col in enumerate(cols):
            with col:
                start = c_idx * rows
                end = min(start + rows, n)
                for recipe in recommendations[start:end]:
                    recipe_name = recipe.get('Name', 'Unnamed')
                    expander = st.expander(recipe_name)

                    recipe_link = recipe.get('image_link', '')
                    if recipe_link:
                        recipe_img = f'<div><center><img src="{recipe_link}" alt="{recipe_name}"></center></div>'
                        expander.markdown(recipe_img, unsafe_allow_html=True)

                    nutritions_df = pd.DataFrame({v: [recipe.get(v, 0)] for v in nutrition_values})
                    expander.markdown('<h5 style="text-align: center;font-family:sans-serif;">Nutritional Values (g):</h5>',
                                      unsafe_allow_html=True)
                    expander.dataframe(nutritions_df)

                    # Ingredients
                    expander.markdown('<h5 style="text-align: center;font-family:sans-serif;">Ingredients:</h5>',
                                      unsafe_allow_html=True)
                    for ingredient in recipe.get('RecipeIngredientParts', []):
                        expander.markdown(f"- {ingredient}")

                    # Instructions
                    expander.markdown('<h5 style="text-align: center;font-family:sans-serif;">Recipe Instructions:</h5>',
                                      unsafe_allow_html=True)
                    for instruction in recipe.get('RecipeInstructions', []):
                        expander.markdown(f"- {instruction}")

                    # Thời gian nấu
                    expander.markdown('<h5 style="text-align: center;font-family:sans-serif;">Cooking and Preparation Time:</h5>',
                                      unsafe_allow_html=True)
                    expander.markdown(
                        f"""
                        - Cook Time       : {recipe.get('CookTime', '')}min  
                        - Preparation Time: {recipe.get('PrepTime', '')}min  
                        - Total Time      : {recipe.get('TotalTime', '')}min
                        """
                    )

    def display_overview(self, recommendations):
        if not recommendations:
            return
        if not (isinstance(recommendations, list) and isinstance(recommendations[0], dict)):
            st.info("Please generate custom recommendations first.", icon="ℹ️")
            return

        st.subheader('Overview:')
        col1, col2, col3 = st.columns(3)
        names = [r.get('Name', 'Unnamed') for r in recommendations]
        with col2:
            selected_recipe_name = st.selectbox('Select a recipe', names)

        selected_recipe = next((r for r in recommendations if r.get('Name') == selected_recipe_name), recommendations[0])

        st.markdown('<h5 style="text-align: center;font-family:sans-serif;">Nutritional Values:</h5>',
                    unsafe_allow_html=True)

        options = {
            "title": {"text": "Nutrition values", "subtext": f"{selected_recipe_name}", "left": "center"},
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left"},
            "series": [
                {
                    "name": "Nutrition values",
                    "type": "pie",
                    "radius": "50%",
                    "data": [{"value": float(selected_recipe.get(nv, 0)), "name": nv} for nv in self.nutrition_values],
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                        }
                    },
                }
            ],
        }
        st_echarts(options=options, height="600px")
        st.caption('You can select/deselect an item (nutrition value) from the legend.')

st.markdown("<h1 style='text-align: center;'>Custom Food Recommendation</h1>", unsafe_allow_html=True)

display = Display()

with st.form("recommendation_form"):
    st.header('Nutritional values:')
    Calories = st.slider('Calories', 0, 2000, 500)
    FatContent = st.slider('FatContent', 0, 100, 50)
    SaturatedFatContent = st.slider('SaturatedFatContent', 0, 13, 0)
    CholesterolContent = st.slider('CholesterolContent', 0, 300, 0)
    SodiumContent = st.slider('SodiumContent', 0, 2300, 400)
    CarbohydrateContent = st.slider('CarbohydrateContent', 0, 325, 100)
    FiberContent = st.slider('FiberContent', 0, 50, 10)
    SugarContent = st.slider('SugarContent', 0, 40, 10)
    ProteinContent = st.slider('ProteinContent', 0, 40, 10)

    nutritions_values_list = [
        Calories, FatContent, SaturatedFatContent, CholesterolContent, SodiumContent,
        CarbohydrateContent, FiberContent, SugarContent, ProteinContent
    ]

    st.header('Recommendation options (OPTIONAL):')
    nb_recommendations = st.slider('Number of recommendations', 5, 20, step=5)
    ingredient_txt = st.text_input(
        'Specify ingredients to include in the recommendations separated by ";" :',
        placeholder='Ingredient1;Ingredient2;...'
    )
    st.caption('Example: Milk;eggs;butter;chicken...')

    generated = st.form_submit_button("Generate")

if generated:
    with st.spinner('Generating recommendations...'):
        recommendation = Recommendation(nutritions_values_list, nb_recommendations, ingredient_txt)
        recs = recommendation.generate()
        st.session_state.custom_recommendations = recs
    st.session_state.custom_generated = True

if st.session_state.custom_generated:
    recs = st.session_state.custom_recommendations
    with st.container():
        display.display_recommendation(recs)
    with st.container():
        display.display_overview(recs)