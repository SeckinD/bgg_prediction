import streamlit as st
import pandas as pd
import joblib
import lightgbm
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(layout="wide")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/BoardGameGeek_Logo.svg/1200px-BoardGameGeek_Logo.svg.png", width=400)
st.title("🎲:red[KUTU] :blue[OYUNU SEÇME] :red[ARACI]🎲")

st.header("🪀KUTU OYUNU SEÇME ARACI🪀")
tab_home, tab_ml = st.tabs(["Ana Sayfa", "Puan Tahmini(ML)"])
tab_home.markdown("Kutu oyunları için bir puan tahmin modeli oluşturduk.")

column_bgg, column_recommender = tab_home.columns([2,1], gap="small")

column_bgg.subheader("Recommendation System")
column_bgg.markdown("Öyle yaptık böyle yaptık")
column_recommender.subheader("Machine Learning")
column_recommender.markdown("Öyle oldu böyle oldu")
# sidebar
st.sidebar.image("https://cf.geekdo-images.com/-Qer2BBPG7qGGDu6KcVDIw__original/img/PlzAH7swN1nsFxOXbfUvE3TkE5w=/0x0/filters:format(png)/pic2452831.png")
st.sidebar.image("https://cf.geekdo-images.com/T1ltXwapFUtghS9A7_tf4g__original/img/xIAzJY7rl-mtPStRZSqnTVsAr8Y=/0x0/filters:format(jpeg)/pic1401448.jpg")
st.sidebar.image("https://cf.geekdo-images.com/EPdI2KbLVtpGWLgL_eJLFg__original/img/ahppJwWWpWQTzT8LR-2ObsjB7OY=/0x0/filters:format(jpeg)/pic5885690.jpg")


# modele beslemek için gerekli seçimleri yap
tab_ml.markdown("<h1 style='text-align: center; '>Oyununuz için değerleri seçiniz</h1>", unsafe_allow_html=True)

# sütunlara ayır
column_one, column_two, column_three = tab_ml.columns([3, 3, 3], gap="small")


def load_model():
    return joblib.load("bgmodel_.joblib")
model = load_model()

def user_input_features():
    subcategories = ["Savas", "Bilgi", "Macera", "Strateji", "Puzzle", "Sosyal", "Sanat", "Rekabetçi"]
    selected_subcategories = column_one.multiselect("Select subcategories:", subcategories)
    subcategories_df = pd.DataFrame(columns=subcategories)
    subcategories_df.loc[0] = [0] * len(subcategories)
    for subcategory in selected_subcategories:
        if subcategory in subcategories_df.columns:
            subcategories_df.at[0, subcategory] = 1
    return subcategories_df

def categorize_game(min_age, average_weight, max_players, max_playtime):
    if (min_age <= 7 and average_weight < 3.5):
        return "Children's_Game"
    elif max_players >= 6 and average_weight <= 2:
        return "Party_Game"
    elif average_weight <= 2 and min_age <= 13 and max_playtime <= 90:
        return "Family_Game"
    elif average_weight > 3.5:
        return "Heavy_Game"
    elif average_weight > 2:
        return "Strategy_Game"
    else:
        return "Family_Game"

min_age = column_one.number_input('Minimum Yaş', min_value=0, value=0, step=1)
min_players = column_two.number_input('Minimum Oyuncu Sayısı', min_value=0, value=0, step=1)
max_players = column_three.number_input('Maksimum Oyuncu Sayısı', min_value=0, value=0, step=1)
max_playtime = column_three.number_input('Oyun Süresi (dakika)', min_value=0, value=0, step=1)
average_weight = column_two.slider('Karmaşıklık', 0.0, 5.0, 0.0)


game_category = categorize_game(min_age, average_weight, max_players, max_playtime)
categories = ["Children's_Game", "Party_Game", "Family_Game", "Heavy_Game", "Strategy_Game"]
category_df = pd.DataFrame(columns=categories)
category_df.loc[0] = [0] * len(categories)
category_df.at[0, game_category] = 1
category_df = category_df.drop("Children's_Game", axis=1)

def categorize_playtime(max_playtime):
    if 0 <= max_playtime <= 30:
        return "Kisa"
    elif 30 < max_playtime <= 60:
        return "Orta"
    elif 60 < max_playtime <= 120:
        return "Uzun"
    else:
        return "Cok uzun"

def categorize_age(min_age):
    if min_age < 7:
        return "0-6"
    elif min_age < 10:
        return "7-10"
    elif min_age < 13:
        return "10-13"
    elif min_age < 18:
        return "13-18"
    else:
        return "+18"

# Additional inputs and one-hot encoding
playtime_category = categorize_playtime(max_playtime)
age_category = categorize_age(min_age)

playtime_categories = ["Kisa", "Orta", "Uzun", "Cok uzun"]
age_categories = ["0-6", "7-10", "10-13", "13-18", "+18"]

playtime_df = pd.DataFrame(columns=playtime_categories)
age_df = pd.DataFrame(columns=age_categories)

playtime_df.loc[0] = [0] * len(playtime_categories)
age_df.loc[0] = [0] * len(age_categories)

playtime_df.at[0, playtime_category] = 1
playtime_df = playtime_df.drop("Cok uzun", axis=1)
age_df.at[0, age_category] = 1
age_df = age_df.drop("+18", axis=1)

# Combine all DataFrames
subcategories_df = user_input_features()
combined_df = pd.concat([subcategories_df, category_df, playtime_df, age_df], axis=1)

tab_ml = tab_ml.container()
tab_ml.write("# Puan Tahmini")
input_df = combined_df


if tab_ml.button('Tahmin Et'):
    prediction = model.predict(input_df)
    tab_ml.write("Oyunun tahmin edilen puanı:")
    tab_ml.write(prediction[0])
