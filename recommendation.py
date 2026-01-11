# recommendation.py
import pandas as pd
import json
import random

def recommend_meal():
    try:
        # Read detected emotion from JSON
        with open("current_emotion.json", "r") as f:
            data = json.load(f)
            emotion = data.get("emotion", "").strip().lower()
    except FileNotFoundError:
        print("[ERROR] current_emotion.json not found.")
        return None, []

    # Load dataset
    try:
        df = pd.read_csv("Emotion_Full_Meal_Menu.csv")
    except:
        print("[ERROR] Could not load CSV file.")
        return emotion, []

    # Normalize emotion column
    df['Emotion'] = df['Emotion'].astype(str).str.strip().str.lower()

    # Filter rows matching detected emotion
    filtered = df[df['Emotion'] == emotion]

    if filtered.empty:
        print(f"[WARN] No meals found for emotion: {emotion}")
        return emotion, []

    # Select a random row for variation
    selected = filtered.sample(1).iloc[0]

    # Build menu structure
    menu = [
        {"Category": "Appetizer", "Item": selected["Appetizer"]},
        {"Category": "Main Course", "Item": selected["Main_Course"]},
        {"Category": "Dessert", "Item": selected["Dessert"]}
    ]

    return emotion, menu
