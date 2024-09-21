import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_results_with_openai(emotion_log_file, slide_paths):
    with open("emotion_log.csv", 'r') as file:
        log_data = file.read()

    emotion_mapping = {
        0: "angry",
        1: "disgusted",
        2: "fearful",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise"

    }

    prompt = f"""Here is the emotion log for a slideshow presentation where emotions are recorded as numbers:
    0: Angry
    1: Disgusted
    2: Fearful
    3: Happy
    4: Neutral
    5: Sad
    6: Surprise

    The following is the log data for each slide:
    {log_data}

    The slides shown were:
    {slide_paths}

    Based on the information given generate a summary on which slides made the person feel each of the 7 emotions"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes emotion data based on logs."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    summary = response['choices'][0]['message']['content']
    print("Summary of Emotions Based on Slideshow:")
    print(summary)
