import openai
import os
from dotenv import load_dotenv
import time
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_results_with_openai(emotion_log_file, slide_paths):
    # Check if the emotion log file exists before attempting to summarize
    time.sleep(50)
    if not os.path.exists(emotion_log_file):
        raise FileNotFoundError(f"{emotion_log_file} not found.")

    with open(emotion_log_file, 'r') as file:
        log_data = file.read()

    # Emotion mapping
    emotion_mapping = {
        0: "angry",
        1: "disgusted",
        2: "fearful",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise"
    }

    # Slide descriptions
    slide_descriptions = [
        "image1.jpg = A beautiful stack of pancakes",
        "image2.jpg = A sunny beach with blue water",
        "image3.jpg = Two people climbing a mountain on skis in a massive snowy mountain range",
        "image4.jpg = Math textbooks insides",
        "image5.jpg = A game of FIFA",
        "image6.jpg = A knight with armor and a sword",
        "image7.jpg = A person on their phone in class",
        "image8.jpg = A jagged mountain range sunrise",
        "image9.jpg = A rainy street",
        "image10.jpg = A campfire at night"
    ]

    # Create a formatted string for the descriptions
    slides_with_descriptions = "\n".join(slide_descriptions)

    # Prompt for OpenAI
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

    which correspond to these descriptions:
    {slides_with_descriptions}

    Based on the information given, generate an overall summary of the persons emotions in relation to different concepts."""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes emotion data based on logs and slide descriptions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    summary = response['choices'][0]['message']['content']
    print("Summary of Emotions Based on Slideshow:")
    print(summary)
