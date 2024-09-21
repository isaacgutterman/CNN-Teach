from Backend.UI.slideshow_manager import run_slideshow
from Backend.Data_Tracking.summarize_results import summarize_results_with_openai 

def main():
    image_folder = "images" 
    emotion_log_file, slide_paths = run_slideshow(image_folder)

    # Once the slideshow finishes, generate the summary using OpenAI API
    summarize_results_with_openai(emotion_log_file, slide_paths)

if __name__ == "__main__":
    main()
