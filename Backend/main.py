from UI.slideshow_manager import run_slideshow
from Data_Tracking.summarize_results import summarize_results_with_openai 

def main():
    image_folder = "/Users/isaacgutterman/CNN-Teach/images" 
    slide_paths = run_slideshow(image_folder)
    emotion_log_file = "/Users/isaacgutterman/CNN-Teach/Backend/Data_Tracking/emotion_log.csv"
    # Once the slideshow finishes, generate the summary using OpenAI API
    summarize_results_with_openai(emotion_log_file, slide_paths)

if __name__ == "__main__":
    main()
