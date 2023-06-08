# Required Libraries
import openai
import os
import json
import time
import ast
import argparse
import warnings
from tqdm import tqdm
from multiprocessing.pool import Pool

# Suppressing all warnings
warnings.filterwarnings('ignore')


def parse_args():
    """
    Command-line argument parser.
    """
    parser = argparse.ArgumentParser(description="Descriptive question-answer-generation-using-GPT-3")
    parser.add_argument("--gt_caption_file", required=True, help="Path to the ground truth captions file.")
    parser.add_argument("--output_dir", required=True, help="Path to save the annotation JSON files.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    parser.add_argument("--num_tasks", required=False, type=int, help="Number of splits.", default=10)

    return parser.parse_args()


def annotate(gt_file, caption_files, output_dir):
    """
    Generates question and answer pairs based on video captions using OpenAI GPT-3.
    """
    for file in tqdm(caption_files):
        key = file[:-4] # Strip file extension.
        caption = gt_file[key]

        # Generate completion with OpenAI GPT-3
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content":
                        "You will play two roles: a human asking questions related to describing a video and "
                        "an intelligent chatbot designed for video description and dense captioning. "
                        "Your task is to generate a detailed and descriptive paragraph based on the provided fragmented information about a video. "
                        "------"
                        "##TASK:"
                        "Users will provide fragmented descriptions of a video, and you will generate ONE conversation-like question and answer related to describing the video in detail. "
                        "The question should ask to describe the video content in detail. "
                        "The answer should be a paraphrased and well-structured paragraph based on the provided description, with a minimum of 150 words and a maximum of 300 words. "
                        "When the provided information is short, aim for a 150-word description, and when the provided information is more detailed, aim for very long descriptions upto 300-word description. "
                        "------"
                        "##INSTRUCTIONS:"
                        "- The question must be like a human conversation and focused on describing the video in detail. "
                        "- The answer must be a paraphrased version of the provided information, very detailed and descriptive, and within the specified word count. "
                },
                {
                    "role": "user",
                    "content":
                        f"The fragmented video description is: {caption}. "
                        "Please generate the response in the form of a Python dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                        "For example, your response should look like this: {'Q': 'Your question here...', 'A': 'Your answer here...'}. "
                        "Emphasize that the answer should focus on describing the video content as detailed as possible."
                }
            ]
        )
        # Convert response to a list of dictionary.
        response_message = completion["choices"][0]["message"]["content"]
        response_dict = ast.literal_eval(response_message)

        # Save the response dictionary into a JSON file
        json_file_path = os.path.join(output_dir, f"{key}.json")
        with open(json_file_path, "w") as f:
            json.dump(response_dict, f)

    print(f"Completed, Annotations saved in {output_dir}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments
    args = parse_args()

    # Read ground truth captions file
    gt_file = args.gt_caption_file
    with open(gt_file) as file:
        gt_captions = json.load(file)

    # Get the video_file_names
    video_files = list(gt_captions.keys())

    caption = {}
    for video_file in tqdm(video_files):
        key = video_file[:-4].rstrip(".") # Strip file extension.
        try:
            gt_sentences = gt_captions[key]['sentences']
        except KeyError:
            print(f"Warning: GT captions not found for video file. Skipping...")
            continue
        caption[key] = gt_sentences

    # Prepare list of caption files
    caption_files = [f'{video_id}.json' for video_id in caption.keys()]

    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set OpenAI API key
    openai.api_key = args.api_key
    num_tasks = args.num_tasks

    # Main loop: Continues until all question-answer pairs are generated for all captions
    while True:
        try:
            # Files that have already been completed.
            completed_files = os.listdir(args.output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            if len(incomplete_files) == 0:
                print("All tasks completed!")
                break

            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            num_tasks = min(len(incomplete_files), num_tasks)
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]

            task_args = [(caption, part, args.output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")
            print("Sleeping for 2 minutes...")
            time.sleep(120)  # wait for 2 minutes before trying again


if __name__ == "__main__":
    main()
