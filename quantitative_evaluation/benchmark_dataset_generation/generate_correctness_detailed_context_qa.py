import openai
import os
import argparse
import warnings
import json
import ast
from multiprocessing.pool import Pool

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--gt_caption_folder", required=True, help="The path to captions")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args


def annotate(gt_file, caption_files, output_dir):
    """
    Generate generic descriptive type questions and answers for each caption file using GPT-3.
    """
    for file in caption_files:
        key = file[:-5] # Strip file extension.
        caption = gt_file[key]
        try:
            # Generate GPT-3 response.
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": 
                            "You will play two roles: a human asking questions related to describing a video and an intelligent chatbot designed for video description and dense captioning. "
                            "Your task is to generate a detailed and descriptive paragraph based on the provided fragmented information about a video. "
                            "------"
                            "##TASK:"
                            "Users will provide a descriptions of a video, and you will generate ONE conversation-like question and answer related to describing the video in detail. "
                            "The question should ask to describe the video content in detail. "
                            "The answer should be a paraphrased and well-structured paragraph based on the provided description, as detailed as possible. "
                    },
                    {
                        "role": "user",
                        "content":
                            f"The user input is: {caption}. "
                            f"Please generate the response in the form of a Python dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                            "For example, your response should look like this: {'Q': 'Your question here...', 'A': 'Your answer here...'}. "
                            f"Emphasize that the answer should focus on describing the video content as detailed as possible."
                    }
                ]
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(response_dict, f)
        except Exception as e:
            print(f"Error processing file '{key}': {e}")


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    # Read ground truth captions.
    gt_captions = {}
    gt_files = os.listdir(args.gt_caption_folder)
    for file in gt_files:
        with open(os.path.join(args.gt_caption_folder, file), mode='r', encoding='utf-8-sig') as f:
            caption = f.read().replace('\n', '').replace('‘', "'").replace('’', "'")
            video_id = file[:-4]
            gt_captions[video_id] = caption

    caption_files = [f"{video_id}.json" for video_id in gt_captions.keys()]
    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the OpenAI API key.
    openai.api_key = args.api_key
    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(gt_captions, part, args.output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine qa pairs into single file when individual qa generation completes
    all_data = {}
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            with open(os.path.join(output_dir, filename)) as f:
                key = filename[:-5]
                all_data[key] = json.load(f)

    with open(args.output_json, 'w') as f:
        json.dump(all_data, f, indent=4)


if __name__ == "__main__":
    main()
