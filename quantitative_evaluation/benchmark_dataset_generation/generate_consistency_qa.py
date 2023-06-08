import openai
import os
import argparse
import warnings
import json
import ast
from multiprocessing.pool import Pool

# Disable warnings.
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
    Generate questions and answers for each caption file using GPT-3.
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
                            "Your primary task is to formulate two distinct but conceptually similar questions, such that when asked about the same video-information, they correspond to the same answer. "
                            "------"
                            "##TASK:"
                            "When given details about a video, your task is to generate two questions asked in different ways. The crucial aspect is to frame these questions so that they are conceptually alike but phrased differently, leading to the exact same answer. "
                            "The questions should be cleverly designed to extract the same information directly from the video details given, so that the provided information or parts of it can serve as the answer. It's important that both questions yield the SAME answer. "
                            "- Generate TWO questions and ONE answer. The purpose is to extract identical information from both questions. Therefore, formulate your questions in a way that the given details can serve directly as the answer. "
                            "------"
                            "##SAMPLE QUESTIONS:"
                            "- {'Q1': 'What is the colour of the cycle the boy rides?', 'Q2': 'Can you describe the cycle the boy is riding?', 'A': 'The boy is riding a red bicycle with a basket.'}"
                            "- {'Q1': 'What is the baby girl doing in the video?', 'Q2': 'Can you see the baby girl engaged in an activity in the video?', 'A': 'The baby girl is reading a book in the video.'}"
                    },
                    {
                        "role": "user",
                        "content":
                            f"The user input is: {caption}. "
                            f"Please generate the response in the form of a Python dictionary string with keys 'Q1', 'Q2', and 'A', where value of 'Q1' is  first question, 'Q2' for second question and 'A' is the answer to both questions. Each corresponding value should be the question or answer text respectively. "
                            "For example, your response should look like this: {'Q1': 'Your first question here...', 'Q2': 'Your second question here...', 'A': 'Your answer to both questions here...'}. "
                            "Remember, it's critical to ensure that both questions are designed to extract the same details from the video, leading to the same answer."
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
        # Read human-assisted annotations from individual text files.
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
            # Files that have already been completed.
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
