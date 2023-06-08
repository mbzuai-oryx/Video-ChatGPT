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

        # Generate QA pairs with OpenAI GPT-3: Summarization
        completion_0 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content":
                        "You play two roles: a human asking questions related to summarizing a video and an intelligent chatbot designed for video summarization and dense captioning. "
                        "Your task is video summarization. "
                        "As an AI assistant, assume that you have watched the video and generated the provided caption as the summary of the video. "
                        "Your task is to play the role of a human who asks three questions related to summarizing the video and then play the role of an AI assistant that provides paraphrased answers based on the video content and the provided caption."
                        "------"
                        "##TASK:"
                        "Users will provide a caption of a video, and you will generate a set of three conversation-like questions related to summarizing the video. "
                        "The questions and answers can be very similar, but they should all focus on summarizing the video content. "
                        "The answers should be paraphrased versions of the provided caption. "
                        "You have information about the video based on the provided caption and have summarized the events in it."
                        "Generate THREE different questions asking to summarize the video and provide detailed answers to each based on the caption. "
                        "------"
                        "##INSTRUCTIONS:"
                        "- The questions must be like a human conversation and focused on summarizing the video. "
                        "- The answers must be paraphrased versions of the provided caption, and they should be detailed and descriptive. "
                        "------"
                        "##SAMPLE QUESTIONS:"
                        "- Can you provide a summary of the video?"
                        "- What are the main events in the video?"
                        "- Could you briefly describe the video content?"
                },
                {
                    "role": "user",
                    "content":
                        f"The video caption is: {caption}. "
                        "Please generate the response in the form of a Python list of dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                        "For example, your response should look like this: [{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, {'Q': 'Your second question here...', 'A': 'Your second answer here...'}, {'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
                        "Emphasize that the questions and answers can be very similar, but they should all focus on summarizing the video content."
                }
            ]
        )
        # Extract Summary Based QA pairs
        # Convert response to a list of dictionary.
        response_message_0 = completion_0["choices"][0]["message"]["content"]
        response_dict_0 = ast.literal_eval(response_message_0)

        # Generate QA pairs with OpenAI GPT-3: Caption Based
        # Answers specifically restricted to information in the caption
        completion_1 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content":
                        "You play two roles: a human asking questions related to a video and an intelligent chatbot designed to help people find information from a given video. "
                        "Your task is video summarization, which will be used by users to understand different events in long videos by asking different questions based on the video. "
                        "The video summarization will be used for various applications such as surveillance, generate previews or summaries of video content for video search engines, "
                        "create highlights or summaries of sporting events, TV shows, and movies. "
                        "Your task is to first play the role of a human who asks questions related to a video and then play the role of an AI assistant that provides information based on the video content."
                        "------"
                        "##TASK:"
                        "Users will provide some information about a video, and you will generate a set of conversation-like questions and answers related to the video. "
                        "The questions should be designed to extract information directly from the given information, so that the provided information or parts of it can serve as the answers. "
                        "Generate THREE different descriptive and conversational style questions and detailed answers based on the given information. "
                        "------"
                        "##INSTRUCTIONS:"
                        "- The questions must be like a human conversation and based on the events in the video. "
                        "- The questions should be designed to extract information DIRECTLY from the given information, so that it or parts of it can serve as the answers. "
                        "- The answers must be detailed and descriptive, and they should directly reference the information provided. "
                        "- The questions can be related to the appearance, motion, trajectory, and reasoning. "
                        "------"
                        "##SAMPLE QUESTIONS:"
                        "- What is the man doing in the video?"
                        "- What are the girls doing in the video?"
                        "- Describe the appearance of the motorbike"
                        "- Is the person riding the bike wearing a helmet?"
                        "- How does the person repair the car?"
                },
                {
                    "role": "user",
                    "content":
                        f"The video caption is: {caption}. "
                        "Please generate the response in the form of a Python list of dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                        "For example, your response should look like this: [{'Q': 'Your first question here...', 'A': 'Your first answer here...'}, {'Q': 'Your second question here...', 'A': 'Your second answer here...'}, {'Q': 'Your third question here...', 'A': 'Your third answer here...'}]. "
                        "Emphasize that the ALL THREE questions must be designed to extract information DIRECTLY from the given information, so that it or parts of it can serve as the answers, and provide detailed and descriptive answers."
                }
            ]
        )
        # Extract Caption Based QA pairs
        # Convert response to a list of dictionary.
        response_message_1 = completion_1["choices"][0]["message"]["content"]
        response_dict_1 = ast.literal_eval(response_message_1)

        # Generate QA pairs with OpenAI GPT-3: Creative Based
        # TODO: Limit to samples with lengthy GT captions
        completion_2 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content":
                        "You play two roles: a human asking questions related to a video and an intelligent chatbot designed to help people find information from a given video. "
                        "You play two roles: a human asking creative questions related to a video and an intelligent chatbot designed to help people explore imaginative aspects of a given video. "
                        "Your task is to generate a conversation that dives into the creative interpretations and ideas inspired by the video, rather than summarizing its content. "
                        "As an AI assistant, assume that you have watched the video and generated the provided caption as the summary of the video. "
                        "Your task is to first play the role of a human who asks creative questions related to a video and then play the role of an AI assistant that provides imaginative responses based on the video content."
                        "##TASK:"
                        "Users will provide a caption of a video, and you will generate a conversation-like creative question and answer related to the video. "
                        "The question should be designed to explore imaginative aspects of the video, such as creating a story, poem, or alternate scenario inspired by the video. "
                        "You have information about the video based on the provided caption."
                        "Generate ONLY ONE creative questions and detailed answers based on the caption. "
                        "------"
                        "##INSTRUCTIONS:"
                        "- The question must be like a human conversation and inspired by the events in the video. "
                        "- The creative question should prompt for a poem, short story, alternate scenario, or other imaginative response inspired by the video content. "
                        "- The answer must be detailed, descriptive, and imaginative, showcasing creative interpretations of the video. "
                        "------"
                        "##SAMPLE QUESTIONS:"
                        "- Can you write a short poem inspired by the video?"
                        "- Create a short story that incorporates elements from the video."
                        "- How would you turn the video into a fairy tale with a moral lesson?"
                        "- Imagine the video as a movie scene. How would you describe its climax?"
                        "- Can you create a haiku that captures the essence of the video?"
                        "- Write a short, suspenseful thriller scene inspired by the video."
                        "- Write a brief scene from a sci-fi or fantasy novel inspired by the video."
                },
                {
                    "role": "user",
                    "content":
                        f"The video caption is: {caption}. "
                        "Please generate the response in the form of a Python dictionary string with keys 'Q' for question and 'A' for answer. Each corresponding value should be the question and answer text respectively. "
                        "For example, your response should look like this: {'Q': 'Your question here...', 'A': 'Your answer here...'}. "
                        "Focus on generating ONLY ONE creative question and answer inspired by the video."
                }
            ]
        )
        # Extract Creative Based QA pairs
        # Convert response to a list of dictionary.
        response_message_2 = completion_2["choices"][0]["message"]["content"]
        response_dict_2 = ast.literal_eval(response_message_2)

        # Combine all QA pairs generated for the sample
        combined_responses = response_dict_0 + response_dict_1
        combined_responses.append(response_dict_2)

        # Save the response dictionary into a JSON file
        json_file_path = os.path.join(output_dir, f"{key}.json")
        with open(json_file_path, "w") as f:
            json.dump(combined_responses, f)

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
