import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--input_json_file", required=True,
                        help="Path to input json file (i.e. VideoInstruct_Dataset.json)")
    parser.add_argument("--output_json_file", required=True,
                        help="Path to output json file (i.e. VideoInstruct_Dataset_Train.json)")
    parser.add_argument("--clip_feature_path", required=False, default="",
                        help="Path to generated CLIP feature paths to filter any missing video ids (optional).")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    input_json_file = args.input_json_file
    output_json_file = args.output_json_file
    clip_feature_path = args.clip_feature_path

    clip_features_files_witout_extension = ""
    if clip_feature_path:
        clip_features_files = os.listdir(clip_feature_path)
        clip_features_files_witout_extension = []
        for file in clip_features_files:
            clip_features_files_witout_extension.append(file.split('.')[0])


    input_json_contents = json.load(open(input_json_file, 'r'))
    output_json_contents = []
    for i, content in enumerate(input_json_contents):
        valid = False
        if not clip_feature_path:
            valid = True
        elif content['video_id'] in clip_features_files_witout_extension:
            valid = True

        if valid:
            output_content = {'id': content['video_id'], 'video': f"{content['video_id']}.pkl", 'conversations': []}
            # This is critical
            if i % 2 == 0:
                output_content['conversations'].append({'from': 'human', 'value': f"{content['q']}\n<video>"})
            else:
                output_content['conversations'].append({'from': 'human', 'value': f"<video>\n{content['q']}"})
            output_content['conversations'].append({'from': 'gpt', 'value': content['a']})
            output_json_contents.append(output_content)

    print(f"Total annotations retained: {len(output_json_contents)}")
    with open(output_json_file, 'w') as f:
        json.dump(output_json_contents, f)


if __name__ == "__main__":
    main()
