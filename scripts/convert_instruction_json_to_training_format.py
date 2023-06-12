import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--input_json_file", required=True, help="")
    parser.add_argument("--output_json_file", required=True, help="")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    input_json_file = args.input_json_file
    output_json_file = args.output_json_file

    input_json_contents = json.load(open(input_json_file, 'r'))
    output_json_contents = []
    for i, content in enumerate(input_json_contents):
        output_content = {'id': content['video_id'], 'video': f"{content['video_id']}.pkl", 'conversations': []}

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
