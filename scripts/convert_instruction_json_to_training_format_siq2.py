from json import load as json_load, dump as json_dump
from argparse import ArgumentParser


PROMPT_STRING = 'Please watch the video and choose the correct answer from the following question:\n'


def parse_args():
    parser = ArgumentParser(description="Training")

    parser.add_argument("--input_json_file", required=True, help="")
    parser.add_argument("--output_json_file", required=True, help="")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    input_json_file = args.input_json_file
    output_json_file = args.output_json_file

    with open(input_json_file, 'r') as file_in:
        input_json_contents = json_load(file_in)
    output_json_contents = []
    for i, content in enumerate(input_json_contents):
        video_id = content['vid_name']
        question = content['q']
        a0 = f"A: {content['a0']}"
        a1 = f"B: {content['a1']}"
        a2 = f"C: {content['a2']}"
        a3 = f"D: {content['a3']}"
        answer_idx = content['answer_idx']
        answer = [a0, a1, a2, a3][answer_idx]
        prompt = f'{PROMPT_STRING}\n{question}\n{a0}\n{a1}\n{a2}\n{a3}'


        output_content = {'id': video_id, 'video': f"{video_id}.pkl", 'conversations': []}

        if i % 2 == 0:
            output_content['conversations'].append({'from': 'human', 'value': f"{prompt}\n<video>"})
        else:
            output_content['conversations'].append({'from': 'human', 'value': f"<video>\n{prompt}"})

        output_content['conversations'].append({'from': 'gpt', 'value': answer})
        output_json_contents.append(output_content)

    print(f"Total annotations retained: {len(output_json_contents)}")
    with open(output_json_file, 'w') as f:
        json_dump(output_json_contents, f)


if __name__ == "__main__":
    main()
