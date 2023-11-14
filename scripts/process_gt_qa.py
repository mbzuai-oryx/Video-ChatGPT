from json import load as json_load, dump as json_dump
from argparse import ArgumentParser
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description="Training")

    parser.add_argument("--input_json_file", required=True, help="")
    parser.add_argument("--ts_dict_fpath", required=True, help="")
    parser.add_argument("--output_json_file", required=True, help="")

    return parser.parse_args()


def main():
    args = parse_args()
    input_json_file = args.input_json_file
    ts_dict_fpath = args.ts_dict_fpath
    output_json_file = args.output_json_file

    with open(input_json_file, 'rb') as file_in:
        input_json_contents = json_load(file_in)

    with open(ts_dict_fpath, 'rb') as file_in:
        ts_dict = json_load(file_in)

    output_json_contents = []
    for content in tqdm(input_json_contents):
        vid_name = content['vid_name']
        ts = content['ts']
        frames = ts_dict[vid_name][ts]
        content['frame_idx_start'] = frames[0]
        content['frame_idx_end'] = frames[-1]
        output_json_contents.append(content)

    print(f"Total annotations retained: {len(output_json_contents)}")
    with open(output_json_file, 'w') as file_out:
        json_dump(output_json_contents, file_out)


if __name__ == "__main__":
    main()
