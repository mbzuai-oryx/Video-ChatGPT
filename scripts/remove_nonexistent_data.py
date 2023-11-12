from json import load as json_load, dump as json_dump
from os.path import join as os_path_join, exists as os_path_exists
from argparse import ArgumentParser
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description="RemoveNonexistentVideo")

    parser.add_argument("--qa_json_fpath_in", required=True, type=str,
                        help="Path to read the qa from.")
    parser.add_argument("--qa_json_fpath_removed_out", required=True, type=str,
                        help="The output file path to save the qas to.")
    parser.add_argument("--qa_json_fpath_nonexistent_out", required=True, type=str,
                        help="The output file path to save the qas to.")
    parser.add_argument("--video_features_dir", required=True, type=str,
                        help="Where the video files are.")

    return parser.parse_args()


def main():
    args = parse_args()
    video_features_dir = args.video_features_dir
    qa_json_fpath_in = args.qa_json_fpath_in
    qa_json_fpath_removed_out = args.qa_json_fpath_removed_out
    qa_json_fpath_nonexistent_out = args.qa_json_fpath_nonexistent_out

    with open(qa_json_fpath_in, 'rb') as file_in:
        qa_json_train = json_load(file_in)

    qa_json_train_removed = []
    nonexistent_features = []
    for qa_pair in tqdm(qa_json_train):
        video_feature_fname = qa_pair['vid_name']
        video_feature_fpath = os_path_join(
            video_features_dir, f'{video_feature_fname}.pkl')
        if os_path_exists(video_feature_fpath):
            qa_json_train_removed.append(qa_pair)
        else:
            nonexistent_features.append(qa_pair)

    with open(qa_json_fpath_removed_out, 'w') as file_out:
        json_dump(qa_json_train_removed, file_out)

    with open(qa_json_fpath_nonexistent_out, 'w') as file_out:
        json_dump(nonexistent_features, file_out)

    len_qa_json_train_removed = len(qa_json_train_removed)
    len_nonexistent_features = len(nonexistent_features)
    print(
        f'Train: Out of a total of {len_qa_json_train_removed+len_nonexistent_features} QA pairs, {len_nonexistent_features} are unavailable, leaving {len_qa_json_train_removed}.')


if __name__ == '__main__':
    main()
