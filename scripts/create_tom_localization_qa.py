from collections import defaultdict
from random import seed as random_seed, shuffle
from json import load as json_load, dump as json_dump
from argparse import ArgumentParser
from typing import Tuple
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description="CreateTomLocalization")

    parser.add_argument("--qa_json_fpath_in", required=True, type=str,
                        help="Path to read the qa from.")
    parser.add_argument("--qa_json_fpath_out", required=True, type=str,
                        help="The output file path to save the qas to.")
    parser.add_argument("--n", required=True, type=int,
                        help="Number of subvideos to split.")
    parser.add_argument("--seed", required=False, type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sep", required=False, type=str, default='+',
                        help="Separator for concatenating subvideo ids.")

    return parser.parse_args()


def shuffle_dict(dict_in: dict) -> dict:
    dict_in_items_list = list(dict_in.items())
    shuffle(dict_in_items_list)
    return dict(dict_in_items_list)


# Split into ns.
def split_dict_into_n_keys(dict_in: dict, n: int) -> list[list[str]]:
    dict_keys = list(dict_in.keys())
    dict_keys_split = [dict_keys[i:i + n] for i in range(0, len(dict_in), n)]
    return dict_keys_split


def add_start_end_to_ts(start: float, end: float, delta: float) -> Tuple[float, float]:
    start += delta
    end += delta
    return start, end


def add_start_end_to_ts_str(start_str: str, end_str: str, delta: float) -> Tuple[float, float]:
    start, end = float(start_str), float(end_str)
    return add_start_end_to_ts(start, end, delta)


def merge_subvideos(dict_in: dict, dict_keys_split_ns: list[list[str]], sep: str) -> dict[str, list]:
    dict_new = {}
    end_new = 0

    for v_ids in tqdm(dict_keys_split_ns):
        v_id_new = ''
        qas_new_all_subvs = []
        delta = 0
        for v_id in v_ids:
            v_id_new += v_id if v_id_new == '' else f'{sep}{v_id}'
            qas = dict_in[v_id]
            for qa_new in qas:
                start_str, end_str = qa_new['ts'].split('-')
                start_new, end_new = add_start_end_to_ts_str(start_str, end_str, delta)
                qa_new['ts'] = f'{start_new}-{end_new}'
            delta = end_new
            qas_new_all_subvs.extend(qas)
        # After v_id_new is finalized, replace
        for qa_new in qas_new_all_subvs:
            qa_new['qid_original'] = qa_new['qid']
            qa_new['qid'] = qa_new['qid'].replace(qa_new['vid_name'], v_id_new)
            qa_new['vid_name_original'] = qa_new['vid_name']
            qa_new['vid_name'] = v_id_new

        shuffle(qas_new_all_subvs) # Mix all subvideos to avoid memorization
        dict_new[v_id_new] = qas_new_all_subvs
    dict_new = shuffle_dict(dict_new)
    list_of_dicts = sum(list(dict_new.values()), [])
    return list_of_dicts


def get_unique_videos(list_of_dicts: list[dict]) -> dict:
    video_qa_dict = defaultdict(list)

    for qa in list_of_dicts:
        video_qa_dict[qa['vid_name']].append(qa)
    return video_qa_dict


def main():
    args = parse_args()
    qa_json_fpath_in = args.qa_json_fpath_in
    qa_json_fpath_out = args.qa_json_fpath_out
    n = args.n
    seed = args.seed
    sep = args.sep
    random_seed(seed)
    with open(qa_json_fpath_in, 'r') as file_in:
        list_of_dicts = json_load(file_in)
    # Second, shuffle the video names and divide them into subvideo groups of N
    dict_in = get_unique_videos(list_of_dicts)
    dict_keys_split_ns = split_dict_into_n_keys(dict_in, n)
    # Third, merge subvideos.
    list_of_dicts_new = merge_subvideos(dict_in, dict_keys_split_ns, sep)
    print(f'create_tom_localization.py: list_of_dicts_og={len(list_of_dicts)}; len(list_of_dicts_new)={len(list_of_dicts_new)}')
    with open(qa_json_fpath_out, 'w') as file_out:
        json_dump(list_of_dicts_new, file_out)

if __name__ == '__main__':
    main()
