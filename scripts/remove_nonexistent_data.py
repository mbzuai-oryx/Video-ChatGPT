from json import load as json_load, dump as json_dump
from os.path import join as os_path_join, exists as os_path_exists
from tqdm import tqdm


VIDEO_FEATURES_DIR = '../data/siq2/video_features'
qa_json_train_fpath = '../data/siq2/qa/qa_train_instruction.json'
qa_json_train_removed_fpath = '../data/siq2/qa/qa_train_instruction_removed.json'
qa_json_train_nonexistent_fpath = '../data/siq2/qa/qa_train_nonexistent.json'


if __name__ == '__main__':
    with open(qa_json_train_fpath, 'rb') as file_in:
        qa_json_train = json_load(file_in)

    qa_json_train_removed = []
    nonexistent_features = []
    for qa_pair in tqdm(qa_json_train):
        video_feature_fname = qa_pair['video']
        video_feature_fpath = os_path_join(VIDEO_FEATURES_DIR, video_feature_fname)
        if os_path_exists(video_feature_fpath):
            qa_json_train_removed.append(qa_pair)
        else:
            nonexistent_features.append(qa_pair)

    with open(qa_json_train_removed_fpath, 'w') as file_out:
        json_dump(qa_json_train_removed, file_out)

    with open(qa_json_train_nonexistent_fpath, 'w') as file_out:
        json_dump(nonexistent_features, file_out)

    len_qa_json_train_removed = len(qa_json_train_removed)
    len_nonexistent_features = len(nonexistent_features)
    print(f'Train: Out of a total of {len_qa_json_train_removed+len_qa_json_train_removed} QA pairs, {len_nonexistent_features} are unavailable, leaving {len_qa_json_train_removed}.')
