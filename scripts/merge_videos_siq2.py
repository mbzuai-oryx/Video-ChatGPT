# pip install git+https://github.com/Zulko/moviepy.git
from json import load as json_load
from collections import defaultdict
from os import makedirs as os_makedirs
from os.path import join as os_path_join, exists as os_path_exists
from argparse import ArgumentParser
from pkg_resources import parse_version
from tqdm import tqdm
from PIL import Image as pil
from moviepy.editor import VideoFileClip, concatenate_videoclips


def parse_args():
    parser = ArgumentParser(description="Merging Videos")

    parser.add_argument("--video_dirpath_in", required=True, help="")
    parser.add_argument("--video_dirpath_out", required=True, help="")
    parser.add_argument("--qa_path", required=True, help="")

    args = parser.parse_args()

    return args


def get_unique_vids(qa_val_merged_n3):
    qas_by_vid = defaultdict(list)

    for qa in qa_val_merged_n3:
        qas_by_vid[qa['vid_name']].append(qa)

    return qas_by_vid


def concatenate(video_clip_paths, output_path, method="compose"):
    """Concatenates several video files into one video file
    and save it to `output_path`. Note that extension (mp4, etc.) must be added to `output_path`
    `method` can be either 'compose' or 'reduce':
        `reduce`: Reduce the quality of the video to the lowest quality on the list of `video_clip_paths`.
        `compose`: type help(concatenate_videoclips) for the info"""
    # create VideoFileClip object for each video file
    clips = [VideoFileClip(c) for c in video_clip_paths]
    if method == "reduce":
        # calculate minimum width & height across all clips
        min_height = min([c.h for c in clips])
        min_width = min([c.w for c in clips])
        # resize the videos to the minimum
        clips = [c.resize((min_width, min_height)) for c in clips]
        # concatenate the final video
        final_clip = concatenate_videoclips(clips)
    elif method == "compose":
        # concatenate the final video with the compose method provided by moviepy
        final_clip = concatenate_videoclips(clips, method="compose")
    # write the output video file
    final_clip.write_videofile(output_path)


def main():

    args = parse_args()
    video_dirpath_in = args.video_dirpath_in
    video_dirpath_out = args.video_dirpath_out
    qa_path = args.qa_path
    if parse_version(pil.__version__)>=parse_version('10.0.0'):
        pil.ANTIALIAS=pil.LANCZOS

    with open(qa_path, 'rb') as file_in:
        qas = json_load(file_in)

    qas_by_vid = get_unique_vids(qas)

    vids_new = list(qas_by_vid.keys())

    if not os_path_exists(video_dirpath_out):
        os_makedirs(video_dirpath_out)

    for vid_new in tqdm(vids_new):
        subv_ids = vid_new.split('+')
        paths = [os_path_join(video_dirpath_in, f'{subv_id}.mp4') for subv_id in subv_ids]
        concatenate(paths, os_path_join(video_dirpath_out, f'{vid_new}.mp4'))


if __name__ == '__main__':
    main()
