from os import environ, makedirs as os_makedirs, listdir as os_listdir
from os.path import exists as os_path_exists
from sys import stdout as sys_stdout
from math import ceil as math_ceil
from pickle import dump as pickle_dump
from argparse import ArgumentParser
from json import load as json_load, dump as json_dump
from collections import defaultdict
import numpy as np
from torch import (
    zeros as torch_zeros,
    as_tensor as torch_as_tensor,
    float16 as torch_float16,
    float32 as torch_float32,
    uint8 as torch_uint8,
    device as torch_device,
)
from torch.nn.functional import interpolate as F_interpolate
from PIL.Image import fromarray as Image_fromarray
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
from decord import VideoReader, cpu, gpu
from decord.bridge import set_bridge
set_bridge('torch')
environ['DECORD_EOF_RETRY_MAX'] = '20480'
# v_UvPUywSVy1k.mp4 from activitynet
# social-iq: 1.21it/s with cpu
# social-iq: 1.09it/s with gpu
# Can't process /home/zhanwen/siq2/siq2/video/n6-ef_YHeJU.mp4


def save_features(video_clip_features, clip_feat_path, ts_by_video, ts_by_videol_fpath):
    for key, features in video_clip_features.items():
        with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
            pickle_dump(features, f)

    with open(ts_by_videol_fpath, 'w') as f:
        json_dump(ts_by_video, f)


def load_video(vis_path, device, cuts, num_frm=100):
    with open(vis_path, 'rb') as file_in:
        # vr = VideoReader(file_in, ctx=gpu(0), num_threads=0)
        vr = VideoReader(file_in, ctx=gpu(0), num_threads=0)
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm, cuts)
    frame_idx, ts_to_frames = get_seq_frames(total_frame_num, total_num_frm, cuts)
    # img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)
    img_array = vr.get_batch(frame_idx) # (n_clips*num_frm, H, W, 3)
    del vr

    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch_as_tensor(img_array, dtype=torch_float32, device=device).permute(0, 3, 1, 2)
        img_array = F_interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(device='cpu', dtype=torch_uint8, non_blocking=True).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    # clip_imgs = []
    # for j in range(total_num_frm):
    #     clip_imgs.append(Image_fromarray(img_array[0, j]))

    clip_imgs = [Image_fromarray(img_array[0, j]) for j in range(total_num_frm)]
    return clip_imgs, ts_to_frames


def get_seq_frames(total_num_frames, desired_num_frames, cuts: list[str]):
    # Even selection.
    cuts = sorted(cuts)
    cuts_float_start = []
    cuts_float_end = []
    for cut in cuts:
        start_str, end_str = cut.split('-')
        start_float, end_float = float(start_str), float(end_str)
        cuts_float_start.append(start_float)
        cuts_float_end.append(end_float)
    cuts_float_end = sorted(cuts_float_end)

    summy = sum(cuts_float_end)

    percentiles = [cut_float/summy for cut_float in cuts_float_end]
    frame_counts = [total_num_frames * percentile for percentile in percentiles]

    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    ts_out = defaultdict(list)
    frame_counts_index = 0
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        frame_idx = (start + end) // 2
        frame_threshold = frame_counts[frame_counts_index]
        if frame_idx >= frame_threshold:
            frame_idx += 1
        ts_subvideo = cuts[frame_counts_index]
        ts_out[ts_subvideo].append(frame_idx)
        seq.append(frame_idx)

    return seq, ts_out


def get_spatio_temporal_features(features, num_temporal_tokens=100):
    t, s, c = features.shape

    # features: [100, 256, 1024]
    temporal_tokens = np.mean(features, axis=1) # [100, 1,1024], or [100, 1024]
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

    spatial_tokens = np.mean(features, axis=0)  # [1, 256, 1024], [256, 1024]
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)
    # The first 100 are the average spatial embedding of each of the frames. The last 256 are the average temporal embedding of each of the spatial embedding.
    # So if we zero out a timeframe, then the corresponding temporal_tokens[t]=0.The last 256 will be reduced by the value of that frame. So if we can load the frame, we can reverse engineer it.
    # However, we might not be able to find the original embedding, because it may have been discarded.
    # spatial_tokens = [1, 256, 1024]+[1, 256, 1024]+[1, 256, 1024]/100 - [1, 1, 1024] * 256/100 as an approximation.
    # The temporal tokens at t = [1, 1, 1024].
    return sp_features


def parse_args():
    parser = ArgumentParser(description="Training")

    parser.add_argument("--ts_by_videol_fpath", required=True,
                        help="Path to read the qa file from.")
    parser.add_argument("--qa_path", required=True, help="Path to read the qa file from.")
    parser.add_argument("--video_dir_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path", required=True, help="The output dir to save the features in.")
    parser.add_argument("--infer_batch", required=False, type=int, default=32,
                        help="Number of frames/images to perform batch inference.")

    args = parser.parse_args()

    return args


def get_unique_videos(qas):
    qas_by_video = defaultdict(list)
    for qa in qas:
        qas_by_video[qa['video_id']].append(qa)
    return qas_by_video


def get_unique_ts(qas):
    qas_by_video = defaultdict(set)
    for qa in qas:
        qas_by_video[qa['video_id']].add(qa['ts'])
    return qas_by_video



def main():
    args = parse_args()
    ts_by_videol_fpath = args.ts_by_videol_fpath
    qa_path = args.qa_path
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    infer_batch = args.infer_batch
    with open(qa_path, 'rb') as file_in:
        qas = json_load(file_in)

    # qas_by_video = get_unique_videos(qas)
    os_makedirs(clip_feat_path, exist_ok=True)
    device = torch_device('cuda:0')
    non_blocking = False

    # Initialize the CLIP model
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch_float16, device_map='cuda')
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch_float16, device_map='cuda')
    vision_tower.eval()

    print('Initialized CLIP model')
    all_videos = os_listdir(video_dir_path)
    video_clip_features = {}
    counter = 0
    image_processor_preprocess = image_processor.preprocess
    select_hidden_state_layer = -2

    cuts_by_video = get_unique_ts(qas)
    ts_by_video = defaultdict(lambda: defaultdict(list))  # vid: {ts: indices}

    for video_name in tqdm(all_videos, file=sys_stdout):
        cuts = cuts_by_video[video_name]
        print(f'Started processing {video_name}')
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]
        if os_path_exists(f"{clip_feat_path}/{video_id}.pkl"):  # Check if the file is already processed
            continue
        try:
            video, ts_out = load_video(video_path, device, cuts, num_frm=100)
            ts_by_video[video_name] = ts_out
            video_tensor = image_processor_preprocess(video, return_tensors='pt')['pixel_values'].to(torch_float16, non_blocking=non_blocking)

            n_chunk = len(video_tensor)
            video_features = torch_zeros((n_chunk, 256, 1024), dtype=torch_float32, device=device)
            n_iter = int(math_ceil(n_chunk / float(infer_batch)))
            for i in range(n_iter):
                min_ind = i * infer_batch
                max_ind = (i + 1) * infer_batch
                video_batch = video_tensor[min_ind:max_ind].to(device=device, non_blocking=non_blocking)

                image_forward_outs = vision_tower(video_batch, output_hidden_states=True)

                select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                batch_features = select_hidden_state[:, 1:]
                video_features[min_ind:max_ind] = batch_features.detach().cpu()

            video_clip_features[video_id] = get_spatio_temporal_features(video_features.to(device='cpu', dtype=torch_float16, non_blocking=non_blocking).numpy())
            counter += 1

        except Exception as e:
            print(f"Can't process {video_path}")
            raise e

        if counter % 512 == 0:  # Save after every 512 videos, update this number as per your requirements
            save_features(video_clip_features,
                          clip_feat_path, ts_by_video, ts_by_videol_fpath)
            video_clip_features = {}

    save_features(video_clip_features, clip_feat_path,
                  ts_by_video, ts_by_videol_fpath)


if __name__ == "__main__":
    main()
