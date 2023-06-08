# Training Video-ChatGPT
We train Video-ChatGPT model on our 100K video instruction dataset. We initialize the training from LLaVA.
Please follow the instructions below to train Video-ChatGPT-7B model.

### Prepare LLaVA weights

Video-ChatGPT is build using LLaVA. Please follow the following instructions to get LLaVA weights.

- Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
- Use the following scripts to get LLaVA weights by applying our delta.
```shell
python scripts/apply_delta.py \ 
        --base-model-path <path to LLaMA 7B weights> \
        --target-model-path LLaVA-Lightning-7B-v1-1 \
        --delta-path liuhaotian/LLaVA-Lightning-7B-delta-v1-1
```

The above command will download the LLaVA-Lightening-7B-v1-1 delta from HuggingFace, apply it to the provided LLaMA 
weights and save the LLaVA-Lightening-7B-v1-1 weights in the current directory.

Alternatively you can download the ready LLaVA-Lightening-7B weights from [mmaaz60/LLaVA-Lightening-7B-v1-1](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1).


## Prepare Dataset

**1. Download our 100K video instruction dataset from** 
[this download link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EWxYslvDeX1PijKWM_WxTkkBDXDDD350YnUQOkbcL8V7Xg?e=Lq9itD).


**2. Convert the downloaded Json into the required format for training,**

```shell
python scripts/convert_instruction_json_to_training_format.py \
        --input_json_file <path to json file downloaded in step 2> \
        --output_json_file video_chatgpt_training.json
```
The above script will generate `video_chatgpt_training.json` required to train our model.

**3. Download ActivityNet videos**

All the videos annotated in our work are taken from ActivityNet dataset. 
We provide the ids of all the required videos in the [train_video_ids.txt](train_video_ids.txt) file. 
Please follow the instructions on the [official site](http://activity-net.org/download.html) to download the videos. 
Alternatively, you can download these from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EnLRDehrr8lGqHpC5w1zZ9QBnsiVffYy5vCv8Hl14deRcg?e=Ul5DUE).

**4. Prepare Spatio-Temporal features using CLIP**
Note that for training efficiency, we pre-computed the video spatio-temporal features and use them directly during training.
After downloading the videos, please use the following command to generate CLIP spatio-temporal features.

```shell
python scripts/save_spatio_temporal_clip_features.py \
        --video_dir_path <path to the directory containing all the videos> \
        --clip_feat_path <The output dir to save the features in.>
```
The script will generate the spatiotempral features for each video and 
save one pickle file per video in directory specified by `--clip_feat_path` argemunt. 
Alternatively, you can download the pre-computed spatiotemporal CLIP features from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EnLRDehrr8lGqHpC5w1zZ9QBnsiVffYy5vCv8Hl14deRcg?e=Ul5DUE).

## Train Video-ChatGPT

Train on 8 A100 40GB GPUs using the following command,
```shell
torchrun --nproc_per_node=8 --master_port 29001 video_chatgpt/train/train_mem.py \
          --model_name_or_path /share/data/drive_4/Maaz/LLMs/llama/LLaVA_7B-1.1 \
          --version v1 \
          --data_path <path to the video_chatgpt using `convert_instruction_json_to_training_format.py` script.> \
          --video_folder <path to the spatio-temporal features generated in step 4 using `save_spatio_temporal_clip_features.py` script> \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./Video-ChatGPT_7B-1.1_Checkpoints \
          --num_train_epochs 3 \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 3000 \
          --save_total_limit 3 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True
```
