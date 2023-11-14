# install_azure.sh

# The following packages will be REMOVED:
#   cuda-drivers-fabricmanager-470 libnvidia-decode-545 libnvidia-encode-545 libnvidia-fbc1-545 nvidia-dkms-545 nvidia-driver-545 nvidia-kernel-source-545
#   xserver-xorg-video-nvidia-545
# The following packages have been kept back:
#   moby-cli moby-engine nvidia-container-toolkit nvidia-container-toolkit-base
# The following packages will be upgraded:
#   libcudnn8 libcudnn8-dev libnccl-dev libnccl2 libnvidia-container-tools libnvidia-container1 libxnvctrl0 nvidia-fabricmanager-470 nvidia-settings
# 9 upgraded, 0 newly installed, 8 to remove and 4 not upgraded.

# 0. (DONE) Remove stale apt sources that break modern NVIDIA toolchain installations:
sudo rm /etc/apt/sources.list.d//nccl-2.2.13-ga-cuda9.2.list /etc/apt/sources.list.d/nccl-2.2.13-ga-cuda9.2.list.save
# TODO: remove a certain old NVIDIA apt-key with sudo apt-key remove ...


# 1. (DONE)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb # 20.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb # 22.04
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 2.1 Install nvidia-driver-545
sudo apt update
sudo apt full-upgrade
sudo apt install nvidia-driver-545

# 2.2 Install CUDA 12.3
cd # Cannot have .deb be in a /mnt drive.

sudo apt update
sudo apt install cuda-toolkit-12-3
sudo bash -c "echo '/usr/local/cuda/lib64' > /etc/ld.so.conf"
sudo bash -c "echo 'LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /etc/environment"
# also prepend /usr/local/cuda/bin: to the PATH variable value in sudo vim /etc/environment.
sudo ldconfig

# (DONE) Install cuDNN
sudo apt install libcudnn8 libcudnn8-dev libcudnn8-samples

# (DONE) NCCL:
sudo apt install libnccl2 libnccl-dev

# (DONE) NVIDIA cuSparse
sudo apt install libcusparselt0 libcusparselt-dev

# (DONE) NVIDIA Docker and Moby Docker engine
sudo apt install nvidia-container-toolkit moby-engine moby-cli

# (DONE) NVSwitch drivers
sudo apt install nvidia-fabricmanager-545 nvidia-fabricmanager-dev-545

# (DONE) Remaining nvidia libs
sudo apt install libxnvctrl0 nvidia-settings

# apt clean up
sudo apt autoremove
sudo apt clean
sudo apt autoclean


# The libs

export ENV_NAME=vtom
export INSTALL_DIR=/mnt/batch/tasks/shared/LS_root/mounts/clusters/vtom-a100-x4-n1/code/Users/video.tom/
export PROJECT_DIR=${INSTALL_DIR}/${ENV_NAME}
export ENVS_DIR=${INSTALL_DIR}/envs

mkdir -p ${ENVS_DIR}
cd ${INSTALL_DIR}


# Instal nv-codec-headers
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
sudo make install

# 1b. Install ffmpeg5 with NVIDIA Video Codec SDK support
cd ${INSTALL_DIR}
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
sudo apt install yasm libx264-dev libgnutls28-dev
export MY_SM=80
./configure \
  --extra-cflags='-I/usr/local/cuda/include' \
  --extra-ldflags='-L/usr/local/cuda/lib64' \
  --nvccflags="-gencode arch=compute_${MY_SM},code=sm_${MY_SM} -O2" \
  --disable-doc \
  --enable-decoder=aac \
  --enable-decoder=h264 \
  --enable-decoder=h264_cuvid \
  --enable-decoder=rawvideo \
  --enable-indev=lavfi \
  --enable-encoder=libx264 \
  --enable-encoder=h264_nvenc \
  --enable-demuxer=mov \
  --enable-muxer=mp4 \
  --enable-filter=scale \
  --enable-filter=testsrc2 \
  --enable-protocol=file \
  --enable-protocol=https \
  --enable-gnutls \
  --enable-shared \
  --enable-gpl \
  --enable-nonfree \
  --enable-cuda-nvcc \
  --enable-libx264 \
  --enable-nvenc \
  --enable-cuvid \
  --disable-postproc \
  --disable-static \
  --enable-nvdec
make clean
make -j
sudo make install
sudo sh -c "echo '/usr/local/lib' >> /etc/ld.so.conf"
sudo ldconfig


# 1c. Confirm your ffmpeg has nvcodec enabled
# Examples in https://pytorch.org/audio/stable/build.ffmpeg.html#checking-the-intallation
ffprobe -hide_banner -decoders | grep h264
ffmpeg -hide_banner -encoders | grep 264
src="https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
ffmpeg -hide_banner -y -vsync 0 \
     -hwaccel cuvid \
     -hwaccel_output_format cuda \
     -c:v h264_cuvid \
     -resize 360x240 \
     -i "${src}" \
     -c:a copy \
     -c:v h264_nvenc \
     -b:v 5M test.mp4
rm test.mp4

# 2.2 Install Anaconda under ${HOME}. Will not work under /mnt
cd
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
sh Anaconda3-2023.09-0-Linux-x86_64.sh
# Say yes to everything
source .bashrc

export ENV_NAME=vtom
export INSTALL_DIR=/mnt/batch/tasks/shared/LS_root/mounts/clusters/vtom-a100-x4-n1/code/Users/video.tom/
export PROJECT_DIR=${INSTALL_DIR}/${ENV_NAME}
export ENVS_DIR=${INSTALL_DIR}/envs

conda update --all
# conda create -p ${ENVS_DIR}/${ENV_NAME} python=3.11 pip numpy
conda create -n vtom python=3.11 pip numpy
# conda config --add pkgs_dirs ${ENVS_DIR}
conda activate vtom

conda install -c pytorch magma-cuda121 # TODO: include magma-cuda122 building in the future
conda install cmake ninja astunparse expecttest hypothesis psutil pyyaml requests setuptools typing-extensions sympy filelock networkx jinja2 fsspec packaging mkl mkl-include
pip install jinja2 types-dataclasses optree

# 2.3 Install PyTorch from source
cd && git clone --recursive --single-branch --branch v2.1.0 https://github.com/pytorch/pytorch.git
cd pytorch
# 1. sync submodules
git submodule sync
git submodule update --init --recursive
conda activate vtom
export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export TORCH_CUDA_ARCH_LIST="8.0"
export USE_FFMPEG=1
export USE_SYSTEM_NCCL=1
python setup.py clean && echo "Done Cleaning"
python setup.py develop | tee install_pytorch.log # NOTE: you cannot remove the pytorch folder. Otherwise you'll get ModuleNotFoundError: No module named 'torch'
echo "DONE building pytorch"
cd
rm ${CONDA_PREFIX}/lib/libffi.7.so ${CONDA_PREFIX}/lib/libffi.so.7 # Solves `Undefined pointer to LIBFFI_7.0_BASE`
python -c "import torch; print(torch.cuda.is_available()); exit()"
python -c "from torch import randn, matmul; tensor1 = tensor2 = randn(3, device='cuda'); out = matmul(tensor1, tensor2); print(out.mean(), out.device);"
pip install -U tqdm gradio matplotlib sentencepiece protobuf transformers tokenizers huggingface_hub accelerate


# 3. Install decord
cd
git clone --recursive https://github.com/zhanwenchen/decord
cd decord
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
make -j
# Install decord Python bindings
conda activate vtom
cd ../python
python setup.py install --user
# Test decord installation
cd examples
# Run all the Jupyter Notebooks under the vtom environment
# You need to install ALSA (`sudo apt install libasound2-dev` and then `pip install simpleaudio opencv-python-headless`)

Additionally, install [FlashAttention](https://github.com/HazyResearch/flash-attention) for training,
```shell
pip install ninja einops

cd
git clone --single-branch --branch v2.3.3 git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=4 python setup.py install # Cannot use pip install . on this repo. Also need to specify


# Move data here


# Finally, run training.
conda activate vtom
cd ~/vtom
# OMP_NUM_THREADS = nb_cpu_threads / nproc_per_node: https://github.com/pytorch/pytorch/issues/22260#issuecomment-508196387
export NPROC_PER_NODE=4
export OMP_NUM_THREADS=$(($(nproc) / ${NPROC_PER_NODE}))
PYTHONPATH="./:$PYTHONPATH" torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port 29001 video_chatgpt/train/train_mem.py \
          --model_name_or_path ./LLaVA-Lightning-7B-v1-1 \
          --version v1 \
          --data_path video_chatgpt_training_removed.json \
          --video_folder data/clip_features_train \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./Video-ChatGPT_7B-1.1_Checkpoints \
          --num_train_epochs 3 \
          --per_device_train_batch_size 8 \
          --per_device_eval_batch_size 8 \
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


az storage blob download --account-name vtom --container-name vtom --name Zero_Shot_QA.zip --file Zero_Shot_QA.zip

az storage file download-batch \
  --account-name <account_name> \
  --source Users/video.tom/vtom/Video-ChatGPT_7B-1.1_Checkpoints_old/checkpoint-9000 \
  --account-key <your_az_account_key> \
  -s <your_az_code> \
  -d .

# Run Eval on ActivityNet QA:
conda activate vtom
cd ~/vtom
PYTHONPATH="./:$PYTHONPATH" python video_chatgpt/eval/run_inference_activitynet_qa.py \
# Generate video features and predictions
PYTHONPATH="./:$PYTHONPATH" python video_chatgpt/eval/run_inference_activitynet_qa.py \
    --model-name Video-ChatGPT_7B-1.1_Checkpoints_old/checkpoint-9000  \
    --video_dir data/ActivityNet/all_test \
    --gt_file_question data/ActivityNet/Zero_Shot_QA/test_q.json \
    --gt_file_answers data/ActivityNet/Zero_Shot_QA/test_a.json \
    --output_dir data/ActivityNet/output \
    --output_name video_chatgpt_activitynet_qa_preds
    # --projection_path Video-ChatGPT_7B-1.1_Checkpoints_old/checkpoint-9000

PYTHONPATH="./:$PYTHONPATH" python quantitative_evaluation/evaluate_activitynet_qa.py \
    --pred_path data/ActivityNet/output/video_chatgpt_activitynet_qa_preds.json \
    --output_dir data/ActivityNet/output \
    --output_json data/ActivityNet/output/video_chatgpt_activitynet_qa_results.json \
    --api_key <my_api_key> \
    --num_tasks 1

# Finally, run training on Social-IQ 2.0.
conda activate vtom
cd ~/vtom
# OMP_NUM_THREADS = nb_cpu_threads / nproc_per_node: https://github.com/pytorch/pytorch/issues/22260#issuecomment-508196387
export CUDA_VISIBLE_DEVICES="0,1"
export NPROC_PER_NODE=$(echo ${CUDA_VISIBLE_DEVICES} | tr -cd , | wc -c); ((NUM_GPUS++))
export OMP_NUM_THREADS=$(($(nproc) / ${NPROC_PER_NODE}))
PYTHONPATH="./:$PYTHONPATH" torchrun --nproc_per_node=${NPROC_PER_NODE} --master_port 29001 video_chatgpt/train/train_mem.py \
          --model_name_or_path ./LLaVA-Lightning-7B-v1-1 \
          --version v1 \
          --data_path data/siq2/qa/qa_train_instruction_removed.json \
          --video_folder data/siq2/video_features \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./vtom_checkpoints_1 \
          --num_train_epochs 3 \
          --per_device_train_batch_size 8 \
          --per_device_eval_batch_size 8 \
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


# Val
# Make qa_val_instruction.json
python scripts/convert_instruction_json_to_training_format_siq2.py \
        --input_json_file data/siq2/qa/qa_val.json \
        --output_json_file data/siq2/qa/qa_val_instruction.json


export NPROC_PER_NODE=4
export OMP_NUM_THREADS=$(($(nproc) / ${NPROC_PER_NODE}))
PYTHONPATH="./:$PYTHONPATH" python video_chatgpt/eval/run_inference_siq2_qa.py \
    --model-name ${HOME}/vtom_checkpoints_1  \
    --video_dir data/siq2/video \
    --gt_file_qa data/siq2/qa/qa_val_instruction_removed.json \
    --output_dir data/siq2/output \
    --output_name video_chatgpt_siq2_qa_preds_val

pip install "openai<1.0.0"

PYTHONPATH="./:$PYTHONPATH" python quantitative_evaluation/evaluate_siq2_qa.py \
    --pred_path data/siq2/output/video_chatgpt_siq2_qa_preds_val.json \
    --output_dir data/siq2/output \
    --output_json data/siq2/output/video_chatgpt_siq2_qa_results.json \
    --api_key <my_api_key> \
    --num_tasks 1


# Test

# Make qa_test_instruction.json
# First, convert qa_test.json to valid JSON (add , to every line except for the last line. Then enclose everything in [])
python scripts/convert_instruction_json_to_training_format_siq2.py \
        --input_json_file data/siq2/qa/qa_test.json \
        --output_json_file data/siq2/qa/qa_test_instruction.json

# python scripts/remove_nonexistent_data.py

export NPROC_PER_NODE=4
export OMP_NUM_THREADS=$(($(nproc) / ${NPROC_PER_NODE}))
PYTHONPATH="./:$PYTHONPATH" python video_chatgpt/eval/run_inference_siq2_qa.py \
    --model-name ${HOME}/vtom_checkpoints_1  \
    --video_dir data/siq2/video \
    --gt_file_qa data/siq2/qa/qa_test_instruction_removed.json \
    --output_dir data/siq2/output \
    --output_name video_chatgpt_siq2_qa_preds_test

PYTHONPATH="./:$PYTHONPATH" python quantitative_evaluation/evaluate_siq2_qa.py \
    --pred_path data/siq2/output/video_chatgpt_siq2_qa_preds_test.json \
    --output_dir data/siq2/output \
    --output_json data/siq2/output/video_chatgpt_siq2_qa_results_test.json \
    --api_key <my_api_key> \
    --num_tasks 1


# New task: Create new task
# 1. Create the merged qa_train and qa_test out of removed.
# Train: Out of a total of 6159 QA pairs, 565 are unavailable, leaving 5594.
python scripts/remove_nonexistent_data.py \
    --qa_json_fpath_in data/siq2/qa/qa_train.json \
    --qa_json_fpath_removed_out data/siq2/qa/qa_train_removed.json \
    --qa_json_fpath_nonexistent_out data/siq2/qa/qa_train_nonexistent.json \
    --video_features_dir data/siq2/video_features


python scripts/create_tom_localization_qa.py \
    --qa_json_fpath_in data/siq2/qa/qa_train_removed.json \
    --qa_json_fpath_out data/siq2/qa/qa_train_removed_merged_n3.json \
    --n 3

# Train: Out of a total of 943 QA pairs, 67 are unavailable, leaving 876.
python scripts/remove_nonexistent_data.py \
    --qa_json_fpath_in data/siq2/qa/qa_val.json \
    --qa_json_fpath_removed_out data/siq2/qa/qa_val_removed.json \
    --qa_json_fpath_nonexistent_out data/siq2/qa/qa_val_nonexistent.json \
    --video_features_dir data/siq2/video_features

python scripts/create_tom_localization_qa.py \
    --qa_json_fpath_in data/siq2/qa/qa_val_removed.json \
    --qa_json_fpath_out data/siq2/qa/qa_val_removed_merged_n3.json \
    --n 3

python scripts/merge_videos_siq2.py \
    --video_dirpath_in data/siq2/video \
    --video_dirpath_out data/siq2/video_merged_n3 \
    --qa_path data/siq2/qa/qa_train_removed_merged_n3.json

python scripts/save_spatio_temporal_clip_features.py \
    --ts_by_videol_fpath data/siq2/qa/ts_by_video_qa_train_removed_merged_n3.json \
    --qa_path data/siq2/qa/qa_train_removed_merged_n3.json \
    --video_dir_path data/siq2/video_merged_n3 \
    --clip_feat_path data/clip_features_train_merged_n3