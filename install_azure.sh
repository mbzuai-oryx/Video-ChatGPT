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
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
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

# Finally, run training.
conda activate vtom
cd ~/vtom
# OMP_NUM_THREADS = nb_cpu_threads / nproc_per_node: https://github.com/pytorch/pytorch/issues/22260#issuecomment-508196387
export NPROC_PER_NODE=4
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
