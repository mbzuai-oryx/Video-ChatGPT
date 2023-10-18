# Video Theory of Mind :movie_camera: :speech_balloon:

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Video Theory of Mind">
</p>

### Video Theory of Mind

#### [Zhanwen Chen](https://www.zhanwenchen.com), [Tianchun Wang](https://ist.psu.edu/directory/tkw5356), and [Yizhou Wang](https://wyzjack.github.io/)

#### **University of Virginia, The Pennsylvanian State University, Northeastern University**

---
## Installation :wrench:

### 1. Common Installation Steps for All Systems
We recommend setting up a conda environment for the project:
```shell
conda create -n vtom python=3.10 intelpython3_full mkl pip numpy mkl-dpcpp mkl-include intel-openmp intel-fortran-rt dpcpp-cpp-rt ninja astunparse psutil pyyaml requests setuptools typing-extensions sympy filelock networkx fsspec packaging -c intel # At the time of writing, Python 3.12 has just been released but the ecosystem has not been completed yet, so we stick with 3.11.
conda activate vtom

cd
git clone git@github.com:zhanwenchen/vtom.git || git clone https://github.com/zhanwenchen/vtom.git
cd vtom
```


# 1. Ubuntu 22.04.3 with nvidia-driver-535 (sub-version 535.113.01). Could possibly work for other Linux distributions.
```shell
# # 1. Install decord GPU
# Bug: error: ‘AVBSFContext’ was not declared in this scope; did you mean ‘AVIOContext’? ffmpeg 5.0 issue?
# Cause: decord lacks ffmpeg5 support.
# Solution: apply patch at https://github.com/dmlc/decord/issues/186#issuecomment-1171882325
# 1.1 Build ffmpeg with NVIDIA Video Codec SDK 12.1: https://docs.nvidia.com/video-technologies/video-codec-sdk/12.1/ffmpeg-with-nvidia-gpu/index.html

# 1. Install nvcodec and ffmpeg5 for PyTorch and decord
# 1a. Install nvcodec headers into your Conda environment
cd
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
vim Makefile   # change the first line to PREFIX = ${CONDA_PREFIX}
make install

# 1b. Install ffmpeg5 with NVIDIA Video Codec SDK support
cd
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
sudo apt install yasm libx264-dev libgnutls28-dev
export MY_SM=86
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
src="https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
ffmpeg -hide_banner -encoders | grep 264
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


# 2. Build Pytorch with CUDA 12.1 from source to use custom ffmpeg5 with nvcodec support
# 2.1 Install system cuSparseLt and NCCL
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install libcusparselt0 libcusparselt-dev libnccl2 libnccl-dev

# 2.2 Install conda deps
conda install -c pytorch magma-cuda121 # TODO: include magma-cuda122 building in the future
conda install -c intel -c defaults cmake ninja astunparse expecttest hypothesis psutil pyyaml requests setuptools typing-extensions sympy filelock networkx jinja2 fsspec packaging
conda install -c defaults -c conda-forge jinja2 types-dataclasses optree # NOTE: jinja2 needs to be >= 3.1.2, so at the time of writing, cannot be from -c intel.


# 2.3 Install PyTorch from source
cd && git clone --recursive --single-branch --branch v2.1.0 https://github.com/pytorch/pytorch.git
cd pytorch
# 1. sync submodules
git submodule sync
git submodule update --init --recursive
conda activate vtom
export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export TORCH_CUDA_ARCH_LIST="8.6"
export USE_FFMPEG=1
export USE_TBB=1
export USE_SYSTEM_TBB=1
export USE_SYSTEM_NCCL=1
# TODO: ONNX_USE_PROTOBUF_SHARED_LIBS
# TODO: XNNPACK enabled shared
python setup.py clean && python setup.py develop > install_pytorch.log 2>&1
echo "DONE building pytorch"
pip install tqdm gradio matplotlib sentencepiece protobuf transformers tokenizers huggingface_hub accelerate


# 3. Install decord
cd
git clone --recursive https://github.com/zhanwenchen/decord
cd decord
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cd .. && rm -rf build && mkdir build && cd build
make -j
# Install decord Python bindings
conda activate vtom
cd ../python
python setup.py install --user
# Test decord installation
cd examples
# Run all the Jupyter Notebooks under the vtom environment
# You need to install ALSA (`sudo apt install libasound2-dev` and then `pip install simpleaudio opencv-python-headless`)
```

Additionally, install [FlashAttention](https://github.com/HazyResearch/flash-attention) for training,
```shell
pip install ninja einops

cd
git clone --single-branch --branch v2.3.2 git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=12 python setup.py install # Cannot use pip install . on this repo. Also need to specify sm_86 because it is not included by default. 16 jobs need 96GB RAM.
```

---

## Running Demo Offline :cd:

To run the demo offline, please refer to the instructions in [offline_demo.md](docs/offline_demo.md).
```shell
PYTHONPATH="./:$PYTHONPATH" python video_chatgpt/demo/video_demo.py --model-name ./LLaVA-Lightning-7B-v1-1 --projection_path ./video_chatgpt-7B.bin
```
---

## Training :train:

For training instructions, check out [train_video_chatgpt.md](docs/train_video_chatgpt.md).
```shell
python scripts/convert_instruction_json_to_training_format.py \
        --input_json_file ./data/VideoInstruct_Dataset.json \
        --output_json_file video_chatgpt_training.json
# Total annotations retained: 100010


python scripts/save_spatio_temporal_clip_features.py \
        --video_dir_path ./data/videos_train \
        --clip_feat_path ./data/clip_features_train


unset LD_PRELOAD
export IOMP5_PATH=${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${IOMP5_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
PYTHONPATH="./:$PYTHONPATH" torchrun --nproc_per_node=2 --master_port 29001 video_chatgpt/train/train_mem.py \
          --model_name_or_path ./LLaVA-7B-Lightening-v1-1-delta \
          --version v1 \
          --data_path video_chatgpt_training.json \
          --video_folder data/clip_features_train \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir ./Video-ChatGPT_7B-1.1_Checkpoints \
          --num_train_epochs 3 \
          --per_device_train_batch_size 1 \
          --per_device_eval_batch_size 1 \
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

---


## Acknowledgements :pray:

+ [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT): A great attempt towards open and efficient LLMs!
+ [Microsoft Research - Accelerate Foundation Models Research Grant](https://www.microsoft.com/en-us/research/collaboration/accelerating-foundation-models-research/phase-ii/)

## License :scroll:
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.


Looking forward to your feedback, contributions, and stars! :star2:
Please raise any issues or questions [here](https://github.com/mbzuai-oryx/Video-ChatGPT/issues).


---
# Notes

-2 layer features, not last layer of clip. 100 (temporal) x 256 (spatial) x 1024 (semantic?). Pool over 100, pool over 256. Linear layer [1024, 512]. Project features to tokens. Both text and video tokens are input into llms, which take tokens anyway.

Finetuning - 3-4 with their dataset with 8xA100. Init model - LLaVA pretrained 7B. ToM datasets - Social-IQ 1.0/2.0/TinySocial. Need to extract CLIP features with pico files.

Social-IQ 1.0: 1, 250 videos, 7500 questions, 30, 000 correct
answers and 22, 500 incorrect answers

Social-IQ 2.0:

TinySocial: 50.

Training code can be complicated. Need to modify dataloader. LazySupervisedDataTraining need to be modified. Can ask Yizhou. OpenAI Clip tokenizer, etc.

JSON file - training.

New task - video frame retrieval based on questions. Novelty not as strong. Literature - next prediction or multiple frames? Video frame prediction - next/segment, claim locate frames based on questions. Salient frame theory of mind change/impact. How to correlate between theory of mind change and impact of. Relational retrival instead of object retrieval. Object-level retrieval.

"Get me the frame of the man walking into the store?"
vs
<!-- "Get the the frame where the man became sad." -->
"Get the the frame where the man realize his wife lied."

Maybe also add temporal gnn to learn temporal changes in mental states.

TODOs for next week:
-[] Apply for MS Azure access and ask if they can use it.
-[] Run eval on Social-IQ (need to modify dataloader)
-[] Can prioritize reproducing finetuning first
-[] Do a survey on video+llm. Video-LLM, VideoChat, Video-ChatGPT.
-[] Define novel task (convert vqa to video frame retrieval)
-[] Run finetuning on on stuff.