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

# 0. Install cuSparseLt
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo apt install ./cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install libcusparselt0 libcusparselt-dev
```
# 1. Ubuntu 22.04.3 with nvidia-driver-535 (sub-version 535.113.01). Could possibly work for other Linux distributions.
```shell
# # 1. Install decord GPU
# Bug: error: ‘AVBSFContext’ was not declared in this scope; did you mean ‘AVIOContext’? ffmpeg 5.0 issue?
# Cause: decord lacks ffmpeg5 support.
# Solution: apply patch at https://github.com/dmlc/decord/issues/186#issuecomment-1171882325
# 1.1 Build ffmpeg with NVIDIA Video Codec SDK 12.1: https://docs.nvidia.com/video-technologies/video-codec-sdk/12.1/ffmpeg-with-nvidia-gpu/index.html
cd
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
vim Makefile   # change the first line to PREFIX = ${CONDA_PREFIX}
make install
cd
# sudo make install && cd
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
# sudo apt install build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev
# Under the same environment
# conda install nasm
# ./configure --prefix=${CONDA_PREFIX} --enable-cuda-nvcc --enable-cuvid --enable-nvenc --enable-nvdec --enable-nonfree --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 -gencode arch=compute_86,code=sm_86 -O2 # Using your system cuda. # https://pytorch.org/audio/stable/build.ffmpeg.html
# ./configure --prefix=${CONDA_PREFIX} --enable-cuda-nvcc --enable-cuvid --enable-nvenc --enable-nonfree --enable-libnpp --extra-cflags=-I${CONDA_PREFIX}/include --extra-ldflags=-L${CONDA_PREFIX}/lib # Using your Conda environment libs. May need to install libnpp separately
# ./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared
sudo apt install yasm libx264-dev libgnutls28-dev
# ccap=86
./configure \
  --extra-cflags='-I/usr/local/cuda/include' \
  --extra-ldflags='-L/usr/local/cuda/lib64' \
  --nvccflags="-gencode arch=compute_86,code=sm_86 -O2" \
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

# Conda installation:
# make -j
# make install # no sudo or ld_config

# Confirm that you are indeed using the Conda ffmpeg
which ffmpeg # The output should include your conda environment

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

# ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -preset p6 -tune hq -b:v 5M -bufsize 5M -maxrate 10M -qmin 0 -g 250 -bf 3 -b_ref_mode middle -temporal-aq 1 -rc-lookahead 20 -i_qfactor 0.75 -b_qfactor 1.1 output.mp4
# ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -preset p6 -tune ll -b:v 5M -bufsize 5M -maxrate 10M -qmin 0 -g 250 -bf 3 -b_ref_mode middle -temporal-aq 1 -rc-lookahead 20 -i_qfactor 0.75 -b_qfactor 1.1 output.mp4

# # Test ffmpeg installation
# ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
# ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -resize 1280x720 -i input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
# ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -crop 16x16x32x32 -i input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
# ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -vf scale_cuda=1280:720 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
# ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -vf scale_npp=1280:720 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
# ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -vf scale_npp=1920:1080 -c:a copy -c:v h264_nvenc -b:v 5M output1.mp4 -vf scale_npp=1280:720 -c:a copy -c:v h264_nvenc -b:v 8M output2.mp4
# ffmpeg -y -vsync 0 -c:v h264_cuvid -i input.mp4 output.yuv
# ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -preset p6 -tune ll -b:v 5M -bufsize 5M -maxrate 10M -qmin 0 -g 250 -bf 3 -b_ref_mode middle -temporal-aq 1 -rc-lookahead 20 -i_qfactor 0.75 -b_qfactor 1.1 output.mp4




# 2. Install Pytorch with CUDA 12.1 from conda. You can also build from source.
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c intel
# conda install -c huggingface transformers tokenizers huggingface_hub
# conda install -c defaults -c conda-forge accelerate # Avoid having conda-forge dependencies supercede main.
# TODO: sentencepiece can be installed from source: https://github.com/google/sentencepiece#build-and-install-sentencepiece-command-line-tools-from-c-source

# From source:
# conda install cmake ninja

# Install Intel LLVM:
# git clone https://github.com/intel/llvm -b sycl && cd llvm
# git checkout 7d03bdadc0428db2c156a5823384bd0e2e510523
# https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md
# python llvm/buildbot/configure.py --cuda
# python llvm/buildbot/compile.py

# Install NCCL
sudo apt install libnccl2 libnccl-dev


sudo apt install intel-basekit
https://developer.codeplay.com/products/oneapi/nvidia/download

# Install conda libs

conda install -c pytorch magma-cuda121
# conda install -c intel ninja astunparse psutil pyyaml requests setuptools typing-extensions sympy filelock networkx jinja2 fsspec packaging
conda install -c intel -c defaults cmake ninja astunparse expecttest hypothesis psutil pyyaml requests setuptools typing-extensions sympy filelock networkx jinja2 fsspec packaging
conda install -c defaults -c conda-forge jinja2 types-dataclasses optree

# pip install types-dataclasses optree


cd && git clone --recursive --single-branch --branch v2.1.0 https://github.com/pytorch/pytorch.git
cp -r pytorch pytorch_original
cd pytorch
# 2023.1.0 and 2023.2.0 have clang versions > 16.
# 2023.1.0 didn't have the clang version error
# TODO: try to install the codeplay shell script with 2023.1.0
# TODO: try again with 2023.2.0
# sudo sh /home/zhanwen/Downloads/oneapi-for-nvidia-gpus-2023.1.0-cuda-12.0-linux.sh # codeplay.com
# export CUDA_NVCC_FLAGS='-allow-unsupported-compiler'
# export NVCC_FLAGS='-allow-unsupported-compiler'
# export TORCH_NVCC_FLAGS='-allow-unsupported-compiler'
# . /opt/intel/oneapi/setvars.sh --include-intel-llvm


# ICPC: can maybe use that instead of icpx with sudo apt install intel-hpckit


# In /home/zhanwen/pytorch/third_party/sleef/Configure.cmake, change extended_float_type => extended_float_types
#   set(FLAGS_STRICTMATH "-fp-model strict -Qoption,cpp,--extended_float_types")
#   set(FLAGS_FASTMATH "-fp-model fast=2 -Qoption,cpp,--extended_float_types")

conda activate vtom
# export CMAKE_CXX_FLAGS="-ffp-model=precise" # Solves google benchmark json_parser NaN/Inf comparison issue with icpx
# export CMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2023.1.0/linux/bin/icpx # or just icpx # doesn't work because of python_functions_1.cpp:47:3:
#  [0m[0;1;31merror: [0m[1mcast from 'PyObject *(*)(THPCppFunction *,
# PyObject *)' (aka '_object *(*)(torch::autograd::THPCppFunction *, _object *)')
# to 'getter' (aka '_object *(*)(_object *, void *)') converts to incompatible
# function type [-Werror,-Wcast-function-type-strict]
# export CMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2023.1.0/linux/bin/icx # or just icx
# TODO: see how Intel built it with icpx and conda and replicate it.
# OpenMP_CXX_LIB_NAMES
# TODO: build shared libs including NNPack, Kineto, libprotobuf, etc.
# unset LD_PRELOAD
# export IOMP5_PATH=${CONDA_PREFIX}/lib/libiomp5.so
# export LD_PRELOAD=${IOMP5_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}
# export KMP_AFFINITY=granularity=fine,compact,1,0
# export KMP_BLOCKTIME=1
# export OpenMP_C_FLAGS="-liomp5"
# export OpenMP_C_FLAGS="-fopenmp=libiomp5"
# export OpenMP_CXX_FLAGS="-fopenmp=libiomp5"
# export OpenMP_CXX_FLAGS="-liomp5"
# export OpenMP_C_LIB_NAMES="iomp5"
# export OpenMP_CXX_LIB_NAMES="iomp5 iomp"
# export CMAKE_SHARED_LINKER_FLAGS="${IOMP5_PATH}"
# export CMAKE_EXE_LINKER_FLAGS="${IOMP5_PATH}"
# export CMAKE_CXX_STANDARD=11
export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export TORCH_CUDA_ARCH_LIST="8.6"
export USE_FFMPEG=1
export USE_TBB=1
export USE_SYSTEM_TBB=1
export USE_SYSTEM_NCCL=1
# export MAX_JOBS=32
# git submodule sync
# git submodule update --init --recursive
# export CXX=/opt/intel/oneapi/compiler/latest/linux/bin/icpx
# export CC=/opt/intel/oneapi/compiler/latest/linux/bin/icx
# 1. sync submodules
# 2. update oneapi to update tbb
# 3. try this build
cd ~/pytorch
# python setup.py build --cmake-only
# python setup.py clean && mkdir build && cp /home/zhanwen/pytorch/CMakeCache_backup.txt /home/zhanwen/pytorch/build/CMakeCache_backup.txt && python setup.py develop > install_pytorch.log 2>&1
# ccmake build  # or cmake-gui build
python setup.py clean && python setup.py develop > install_pytorch.log 2>&1
echo "DONE building pytorch"
# TODO: ONNX_USE_PROTOBUF_SHARED_LIBS
# TODO: XNNPACK enabled shared
# HAS_WMISSING_PROTOTYPES failed
# HAS_WERROR_MISSING_PROTOTYPES failed
# /usr/bin/ld: lib/libc10.so: undefined reference to `std::condition_variable::wait(std::unique_lock<std::mutex>&)@GLIBCXX_3.4.30'
# collect2: error: ld returned 1 exit status
pip install tqdm gradio sentencepiece protobuf transformers tokenizers huggingface_hub accelerate


# 3. Install decord
# conda install pytorch-cuda=12.1 -c pytorch -c nvidia
cd
git clone --recursive https://github.com/zhanwenchen/decord
cd decord
mkdir build && cd build
# cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DFFMPEG_DIR=${CONDA_PREFIX} -DUSE_CUDA=${CONDA_PREFIX}/lib -DCMAKE_BUILD_TYPE=Release
# cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DFFMPEG_DIR=${CONDA_PREFIX} -DCMAKE_BUILD_TYPE=Release
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cd .. && rm -rf build && mkdir build && cd build
# cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DFFMPEG_DIR=/home/zhanwen/ffmpeg -DCMAKE_BUILD_TYPE=Release
make -j
# Install decord Python bindings
conda activate vtom
cd ../python
python setup.py install --user
# pip install .
# Test decord installation
cd examples # You need to install ALSA (`sudo apt install libasound2-dev` and then `pip install matplotlib simpleaudio opencv-python-headless`)
# Run all the Jupyter Notebooks under the vtom environment
# -- Custom CUDA_PATH=/home/zhanwen/anaconda3/envs/vtom/lib
# -- Found CUDA_TOOLKIT_ROOT_DIR=/home/zhanwen/anaconda3/envs/vtom/lib
# -- Found CUDA_CUDA_LIBRARY=
# -- Found CUDA_CUDART_LIBRARY=CUDA_CUDART_LIBRARY-NOTFOUND
# -- Found CUDA_NVRTC_LIBRARY=CUDA_NVRTC_LIBRARY-NOTFOUND
# -- Found CUDA_CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so
# -- Found CUDA_CUBLAS_LIBRARY=CUDA_CUBLAS_LIBRARY-NOTFOUND
# -- Found CUDA_NVIDIA_ML_LIBRARY=CUDA_NVIDIA_ML_LIBRARY-NOTFOUND
# -- Found CUDA_NVCUVID_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcuvid.so

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