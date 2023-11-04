
# 1. Base system
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

# TODO: change these:
# SM=86 for 3090 TI. SM=80 for A100 80GB.
ENV SM=80
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV NUM_GPUS 8
ENV PORT 59048


ENV PATH /opt/conda/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/stubs/:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.2/compat/:/usr/local/cuda-12.2/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
# ARG CODE_VERSION=latest
# ARG BUILDPLATFORM BUILDOS BUILDARCH
ENV CONDA_OS=Linux
ENV CONDA_ARCH=x86_64
# BUILDPLATFORM: linux/amd64
# BUILDARCH: amd64, arm64, riscv64
# BUILDOS: linux # TODO: echo this on MacOS.

# Environment Variables
ENV BUILD_DIR=/tmp
ENV PYTORCH_VERSION=2.1.0

# Base System utilities.
RUN apt update && apt full-upgrade -y # Good
RUN apt install -y build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev git pkg-config # Good

# 3. Install Miniconda
# echo
# RUN CONDA_ARCH="${BUILDARCH}"; \
#     if [ "${CONDA_ARCH}" = "amd64" ]; then CONDA_ARCH=x86_64; fi; \
#     CONDA_OS="${BUILDOS}"; \
#     if [ "${CONDA_OS}" = "linux" ]; then CONDA_OS=Linux; fi; \

RUN wget -nv --show-progress --progress=bar:force:noscroll https://repo.anaconda.com/miniconda/Miniconda3-latest-${CONDA_OS}-${CONDA_ARCH}.sh -O Miniconda3-latest.sh
RUN chmod +x ./Miniconda3-latest.sh
RUN ./Miniconda3-latest.sh -b -p /opt/conda # Good
RUN conda init bash
# RUN source /root/.bashrc
RUN conda update --all -y

# 2. Install CUDA 12.2 and cuDNN 8.5.29
# 2.1 Install system cuSparseLt and NCCL
RUN apt install -y libcusparselt0 libcusparselt-dev libnccl2 libnccl-dev

# 3. Install NVCODEC
RUN echo "Start installing NVCODEC headers."
WORKDIR /tmp
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
WORKDIR /tmp/nv-codec-headers
# No need for CONDA_PREFIX since we are root in Docker.
RUN make install && echo "Finished installing NVCODEC headers."

# 3.2 Build ffmpeg
RUN apt install -y libx264-dev libgnutls28-dev
WORKDIR /tmp
RUN git clone https://git.ffmpeg.org/ffmpeg.git
WORKDIR /tmp/ffmpeg
RUN ./configure \
    --extra-cflags='-I/usr/local/cuda/include' \
    --extra-ldflags='-L/usr/local/cuda/lib64' \
    --nvccflags="-gencode arch=compute_${SM},code=sm_${SM} -O2" \
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
RUN make clean
RUN make -j
RUN make install
RUN echo '/usr/local/lib' >> /etc/ld.so.conf
RUN ldconfig


ENV VIDEO_URL="https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"

RUN /usr/local/bin/ffmpeg -hide_banner -y -vsync 0 \
    -hwaccel cuvid \
    -hwaccel_output_format cuda \
    -c:v h264_cuvid \
    -resize 360x240 \
    -i "${VIDEO_URL}" \
    -c:a copy \
    -c:v h264_nvenc \
    -b:v 5M test.mp4
RUN rm test.mp4

# 4. Build PyTorch
# 4.1 Install Intel basekit
# TODO: update intel gpg for apt sources lists.
# RUN apt install intel-basekit


# 4.2 Install conda deps
RUN echo "Installing PyTorch dependencies"
RUN conda install -c pytorch magma-cuda121 -y # TODO: include magma-cuda122 building in the future
# RUN conda install -c intel -c defaults cmake ninja astunparse expecttest hypothesis psutil pyyaml requests setuptools typing-extensions sympy filelock networkx jinja2 fsspec packaging -y
RUN conda install cmake ninja astunparse expecttest hypothesis psutil pyyaml requests setuptools typing-extensions sympy filelock networkx jinja2 fsspec packaging -y
RUN conda install -c defaults -c conda-forge jinja2 types-dataclasses optree -y # NOTE: jinja2 needs to be >= 3.1.2, so at the time of writing, cannot be from -c intel.


# # 4.3 Install PyTorch from source
WORKDIR ${BUILD_DIR}
RUN git clone --recursive --single-branch --branch v${PYTORCH_VERSION} https://github.com/pytorch/pytorch.git
# # 1. sync submodules
WORKDIR ${BUILD_DIR}/pytorch
RUN git submodule sync
RUN git submodule update --init --recursive

# RUN conda activate vtom
ENV _GLIBCXX_USE_CXX11_ABI=1
ENV CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
ENV USE_FFMPEG=1
# ENV USE_TBB=1
# ENV USE_SYSTEM_TBB=1
ENV USE_SYSTEM_NCCL=1
# RUN # TODO: ONNX_USE_PROTOBUF_SHARED_LIBS
# RUN # TODO: XNNPACK enabled shared
RUN echo "Start building pytorch"
RUN python setup.py clean && python setup.py develop > install_pytorch.log 2>&1
RUN echo "DONE building pytorch"
RUN pip install tqdm gradio matplotlib sentencepiece protobuf transformers tokenizers huggingface_hub accelerate
WORKDIR ${BUILD_DIR}
RUN rm -rf pytorch

# 4. Install decord
WORKDIR ${BUILD_DIR}
RUN git clone --recursive https://github.com/zhanwenchen/decord
WORKDIR ${BUILD_DIR}/decord
RUN mkdir build
WORKDIR ${BUILD_DIR}/decord/build
RUN cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${SM} -DCMAKE_BUILD_TYPE=Release
RUN make -j
WORKDIR ${BUILD_DIR}/decord/python
RUN python setup.py install --user
RUN rm -rf build
# Cleanup
WORKDIR ${BUILD_DIR}
RUN rm -rf decord

# # 6. Install flash-attention
RUN pip install ninja einops
WORKDIR ${BUILD_DIR}
RUN git clone --single-branch --branch v2.3.2 https://github.com/Dao-AILab/flash-attention.git
WORKDIR ${BUILD_DIR}flash-attention
RUN MAX_JOBS=4 python setup.py install

# Clean up packages:
RUN conda clean --all -y
RUN pip cache purge


# Install vtom
WORKDIR /workspace
RUN git clone https://github.com/zhanwenchen/vtom.git
WORKDIR /workspace/vtom
RUN module load awscli
# RUN pip install --user awscli
WORKDIR /workspace/vtom/data
RUN aws s3 cp s3://vtom/video_chatgpt_training_removed.json video_chatgpt_training_removed.json
RUN aws s3 cp s3://vtom/ActivityNet_Train_Video-ChatGPT_Clip-L14_Features.zip ActivityNet_Train_Video-ChatGPT_Clip-L14_Features.zip
RUN unzip ./ActivityNet_Train_Video-ChatGPT_Clip-L14_Features.zip -d ./clip_features_train
RUN rm ActivityNet_Train_Video-ChatGPT_Clip-L14_Features.zip

# Download models
WORKDIR /workspace/vtom
RUN 7z e ./LLaVA-7B-Lightening-v1-1-delta.7z

# Run training
WORKDIR /workspace/vtom
RUN PYTHONPATH="./:$PYTHONPATH" torchrun --nproc_per_node=${NUM_GPUS} --master_port ${PORT} video_chatgpt/train/train_mem.py \
    --model_name_or_path ./LLaVA-7B-Lightening-v1-1-delta \
    --version v1 \
    --data_path video_chatgpt_training_removed.json \
    --video_folder ./data/clip_features_train \
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
# RUN echo "Installing PyTorch dependencies"
# RUN conda install -c pytorch magma-cuda121 -y # TODO: include magma-cuda122 building in the future
# # RUN conda install -c intel -c defaults cmake ninja astunparse expecttest hypothesis psutil pyyaml requests setuptools typing-extensions sympy filelock networkx jinja2 fsspec packaging -y
# RUN conda install cmake ninja astunparse expecttest hypothesis psutil pyyaml requests setuptools typing-extensions sympy filelock networkx jinja2 fsspec packaging -y
# RUN conda install -c defaults -c conda-forge jinja2 types-dataclasses optree -y # NOTE: jinja2 needs to be >= 3.1.2, so at the time of writing, cannot be from -c intel.


# # # 4.3 Install PyTorch from source
# WORKDIR ${BUILD_DIR}
# RUN git clone --recursive --single-branch --branch v${PYTORCH_VERSION} https://github.com/pytorch/pytorch.git
# # # 1. sync submodules
# WORKDIR ${BUILD_DIR}/pytorch
# RUN git submodule sync
# RUN git submodule update --init --recursive

# # RUN conda activate vtom
# ENV _GLIBCXX_USE_CXX11_ABI=1
# ENV CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# ENV USE_FFMPEG=1
# # ENV USE_TBB=1
# # ENV USE_SYSTEM_TBB=1
# ENV USE_SYSTEM_NCCL=1
# # RUN # TODO: ONNX_USE_PROTOBUF_SHARED_LIBS
# # RUN # TODO: XNNPACK enabled shared
# RUN echo "Start building pytorch"
# RUN python setup.py clean && python setup.py develop > install_pytorch.log 2>&1
# RUN echo "DONE building pytorch"
# RUN pip install tqdm gradio matplotlib sentencepiece protobuf transformers tokenizers huggingface_hub accelerate
# WORKDIR ${BUILD_DIR}
# RUN rm -rf pytorch
# RUN python -c 'import torch; print(torch.cuda.is_available())'


# # 4. Install decord
# WORKDIR ${BUILD_DIR}
# RUN git clone --recursive https://github.com/zhanwenchen/decord
# WORKDIR ${BUILD_DIR}/decord
# RUN mkdir build
# WORKDIR ${BUILD_DIR}/decord/build
# RUN cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${SM} -DCMAKE_BUILD_TYPE=Release
# RUN make -j
# WORKDIR ${BUILD_DIR}/decord/python
# RUN python setup.py install --user
# # Cleanup
# WORKDIR ${BUILD_DIR}
# RUN rm -rf decord

# # # 6. Install flash-attention
# RUN pip install ninja einops
# WORKDIR ${BUILD_DIR}
# RUN git clone --single-branch --branch v2.3.2 https://github.com/Dao-AILab/flash-attention.git
# WORKDIR ${BUILD_DIR}flash-attention
# RUN MAX_JOBS=4 python setup.py install

# # Clean up packages:
# RUN conda clean --all -y
# RUN pip cache purge


# # Install vtom
# WORKDIR /workspace
# RUN git clone https://github.com/zhanwenchen/vtom.git
# WORKDIR /workspace/vtom
# RUN module load awscli
# # RUN pip install --user awscli
# WORKDIR /workspace/vtom/data
# RUN aws s3 cp s3://vtom/video_chatgpt_training_removed.json video_chatgpt_training_removed.json
# RUN aws s3 cp s3://vtom/ActivityNet_Train_Video-ChatGPT_Clip-L14_Features.zip ActivityNet_Train_Video-ChatGPT_Clip-L14_Features.zip
# RUN unzip ./ActivityNet_Train_Video-ChatGPT_Clip-L14_Features.zip -d ./clip_features_train
# RUN rm ActivityNet_Train_Video-ChatGPT_Clip-L14_Features.zip

# # Download models
# WORKDIR /workspace/vtom
# RUN 7z e ./LLaVA-7B-Lightening-v1-1-delta.7z

# # Run training
# WORKDIR /workspace/vtom
# RUN PYTHONPATH="./:$PYTHONPATH" torchrun --nproc_per_node=${NUM_GPUS} --master_port ${PORT} video_chatgpt/train/train_mem.py \
#     --model_name_or_path ./LLaVA-7B-Lightening-v1-1-delta \
#     --version v1 \
#     --data_path video_chatgpt_training_removed.json \
#     --video_folder ./data/clip_features_train \
#     --tune_mm_mlp_adapter True \
#     --mm_use_vid_start_end \
#     --bf16 True \
#     --output_dir ./Video-ChatGPT_7B-1.1_Checkpoints \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 3000 \
#     --save_total_limit 3 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 100 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess True
