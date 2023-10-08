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
conda create --name=vtom python=3.11 # At the time of writing, Python 3.12 has just been released but the ecosystem has not been completed yet, so we stick with 3.11.
conda activate vtom

cd
git clone git@github.com:zhanwenchen/vtom.git || git clone https://github.com/zhanwenchen/vtom.git
cd vtom
```


# 1. Ubuntu 22.04.3 with nvidia-driver-535 (sub-version 535.113.01). Could possibly work for other Linux distributions.
```shell
# 1. Install Pytorch with CUDA 12.1 from conda. You can also build from source.
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c huggingface transformers tokenizers huggingface_hub
conda install -c defaults -c conda-forge accelerate # Avoid having conda-forge dependencies supercede main.
# TODO: sentencepiece can be installed from source: https://github.com/google/sentencepiece#build-and-install-sentencepiece-command-line-tools-from-c-source
pip install tqdm gradio sentencepiece protobuf


# 2. Install decord GPU
# 2.1 Build ffmpeg with NVIDIA Video Codec SDK 12.1: https://docs.nvidia.com/video-technologies/video-codec-sdk/12.1/ffmpeg-with-nvidia-gpu/index.html
cd
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers && sudo make install && cd
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
sudo apt install build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev
./configure --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --disable-static --enable-shared
make -j ${nproc}
sudo make install
sudo sh -c "echo '/usr/local/lib' >> /etc/ld.so.conf"
sudo ldconfig
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -preset p6 -tune hq -b:v 5M -bufsize 5M -maxrate 10M -qmin 0 -g 250 -bf 3 -b_ref_mode middle -temporal-aq 1 -rc-lookahead 20 -i_qfactor 0.75 -b_qfactor 1.1 output.mp4
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -preset p6 -tune ll -b:v 5M -bufsize 5M -maxrate 10M -qmin 0 -g 250 -bf 3 -b_ref_mode middle -temporal-aq 1 -rc-lookahead 20 -i_qfactor 0.75 -b_qfactor 1.1 output.mp4

# Test ffmpeg installation
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -resize 1280x720 -i input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -crop 16x16x32x32 -i input.mp4 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -vf scale_cuda=1280:720 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -vf scale_npp=1280:720 -c:a copy -c:v h264_nvenc -b:v 5M output.mp4
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -vf scale_npp=1920:1080 -c:a copy -c:v h264_nvenc -b:v 5M output1.mp4 -vf scale_npp=1280:720 -c:a copy -c:v h264_nvenc -b:v 8M output2.mp4
ffmpeg -y -vsync 0 -c:v h264_cuvid -i input.mp4 output.yuv
ffmpeg -y -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 -c:a copy -c:v h264_nvenc -preset p6 -tune ll -b:v 5M -bufsize 5M -maxrate 10M -qmin 0 -g 250 -bf 3 -b_ref_mode middle -temporal-aq 1 -rc-lookahead 20 -i_qfactor 0.75 -b_qfactor 1.1 output.mp4

# 2.2 Install decord
cd
git clone --recursive https://github.com/dmlc/decord
cd decord
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
make -j $(nproc)
# Install decord Python bindings
conda activate vtom
cd ../python
pip install .
# Test decord installation
cd examples
# Run all the Jupyter Notebooks under the vtom environment
```

Additionally, install [FlashAttention](https://github.com/HazyResearch/flash-attention) for training,
```shell
pip install ninja

git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
git checkout v1.0.7
python setup.py install
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