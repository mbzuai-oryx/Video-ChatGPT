# Video Theory of Mind :movie_camera: :speech_balloon:

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Video Theory of Mind">
</p>

### Video Theory of Mind

#### [Zhanwen Chen](https://www.zhanwenchen.com), [Tianchun Wang](https://ist.psu.edu/directory/tkw5356), and [Yizhou Wang](https://wyzjack.github.io/)

#### **University of Virginia, The Pennsylvanian State University, Northeastern University**

---
## Installation :wrench:

We recommend setting up a conda environment for the project:
```shell
conda create --name=vtom python=3.11
conda activate vtom

cd
git clone git@github.com:zhanwenchen/vtom.git || git clone https://github.com/zhanwenchen/vtom.git
cd vtom


export PYTHONPATH="./:$PYTHONPATH"
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
PYTHONPATH="./:$PYTHONPATH" python video_chatgpt/demo/video_demo.py --model-name /home/zhanwen/Video-ChatGPT/LLaVA-Lightning-7B-v1-1 --projection_path /home/zhanwen/Video-ChatGPT/video_chatgpt-7B.bin

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