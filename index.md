
## Quantitative Evaluation Framework for Video-based Conversational Models

This page provides a detailed walkthrough of quantitative benchmarking framework for Video Conversational Models proposed in [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT).
The framework enables an in-depth evaluation of video-based conversational models through two types of assessments:

1. Video-based Generative Performance Benchmarking
2. Zero-Shot Question-Answer Evaluation

---

## Video-based Generative Performance Benchmarking

Our framework introduces a benchmark designed to assess the text generation performance of video-based conversational models. We leverage a test set of 500 samples curated from the ActivityNet-200 videos for this purpose.

You can download the videos from [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EatOpE7j68tLm2XAd0u6b8ABGGdVAwLMN6rqlDGM_DwhVA?e=90WIuW) and 
corresponding human-generated detailed descriptions from [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EVheQNh9Y4tEv38kC19522kBYx5IkCxPTVTfmFFGERrezA?e=Y70d2q).

Our benchmarks cover five key aspects:

1. Correctness of Information
2. Detailed Orientation
3. Contextual Understanding
4. Temporal Understanding
5. Consistency


| **Evaluation Aspect**      | **Video Chat** | **LLaMA Adapter** | **Video LLaMA** | **Video-ChatGPT** |
|:---------------------------|:--------------:|:-----------------:|:--------------:|:-----------------:|
| Correctness of Information |      2.23      |       2.03        |      1.96      |       **2.40**        |
| Detail Orientation         |      2.50      |       2.32        |      2.18      |       **2.52**        |
| Contextual Understanding   |      2.53      |       2.30        |      2.16      |       **2.62**        |
| Temporal Understanding     |      1.94      |       **1.98**        |      1.82      |       **1.98**        |
| Consistency                |      2.24      |       2.15        |      1.79      |       **2.37**        |

&nbsp;

We generate task-specific question-answers by querying the GPT-3.5-Turbo model using the human-generated detailed video descriptions. The generated question-answer pairs are available for download [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EoS-mdm-KchDqCVbGv8v-9IB_ZZNXtcYAHtyvI06PqbF_A?e=1sNbaa).

Follow the steps below to perform the quantitative benchmarking:

**Step 1:** Run the inference using the provided question-answer pairs for each criteria.

```shell
python video_chatgpt/eval/run_inference_benchmark_general.py \
    --video_dir <path-to-directory-containing-videos> \
    --gt_file <ground-truth-file-containing-question-answer-pairs> \
    --output_dir <output-dir-path> \
    --output_name <output-file-name> \
    --model-name <path-to-LLaVA-Lightening-7B-v1-1> \
    --projection_path <path-to-Video-ChatGPT-weights>
```
- Note that the question-answer pairs (gt_file) are the same for correctness, detailed orientation and Contextual understanding.

- For temporal understanding and consistency, separate question-answer pairs are provided.
  
**Step 2:** Execute the corresponding evaluation script to perform benchmarking.

For example, for correctness criteria:
```shell
python quantitative_evaluation/evaluate_benchmark_1_correctness.py \
    --pred_path <path-to-prediction-file-generated-using-inference-script> \
    --output_dir <output-directory-path> \
    --output_json <path-to-save-annotation-final-combined-json-file> \
    --api_key <openai-api-key-to-access-GPT3.5-Turbo-model>
```

For evaluation on all 5 criteria, you can use:
```shell
bash quantitative_evaluation/evaluate_benchmark.sh
```

Note: To further understand how the question-answer annotations are prepared for the benchmarking, refer to: [benchmark_dataset_generation](https://github.com/mbzuai-oryx/Video-ChatGPT/tree/main/quantitative_evaluation/benchmark_dataset_generation).

---
## Zero-Shot Question-Answer Evaluation

Our framework facilitates zero-shot evaluation on five standard open-ended question-answer datasets: MSRVTT, MSVD, TGIF, and ActivityNet-QA. For the sake of brevity, we present the evaluation method on ActivityNet-QA. The evaluation protocol remains the same for all datasets, except for some dataset-specific changes related to videos and annotations.


| **Model**     | **MSVD-QA** |  | **MSRVTT-QA** |  | **TGIF-QA** |  | **Activity Net-QA** |  |
|:--------------| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|               | **Accuracy** | **Score** | **Accuracy** | **Score** | **Accuracy** | **Score** | **Accuracy** | **Score** |
| FrozenBiLM    | 32.2 | -- | 16.8 | -- | 41.0 | -- | 24.7 | -- |
| Video Chat    | 56.3 | 2.8 | 45.0 | 2.5 | 34.4 | 2.3 | 26.5 | 2.2 |
| LLaMA Adapter | 54.9 | 3.1 | 43.8 | 2.7 | - | - | 34.2 | 2.7 |
| Video LLaMA   | 51.6 | 2.5 | 29.6 | 1.8 | - | - | 12.4 | 1.1 |
| Video-ChatGPT | **64.9** | **3.3** | **49.3** | **2.8** | **51.4** | **3.0** | **35.2** | **2.7** |

&nbsp;

Follow these steps to conduct the evaluation:

**Step 1:** Run the inference. You'll need the following:

a) Videos: Download the videos for ActivityNet-QA from [here](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/ESa302OCJMNHsMk7wuBbQc8BZH5CqlcdCWiSpXynQZDfAQ?e=CrOPbm).

b) Question and answer annotations: You can obtain these from the official [GitHub repository](https://github.com/MILVLG/activitynet-qa/tree/master/dataset), or download from [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/El1SR1Mri2NLgptt4jTOy1wBJkGyzXDKGvsWFLxvdbpKPw?e=vxtpNu).

Run the command:

```shell
python video_chatgpt/eval/run_inference_activitynet_qa.py \
    --video_dir <path-to-video-dir> \
    --gt_file_question <test_q.json> \
    --gt_file_answers <test_a.json> \
    --output_dir <path-to-out-dir> \
    --output_name video_chatgpt_activitynet_qa_preds \
    --projection_path <path-to-video-chat-gpt-checkpoint>
```
This will generate a JSON file containing the model's predicted responses.

**Step 2:** Evaluate the predicted responses. The evaluation process computes the accuracy and assigns a score on a scale of 1-5. This step requires the predictions from step-1, question-answer pair annotations, and an OpenAI API key.

Run the command:

```shell
python quantitative_evaluation/evaluate_activitynet_qa.py \
    --pred_path <video_chatgpt_activitynet_qa_preds> \
    --output_dir <path-to-out-dir> \
    --output_json <video_chatgpt_activitynet_qa_results> \
    --api_key <your-openai-api_key> \
    --num_tasks 1
```

## Citation

```bibtex
    @article{Maaz2023VideoChatGPT,
        title={Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models},
        author={Muhammad Maaz, Hanoona Rasheed, Salman Khan and Fahad Khan},
        journal={ArXiv 2306.05424},
        year={2023}
    }
```

---
[<img src="images/IVAL_logo.png" width="200" height="100">](https://www.ival-mbzuai.com)
[<img src="images/Oryx_logo.png" width="100" height="100">](https://github.com/mbzuai-oryx)
[<img src="images/video_chatgpt_logo.png" width="300" height="100">](https://github.com/mbzuai-oryx/Video-ChatGPT)
[<img src="images/MBZUAI_logo.png" width="360" height="85">](https://mbzuai.ac.ae)