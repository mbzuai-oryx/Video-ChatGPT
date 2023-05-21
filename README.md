# Oryx Video-ChatGPT
![](https://i.imgur.com/waxVImv.png)
[Muhammad Maaz](https://www.muhammadmaaz.com)* , [Hanoona Rasheed](https://www.hanoonarasheed.com/)* , [Salman Khan](https://salman-h-khan.github.io/) and [Fahad Khan](https://sites.google.com/view/fahadkhans/home).

*Equal Contribution

**Mohamed bin Zayed University of Artificial Intelligence**

<a href='#'><img src='https://img.shields.io/badge/Project-Page-Green'></a> [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/fRhm---HWJY)

<p align="center">
    <img src="images/video_chatgpt_logo.png" style="max-width: 50%; max-height: 50%;"> <br>
</p>

## :rocket: News

+ May-21 : Video-ChatGPT: demo released.

## Online Demo

:fire::fire: You can try our demo using the provided examples or by uploading your own videos [here](https://www.ival-mbzuai.com/video-chatgpt). :fire::fire:

## About Video-ChatGPT

+ Video-ChatGPT is a large vision-language model with a dedicated video-encoder and large language model (LLM), enabling video understanding and conversation about videos.
+ A simple and scalable multimodal design on top of pretrained video and language encoders that adapts only a linear projection layer for multimodal alignment.
+ Data centric focus with human assisted and semi-automatic annotation framework for high-quality video instruction data.
+ Unique multimodal (vision-language) capability combining video understanding and language generation that is comprehensively evaluated using quantitative and qualitiative comparisons on video reasoning, creativitiy, spatial and temporal understanding, and action recognition tasks.


![architechtural_overview](images/Video-ChatGPT.gif)

## Contributions
![contributions](images/contributions.png)

# Qualitative Analysis
### A Comprehensive Evaluation of Video-ChatGPT's Performance across Multiple Tasks.

## Video Reasoning Tasks
![sample1](demo_samples/sample_demo_1.png)
---
![sample2](demo_samples/sample_demo_2.png)
---
![sample3](demo_samples/sample_demo_3.png)
---
![sample4](demo_samples/sample_demo_4.png)
## Creative and Generative Tasks
![sample5](demo_samples/sample_demo_5.png)
---
![sample6](demo_samples/sample_demo_6.png)
---
![sample7](demo_samples/sample_demo_7.png)
## Spatial Understanding
![sample8](demo_samples/sample_demo_8.png)
---
![sample9](demo_samples/sample_demo_9.png)
## Video Understanding and Conversational Tasks
![sample10](demo_samples/sample_demo_10.png)
---
![sample11](demo_samples/sample_demo_11.png)
---
![sample12](demo_samples/sample_demo_12.png)
---
![sample13](demo_samples/sample_demo_13.png)
## Question Answering Tasks
![sample14](demo_samples/sample_demo_14.png)
---
![sample15](demo_samples/sample_demo_15.png)
---
![sample16](demo_samples/sample_demo_16.png)
---
![sample17](demo_samples/sample_demo_17.png)
## Temporal Understanding
![sample18](demo_samples/sample_demo_18.png)
---
![sample19](demo_samples/sample_demo_19.png)
---
![sample20](demo_samples/sample_demo_20.png)
---
![sample21](demo_samples/sample_demo_21.png)
## Action recognition
![sample22](demo_samples/sample_demo_22.png)
---
![sample23](demo_samples/sample_demo_23.png)
---
# Quantitative Analysis
### Benchmarking Video-ChatGPT's Performance with State-of-the-Art Metrics and Comparative Evaluation.
+ Develops the first quantitative video conversation evaluation framework for benchmarking the performance of video understanding generative models.
+ Evaluates Video-ChatGPT on open-ended question answering tasks using the MSRVTT and MSVD datasets.
+ Uses GPT-assisted evaluation to assess the model's capabilities, measuring the accuracy and relative score of generated predictions on a scale of 1-5 (shown on top of the bars in the bar chart below).
+ Compares the performance of VideoChat GPT with other models, including the generic video foundation model InternVideo and the video generative model Ask-Anything Video Chat.
+ Achieves state-of-the-art (SOTA) performance on both the MSRVTT and MSVD datasets, showcasing the model's exceptional performance in video understanding and question answering tasks.

<p align="center">
  <img src="images/VQA_accuracy_score_plot.png" width="450" height="300">
</p>

# Instruction Data for Model Tuning
We present the different types of data included in the instructional data prepared for model tuning, along with the methods used to enrich the ground truth annotations.
+ Data Types: The instructional data encompasses detailed descriptions, summarizations, question-answer pairs, creative/generative tasks, and conversational tasks, covering concepts from appearance, temporal relations, reasoning, and more.
+ Human Annotation Expansion: The original ground truth annotations are expanded and enriched by human annotators, who provide additional context and detail to enhance the instructional data.
+ Incorporation of context from Off-the-Shelf dense image captioning models: State-of-the-art dense captioning and prediction models are utilized to generate predictions that offer supplementary contextual information. A comprehensive method is employed to combine these predictions, leveraging some models specifically for removing noisy context from the data.
+ GPT-Assisted Postprocessing: The enriched data undergoes postprocessing using GPT models to refine and optimize the annotations, ensuring high-quality data for effective model training and improved performance.

## Instruction Data Types
![sample1](dataset_samples/dataset_sample_1.png)
---
![sample2](dataset_samples/dataset_sample_2.png)
## Data Enrichment Methods
![sample1](dataset_samples/dataset_sample_3.png)
---
![sample2](dataset_samples/dataset_sample_4.png)


## Acknowledgement

+ [LLaMA](https://github.com/facebookresearch/llama): A great attempt towards open and efficient LLMs!
+ [Vicuna](https://github.com/lm-sys/FastChat): Has the amazing language capabilities!
+ [LLaVA](https://github.com/haotian-liu/LLaVA): our architecture is inspired from LLaVA.
+ Thanks to our colleagues at MBZUAI for their essential contribution to the video annotation task, 
including Dr. Salman Khan, Dr. Fahad Khan, Abdelrahman Shaker, Shahina Kunhimon, Muhammad Uzair, Sanoojan Baliah, Malitha Gunawardhana, Akhtar Munir, 
Vishal Thengane, Vignagajan Vigneswaran, Dr. Jiale Cao, Dr. Nian Liu, Muhammad Ali, Gayal Kurrupu, Roba Al Majzoub, 
Jameel Hassan, Hanan Ghani, Dr. Muzammal Naseer, Dr. Akshay Dudhane, Dr. Jean Lahoud, Awais Rauf,
without which this project would not be possible.


If you're using Video-ChatGPT in your research or applications, please cite using this BibTeX:
```bibtex
@misc{maaz2023videochatgpt,
      title={Video-ChatGPT}, 
      author={Muhammad Maaz, Hanoona Rasheed, Salman Khan and Fahad Khan},
      journal={GitHub repository},
      year={2023},
      howpublished = {\url{https://github.com/hanoonaR/Video-ChatGPT}}}
```

## License
Non-commercial bespoke license. Please refer to license terms [here](LICENSE).

---
<div>
  <img src="images/IVAL_logo.png" width="200" height="100">
  <img src="images/Oryx_logo.png" width="100" height="100">
  <img src="images/MBZUAI_logo.png" width="360" height="85">
</div>
