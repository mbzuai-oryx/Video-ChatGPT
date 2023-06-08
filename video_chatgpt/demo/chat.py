import time
import torch
import gradio as gr
from video_chatgpt.utils import (build_logger)
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.video_conversation import load_video
from video_chatgpt.model.utils import KeywordsStoppingCriteria
import logging
from video_chatgpt.constants import *


logging.basicConfig(level=logging.WARNING)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "Video-ChatGPT"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


class Chat:
    def __init__(self, model_name, conv_mode, tokenizer, image_processor, vision_tower, model, replace_token):
        self.model_name = model_name
        self.conv_mode = conv_mode
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.vision_tower = vision_tower
        self.model = model
        self.replace_token = replace_token

    def upload_video(self, video, img_list):
        if isinstance(video, str):  # is a path
            frames = load_video(video)
            image_tensor = self.image_processor.preprocess(frames, return_tensors='pt')['pixel_values']
            img_list.append(image_tensor)
        else:
            raise NotImplementedError
        msg = "Received."
        return msg

    def get_spatio_temporal_features_torch(self, features):
        t, s, c = features.shape
        temporal_tokens = torch.mean(features, dim=1)
        padding_size = 100 - t
        if padding_size > 0:
            temporal_tokens = torch.cat((temporal_tokens, torch.zeros(padding_size, c, device=features.device)), dim=0)

        spatial_tokens = torch.mean(features, dim=0)
        concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

        return concat_tokens

    def answer(self, state, img_list, temperature, max_new_tokens, first_run):
        if state.skip_next:
            # This generates call is skipped due to invalid inputs
            yield (state, state.to_gradio_chatbot(), img_list, first_run) + (no_change_btn,) * 5
            return

        if first_run:
            conv_mode = self.conv_mode
            new_state = conv_templates[conv_mode].copy()
            new_state.append_message(new_state.roles[0], state.messages[-2][1])
            new_state.append_message(new_state.roles[1], None)
            state = new_state
            first_run = False

        # Construct prompt
        prompt = state.get_prompt()
        prompt = prompt.replace(DEFAULT_VIDEO_TOKEN, self.replace_token, 1)

        inputs = self.tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = state.sep if state.sep_style != SeparatorStyle.TWO else state.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # Uncomment this for debugging purposes
        # pload = {
        #     "model": self.model_name,
        #     "prompt": prompt,
        #     "temperature": float(temperature),
        #     "max_new_tokens": min(int(max_new_tokens), 1536),
        #     "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        # }
        # logger.info(f"==== request ====\n{pload}")

        state.messages[-1][-1] = ""
        yield (state, state.to_gradio_chatbot(), img_list, first_run) + (disable_btn,) * 5

        image_tensor = img_list[0]
        # Generate video spatio-temporal features
        image_tensor = image_tensor.half().cuda()
        with torch.no_grad():
            image_forward_outs = self.vision_tower(image_tensor, output_hidden_states=True)
            select_hidden_state_layer = -2  # Same as used in LLaVA
            select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
            frame_features = select_hidden_state[:, 1:]
        video_spatio_temporal_features = self.get_spatio_temporal_features_torch(frame_features)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
                do_sample=True,
                temperature=float(temperature),
                max_new_tokens=min(int(max_new_tokens), 1536),
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        output = post_process_code(outputs)
        for character in output:
            state.messages[-1][-1] += character
            time.sleep(0.01)
            yield (state, state.to_gradio_chatbot(), img_list, first_run) + (enable_btn,) * 5
        # state.messages[-1][-1] = state.messages[-1][-1][:-1]
        logger.info(f"{output}")
        yield (state, state.to_gradio_chatbot(), img_list, first_run) + (enable_btn,) * 5
