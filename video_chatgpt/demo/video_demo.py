import os
import argparse
import datetime
import json
import time
import logging
from gradio import Button, Chatbot, Textbox, Row, Accordion, State, Blocks, Video, Markdown, Column, Slider, Examples, JSON
from video_chatgpt.video_conversation import (default_conversation)
from video_chatgpt.utils import (build_logger, violates_moderation, moderation_msg)
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.demo.chat import Chat
from video_chatgpt.demo.template import tos_markdown, css, title, disclaimer, Seafoam
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, LOGDIR


logging.basicConfig(level=logging.WARNING)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "Video-ChatGPT"}

no_change_btn = Button()
enable_btn = Button(interactive=True)
disable_btn = Button(interactive=False)


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params):
    logger.info(f"load_demo.. params: {url_params}")

    state = default_conversation.copy()
    return (state,
            Chatbot(visible=True),
            Textbox(visible=True),
            Button(visible=True),
            Button(visible=True),
            Row(visible=True),
            Accordion(visible=True))


def vote_last_response(state, vote_type):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "state": state.dict()
        }
        logger.info(f"{get_conv_log_filename}.")
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, imagebox):
    logger.info(f"upvote.")
    vote_last_response(state, "upvote")
    return ("", imagebox) + (disable_btn,) * 3


def downvote_last_response(state, imagebox):
    logger.info(f"downvote.")
    vote_last_response(state, "downvote")
    return ("", imagebox) + (disable_btn,) * 3


def flag_last_response(state, imagebox):
    logger.info(f"flag.")
    vote_last_response(state, "flag")
    return ("", imagebox) + (disable_btn,) * 3


def regenerate(state):
    logger.info(f"regenerate.")
    state.messages[-1][-1] = None
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(img_list):
    logger.info(f"clear_history.")
    state = default_conversation.copy()
    if img_list is not None:
        img_list = []
    return (state, state.to_gradio_chatbot(), "", Textbox(value=None, interactive=True),
            Video(value="Upload Video", interactive=True), img_list
            ) + (disable_btn,) * 5


def add_text(state, text, image, first_run):
    logger.info(f"add_text. ip:. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # Hard cut-off
    if first_run:
        text = text[:1200]  # Hard cut-off for videos
        if '<video>' not in text:
            text = text + '\n<video>'
        text = (text, image)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def upload_image(image, state):
    if image is None:
        return None, Button(interactive=True), state, None, None
    state = default_conversation.copy()
    img_list = []
    first_run = True
    llm_message = chat.upload_video(image, img_list)
    return Video(interactive=False), Button(value="Start Chatting",
                                                   interactive=False), state, img_list, first_run


seafoam = Seafoam()


def build_demo(embed_mode):
    textbox = Textbox(show_label=False,
                         placeholder="Please upload your video first by clicking 'Upload Video'. Enter text and press ENTER",
                         visible=True, container=False)
    with Blocks(title="Oryx Video-ChatGPT", theme=seafoam, css=css) as demo:
        state = State()
        img_list = State()
        first_run = State()

        if not embed_mode:
            Markdown(title)
            # Markdown(description)

        with Row():
            with Column(scale=3):
                imagebox = Video()
                upload_button = Button(value="Upload Video", interactive=True, variant="primary")

                # Add a text note beneath the button
                Markdown(
                    "NOTE: Please make sure you **<span style='color:red'>press the 'Upload Video' button</span>**"
                    " and wait for it to display 'Start Chatting "
                    "before submitting question to Video-ChatGPT.")
                cur_dir = os.path.dirname(os.path.abspath(__file__))
                Examples(examples=[
                    [f"{cur_dir}/demo_sample_videos/sample_2.mp4", "Why is this video strange?"],
                    [f"{cur_dir}/demo_sample_videos/sample_6.mp4",
                     "Can you write a short poem inspired from the video."],
                    [f"{cur_dir}/demo_sample_videos/sample_8.mp4",
                     "Where is this video taken? What place/landmark is shown in the video?"],
                    [f"{cur_dir}/demo_sample_videos/sample_15.mp4",
                     "What is the main challenge faced by the people on the boat."],
                ], inputs=[imagebox, textbox])

                with Accordion("Parameters", open=False, visible=False) as parameter_row:
                    temperature = Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                            label="Temperature", )
                    max_output_tokens = Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True,
                                                  label="Max output tokens", )

            with Column(scale=6):
                chatbot = Chatbot(elem_id="chatbot", label="VideoChat-GPT Chatbot", visible=True, height=600)
                with Row():
                    with Column(scale=8):
                        textbox.render()
                    with Column(scale=1, min_width=60):
                        submit_btn = Button(value="Submit", visible=True)
                with Row(visible=True) as button_row:
                    upvote_btn = Button(value="👍  Upvote", interactive=False)
                    downvote_btn = Button(value="👎  Downvote", interactive=False)
                    flag_btn = Button(value="⚠️  Flag", interactive=False)
                    regenerate_btn = Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = Button(value="🗑️  Clear history", interactive=False)

        Markdown(disclaimer)

        if not embed_mode:
            Markdown(tos_markdown)

        url_params = JSON(visible=False)
        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(upvote_last_response,
                         [state, imagebox], [textbox, imagebox, upvote_btn, downvote_btn, flag_btn])
        downvote_btn.click(downvote_last_response,
                           [state, imagebox], [textbox, imagebox, upvote_btn, downvote_btn, flag_btn])
        flag_btn.click(flag_last_response,
                       [state, imagebox], [textbox, imagebox, upvote_btn, downvote_btn, flag_btn])
        regenerate_btn.click(regenerate, [state],
                             [state, chatbot, textbox, imagebox] + btn_list).then(
            chat.answer, [state, img_list, temperature, max_output_tokens, first_run],
            [state, chatbot, img_list, first_run] + btn_list)
        clear_btn.click(clear_history, [img_list],
                        [state, chatbot, textbox, imagebox, upload_button, img_list] + btn_list)

        upload_button.click(upload_image, [imagebox, state],
                            [imagebox, upload_button, state, img_list, first_run])

        textbox.submit(add_text, [state, textbox, imagebox, first_run], [state, chatbot, textbox] + btn_list
                       ).then(chat.answer, [state, img_list, temperature, max_output_tokens, first_run],
                              [state, chatbot, img_list, first_run] + btn_list)
        submit_btn.click(add_text, [state, textbox, imagebox, first_run], [state, chatbot, textbox] + btn_list
                         ).then(chat.answer, [state, img_list, temperature, max_output_tokens, first_run],
                                [state, chatbot, img_list, first_run] + btn_list)

        demo.load(load_demo, [url_params],
                  [state, chatbot, textbox, upload_button, submit_btn, button_row, parameter_row],
                  _js=get_window_url_params)

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--vision_tower_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--conv-mode", type=str, default="video-chatgpt_v1")
    parser.add_argument("--projection_path", type=str, required=False, default="")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"args: {args}")
    logger.info(args)
    disable_torch_init()

    model, vision_tower, tokenizer, image_processor, video_token_len = \
        initialize_model(args.model_name, args.projection_path)

    # Create replace token, this will replace the <video> in the prompt.
    replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN

    # Create chat for the demo
    chat = Chat(args.model_name, args.conv_mode, tokenizer, image_processor, vision_tower, model, replace_token)
    print('Initialization Finished')

    demo = build_demo(args.embed)
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10).launch(share=True)