import gradio as gr
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import time

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344

title = """
        # Stability AI - Developer Platform WebUI
        ### UI for using the stable image api

        API Key is required to use this service.
        https://platform.stability.ai/account/keys
        
        Contact: D̷ELL@Stability AI - Advocate (https://x.com/xqdior) / Author: umise (https://x.com/UiE029)
"""

title_jp = """
        # Stability AI - Developer Platform WebUI
        ### このSpaceは、Stable Image APIを使用するためのWEB-UIです。

        このサービスを利用するにはAPIキーが必要です。以下のリンクから取得してください。
        https://platform.stability.ai/account/keys

        お問い合わせ先: D̷ELL@Stability AI - Advocate (https://x.com/xqdior) / Author: umise (https://x.com/UiE029)
"""

overview = """

        **Overview**
        Stability AI’s Stable Image services offer a growing set of APIs for developers to build the best in class image applications.

        - **Disrupting Content Creation:** Stability’s Image APIs are the foundation for applications disrupting publishing, media, gaming, marketing, advertising, design, and more.
        - **For Developers:** Application developers can build advanced features for designers, photographers, content creators, and a variety of B2C customers.
        - **Simple APIs:** Stability AI is focused on delivering simple APIs for easy integration into applications with a high bar for quality, alignment, speed, and safety.

        Get Started Now: https://platform.stability.ai/docs/getting-started/stable-image
        """

overview_jp = """
        各種モードについて
        - テキストから生成:
            プロンプトを基に画像を生成します。
        - アップスケール:
            - 通常のアップスケール:
                できるだけ絵を変更せずにアップスケールします
            - クリエィティブなアップスケール:
                絵全体をいい感じになるようにアップスケールします
        - 画像の編集:
            - 消去:
                マスクした要素を削除します
            - インペイント:
                マスクされた部分とプロンプトを基に再生成します
            - アウトペイント:
                指定した範囲分、画像を拡張します
            - 検索と置き換え:
                検索プロンプトに入力されたオブジェクトを探し、プロンプトをもとにオブジェクトを再生成します
        - コントロール:
            - 構図:
                入力画像の構図を基に新しい画像を生成します
            - スケッチ:
                ラフなスケッチとプロンプトを基に生成します
        """

model_url = {
    "ImageUltra": "https://api.stability.ai/v2beta/stable-image/generate/ultra",
    "ImageCore": "https://api.stability.ai/v2beta/stable-image/generate/core",
    "StableDiffusion3": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
}

service_url = {
    "Conservative_Upscale": "https://api.stability.ai/v2beta/stable-image/upscale/conservative",
    "Creative_Upscale": "https://api.stability.ai/v2beta/stable-image/upscale/creative",
    "Erase": "https://api.stability.ai/v2beta/stable-image/edit/erase",
    "Inpaint": "https://api.stability.ai/v2beta/stable-image/edit/inpaint",
    "Outpaint": "https://api.stability.ai/v2beta/stable-image/edit/outpaint",
    "SR": "https://api.stability.ai/v2beta/stable-image/edit/search-and-replace",
    "RMBG": "https://api.stability.ai/v2beta/stable-image/edit/remove-background",
    "Sketch": "https://api.stability.ai/v2beta/stable-image/control/sketch",
    "Structure": "https://api.stability.ai/v2beta/stable-image/control/structure",
}


translations = {
    "en": {
        "api_key": "API Key",
        "api_key_placeholder": "Enter your API key",
        "model_label": "Model",
        "mode_label": "Mode",
        "prompt_placeholder": "Enter your prompt",
        "negative_prompt_placeholder": "Enter a negative prompt",
        "seed_label": "Seed",
        "randomize_seed_label": "Randomize seed",
        "aspect_label": "Aspect ratio",
        "run_button": "Run",
        "result_label": "Result",
        "copy_field_placeholder": "Copy the field",
        "Negative_prompt": "Negative prompt",
        "Advanced_Settings": "Advanced Settings",
        "Example": "Example",
        "Generate": "Generate",
        "Upscale": "Upscale",
        "Edit": "Edit",
        "Control": "Control",
        "Submode": "Submode",
        "Conservative": "Conservative",
        "Creative": "Creative",
        "Erase": "Erase",
        "Inpaint": "Inpaint",
        "Outpaint": "Outpaint",
        "Structure": "Structure",
        "Sketch": "Sketch",
        "Search_and_Replace": "Search and Replace",
        "Remove_Background": "Remove Background",
        "input_image": "Input Image",
        "style_preset": "Style preset",
        "preset_description": "This parameter is only available for ImageCore model.",
        "Search_prompt_placeholder": "Enter a search prompt",
        "Control_Strength": "Control Strength",
        "overview": overview,
        "overview_label": "How to use",
        "title": title,
    },
    "ja": {
        "api_key": "APIキー",
        "api_key_placeholder": "APIキーを入力してください",
        "model_label": "モデル",
        "mode_label": "モード",
        "prompt_placeholder": "プロンプトを入力してください",
        "negative_prompt_placeholder": "ネガティブプロンプトを入力してください",
        "seed_label": "シード",
        "randomize_seed_label": "シードをランダム化",
        "aspect_label": "アスペクト比",
        "run_button": "実行",
        "result_label": "結果",
        "copy_field_placeholder": "ここに貼り付け用の情報が出てきます",
        "Negative_prompt": "ネガティブプロンプト",
        "Advanced_Settings": "追加設定",
        "Example": "例",
        "Generate": "テキストから生成",
        "Upscale": "アップスケール",
        "Edit": "画像の編集",
        "Control": "コントロールモード",
        "Submode": "サブモード",
        "Conservative": "通常のアップスケール",
        "Creative": "クリエィティブなアップスケール",
        "Erase": "消去",
        "Inpaint": "インペイント",
        "Outpaint": "アウトペイント(拡張)",
        "Structure": "構図",
        "Sketch": "スケッチ",
        "Search_and_Replace": "検索と置き換え",
        "Remove_Background": "背景削除",
        "input_image": "入力画像",
        "style_preset": "スタイルのプリセット",
        "preset_description": "このパラメータはimage coreのときにだけ有効になります",
        "Search_prompt_placeholder": "探したい要素を入力してください",
        "Control_Strength": "コントロールネットの適用強度",
        "overview": overview_jp,
        "overview_label": "使い方",
        "title": title_jp,
    },
}

lang = "ja"


def bytes_to_image(image):
    image = BytesIO(image)
    image = Image.open(image).convert("RGB")
    return image


def image_to_bytes(image):
    byte_io = BytesIO()
    image.save(byte_io, format="PNG")
    byte_data = byte_io.getvalue()
    return byte_data


def send_request(url, api_key, file, data):
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "image/*"},
        files=file,
        data=data,
    )
    return response


def generate(
    prompt,
    negative_prompt,
    seed,
    mode,
    submode,
    input_image,
    mask,
    CNstrength,
    search_prompt,
    op_left,
    op_right,
    op_up,
    op_down,
    randomize_seed,
    aspect,
    model,
    preset,
    api_key,
):
    if randomize_seed:
        seed = 0

    file = {}
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "output_format": "png",
        "seed": seed,
        "aspect_ratio": aspect,
    }
    data_rmbg = {
        "output_format": "png",
    }
    if input_image is not None:
        file["image"] = image_to_bytes(input_image)
        if mask is not None:
            file["mask"] = image_to_bytes(mask)

    if mode == translations[lang]["Generate"]:
        file["none"] = ""
        if model == "Stable Image Ultra (8B + workflow)":
            url = model_url["ImageUltra"]
        elif model == "Stable Image Core (2B + workflow)":
            url = model_url["ImageCore"]
            data["style_preset"] = preset
        elif model == "Stable Diffusion 3 Medium (2B)":
            url = model_url["StableDiffusion3"]
            data["model"] = "sd3-medium"
        elif model == "Stable Diffusion 3 Large (8B)":
            url = model_url["StableDiffusion3"]
            data["model"] = "sd3-large"
        elif model == "Stable Diffusion 3 Large Turbo (8B Turbo)":
            url = model_url["StableDiffusion3"]
            data["model"] = "sd3-large-turbo"
        else:
            raise ValueError("Invalid model type")

    elif mode == translations[lang]["Upscale"]:
        if submode == translations[lang]["Conservative"]:
            url = service_url["Conservative_Upscale"]
        elif submode == translations[lang]["Creative"]:
            url = service_url["Creative_Upscale"]

    elif mode == translations[lang]["Edit"]:
        if submode == translations[lang]["Erase"]:
            url = service_url["Erase"]
        elif submode == translations[lang]["Inpaint"]:
            url = service_url["Inpaint"]
        elif submode == translations[lang]["Outpaint"]:
            url = service_url["Outpaint"]
            data["left"] = op_left
            data["right"] = op_right
            data["up"] = op_up
            data["down"] = op_down
        elif submode == translations[lang]["Search_and_Replace"]:
            url = service_url["SR"]
            data["search_prompt"] = search_prompt
        elif submode == translations[lang]["Remove_Background"]:
            data = data_rmbg
            url = service_url["RMBG"]

    elif mode == translations[lang]["Control"]:
        data["control_strength"] = CNstrength
        if submode == translations[lang]["Sketch"]:
            url = service_url["Sketch"]
        elif submode == translations[lang]["Structure"]:
            url = service_url["Structure"]
    response = send_request(url, api_key, file, data)
    
    if response.status_code == 200:
        if (
            mode == translations[lang]["Upscale"]
            and submode == translations[lang]["Creative"]
        ):
            generation_id = response.json().get("id")
            if not generation_id:
                raise Exception("No generation ID returned for creative upscale")

            # Polling for the result
            result_url = f"https://api.stability.ai/v2beta/stable-image/upscale/creative/result/{generation_id}"
            while True:
                result_response = requests.get(
                    result_url,
                    headers={"accept": "image/*", "authorization": f"Bearer {api_key}"},
                )
                if result_response.status_code == 202:
                    print("Generation in-progress, try again in 10 seconds.")
                    time.sleep(10)
                elif result_response.status_code == 200:
                    print("Generation complete!")
                    image = result_response.content
                    image = bytes_to_image(image)
                    copy_filed_value = f"prompt:{prompt}, negative:{negative_prompt}, mode:{mode}, submode:{submode}"
                    return image, seed, copy_filed_value
                else:
                    raise Exception(str(result_response.json()))
        else:
            image = response.content
            image = bytes_to_image(image)
            copy_filed_value = f"prompt:{prompt}, negative:{negative_prompt}, mode:{mode}, submode:{submode}"
            return image, seed, copy_filed_value
    else:
        raise Exception(str(response.json()))


examples = [
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "An astronaut riding a green horse",
    "A delicious ceviche cheesecake slice",
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 50vw;
}
"""


def update_style_visibility(model):
    if model == "Stable Image Core (2B + workflow)":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def update_mode(mode):
    submode_update = gr.update(choices=["None"], visible=False)
    image_label_update = gr.update(visible=False)
    img_input_update = gr.update(visible=False)
    mask_update = gr.update(visible=False)

    if mode == translations[lang]["Generate"]:
        submode_update = gr.update(visible=False)
    elif mode == translations[lang]["Upscale"]:
        submode_update = gr.update(
            choices=[
                translations[lang]["Conservative"],
                translations[lang]["Creative"],
            ],
            value=translations[lang]["Conservative"],
            visible=True,
        )
        img_input_update = gr.update(visible=True)
        image_label_update = gr.update(visible=True)
    elif mode == translations[lang]["Edit"]:
        submode_update = gr.update(
            choices=[
                translations[lang]["Erase"],
                translations[lang]["Inpaint"],
                translations[lang]["Outpaint"],
                translations[lang]["Search_and_Replace"],
                translations[lang]["Remove_Background"],
            ],
            value=translations[lang]["Erase"],
            visible=True,
        )
        img_input_update = gr.update(visible=True)
        image_label_update = gr.update(visible=True)
    elif mode == translations[lang]["Control"]:
        submode_update = gr.update(
            choices=[
                translations[lang]["Structure"],
                translations[lang]["Sketch"],
            ],
            value=translations[lang]["Structure"],
            visible=True,
        )
        img_input_update = gr.update(visible=True)
        image_label_update = gr.update(visible=True)

    return submode_update, img_input_update, mask_update, image_label_update


def update_submode(submode):
    mask = gr.update(visible=False)
    outpaint = gr.update(visible=False)
    cn = gr.update(visible=False)
    search_prompt = gr.update(visible=False)

    if submode in [translations[lang]["Erase"], translations[lang]["Inpaint"]]:
        mask = gr.update(visible=True)

    else:
        if submode == translations[lang]["Outpaint"]:
            outpaint = gr.update(visible=True)

        elif submode in [translations[lang]["Structure"], translations[lang]["Sketch"]]:
            cn = gr.update(visible=True)

        elif submode == translations[lang]["Search_and_Replace"]:
            search_prompt = gr.update(visible=True)

    return mask, outpaint, cn, search_prompt


with gr.Blocks(css=css, theme="NoCrypt/miku") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            translations[lang]["title"],
        )
        with gr.Accordion(translations[lang]["overview_label"], open=False):
            gr.Markdown(
                translations[lang]["overview"],
            )

        with gr.Row():
            api_key = gr.Text(
                label=translations[lang]["api_key"],
                type="password",
                placeholder=translations[lang]["api_key_placeholder"],
                max_lines=1,
                container=False,
            )

        with gr.Row():
            model = gr.Dropdown(
                label=translations[lang]["model_label"],
                choices=[
                    "Stable Image Ultra (8B + workflow)",
                    "Stable Image Core (2B + workflow)",
                    "Stable Diffusion 3 Large Turbo (8B Turbo)",
                    "Stable Diffusion 3 Large (8B)",
                    "Stable Diffusion 3 Medium (2B)",
                ],
                value="Stable Image Ultra (8B + workflow)",
            )
            mode = gr.Dropdown(
                label=translations[lang]["mode_label"],
                choices=[
                    translations[lang]["Generate"],
                    translations[lang]["Upscale"],
                    translations[lang]["Edit"],
                    translations[lang]["Control"],
                ],
                value=translations[lang]["Generate"],
            )

        submode = gr.Dropdown(
            label=translations[lang]["Submode"],
            choices=["None"],
            visible=False,
            value="None",
        )

        with gr.Row():
            with gr.Column():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder=translations[lang]["prompt_placeholder"],
                    container=False,
                )
                search_prompt = gr.Text(
                    label="search prompt",
                    visible=False,
                    show_label=False,
                    max_lines=1,
                    placeholder=translations[lang]["Search_prompt_placeholder"],
                )
            run_button = gr.Button(translations[lang]["run_button"], scale=0)
        with gr.Row():
            gr.Examples(
                label=translations[lang]["Example"], examples=examples, inputs=[prompt]
            )
        with gr.Row():
            with gr.Column():
                image_label = gr.Markdown(
                    value=translations[lang]["input_image"], visible=False
                )
                image = gr.Image(
                    type="pil",
                    label="img input",
                    width="20vw",
                    height="20vw",
                    show_label=True,
                    visible=False,
                    interactive=True,
                    container=False,
                )
            with gr.Column(visible=False) as mask:
                mask_label = gr.Markdown(value="input mask")
                mask_input = gr.Image(
                    type="pil",
                    label="mask",
                    width="20vw",
                    height="20vw",
                    show_label=True,
                    interactive=True,
                    container=False,
                )

        with gr.Row():
            result = gr.Image(
                label=translations[lang]["result_label"], width="20vw", height="20%"
            )

        with gr.Accordion(translations[lang]["Advanced_Settings"], open=False):
            negative_prompt = gr.Text(
                label=translations[lang]["Negative_prompt"],
                max_lines=1,
                placeholder=translations[lang]["negative_prompt_placeholder"],
            )
            seed = gr.Slider(
                label=translations[lang]["seed_label"],
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            CN_strength = gr.Slider(
                label=translations[lang]["Control_Strength"],
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.5,
                visible=False,
            )
            randomize_seed = gr.Checkbox(
                label=translations[lang]["randomize_seed_label"], value=True
            )
            aspect = gr.Radio(
                choices=[
                    "1:1",
                    "16:9",
                    "21:9",
                    "2:3",
                    "3:2",
                    "4:5",
                    "5:4",
                    "9:16",
                    "9:21",
                ],
                label=translations[lang]["aspect_label"],
                value="1:1",
            )
            with gr.Row(visible=False) as style:
                style_preset = gr.Radio(
                    choices=[
                        "3d-model",
                        "analog-film",
                        "anime",
                        "cinematic",
                        "comic-book",
                        "digital-art",
                        "enhance",
                        "fantasy-art",
                        "isometric",
                        "line-art",
                        "low-poly",
                        "modeling-compound",
                        "neon-punk",
                        "origami",
                        "photographic",
                        "pixel-art",
                        "tile-texture",
                    ],
                    label=translations[lang]["style_preset"],
                    value="anime",
                    info=translations[lang]["preset_description"],
                )
            with gr.Row(visible=False) as outpaint_scale:
                paint = gr.Markdown(value="Outpain Scale")
                op_left = gr.Slider(
                    label="left", minimum=0, maximum=2000, step=4, value=200
                )
                op_right = gr.Slider(
                    label="right", minimum=0, maximum=2000, step=4, value=200
                )
                op_up = gr.Slider(
                    label="up", minimum=0, maximum=2000, step=4, value=200
                )
                op_down = gr.Slider(
                    label="down", minimum=0, maximum=2000, step=4, value=200
                )

        copy_filed = gr.TextArea(
            value="",
            label="Copy Field",
            max_lines=1,
            placeholder=translations[lang]["copy_field_placeholder"],
            show_copy_button=True,
            container=False,
        )
        gr.Markdown(
            f"""
        ## License
        This work is licensed under a
        [Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

        [![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

        [cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
        [cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
        [cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

        **MIT Licensed Source Code**
        Portions of this work are licensed under the MIT License. For more details, please refer to the original source at: [stabilityai/stable-diffusion-3-medium](https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium) 
        """
        )
    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            mode,
            submode,
            image,
            mask_input,
            CN_strength,
            search_prompt,
            op_left,
            op_right,
            op_up,
            op_down,
            randomize_seed,
            aspect,
            model,
            style_preset,
            api_key,
        ],
        outputs=[result, seed, copy_filed],
    )

    mode.change(
        fn=update_mode, inputs=mode, outputs=[submode, image, mask, image_label]
    )
    submode.change(
        fn=update_submode,
        inputs=submode,
        outputs=[mask, outpaint_scale, CN_strength, search_prompt],
    )
    model.change(fn=update_style_visibility, inputs=model, outputs=style)
demo.launch(server_name="0.0.0.0")
