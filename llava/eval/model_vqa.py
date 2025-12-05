import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
import numpy as np
from PIL import Image
import math

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16  # or bfloat16, float16, float32
    # Disable default torch initialization
    disable_torch_init()

    # Load model and check if successfully loaded
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, vision_tower, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # if not (tokenizer and model and image_processor):
    #     raise RuntimeError("Failed to load model, tokenizer, or image processor.")

    # Load and split questions into chunks
    with open(args.question_file) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # Prepare to save answers
    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Set prompt
    # qs = '我们将为您提供一批患者的多种模态医学影像，希望您能够基于这些影像制定诊断。\n这些图像是：'

    # Open answer file for writing
    with open(answers_file, "w") as ans_file:
        for line in tqdm(questions):
            idx = line["id"]
            image_file = line["image"]
            # human_answer = line["human_answer"]
            human_answer = line["conversations"][1]['value']
            qs = line["conversations"][0]['value']
            # Construct question with image placeholders
            cur_prompt = qs
            hospital = line["hospital"]
            patient_name = line["patient_name"]
            information = hospital + '-' + patient_name            
            # qs_with_images = qs + '\n' + ''.join([f"<image>" for _ in range(len(image_file))])
            qs_with_images = qs 
            # Setup conversation template
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs_with_images)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

            # Load and process images
            # images = [np.load(image_name) for image_name in image_file]
            # # image_tensor = process_images(images, image_processor,model_cfg).to(device, dtype=torch.float16)
            # # image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values'].to(device, dtype=torch.float16)
            # image_tensor = [torch.from_numpy(img).unsqueeze(0) for img in images]
            image_tensor = []
            for image_name in image_file:
                image = np.load(image_name)
                if len(image.shape) == 3:  # 如果是 (D, H, W)，假设单通道数据
                    image = np.expand_dims(image, axis=0)  # 转为 (C, D, H, W)，其中 C=1
                    #print(image.shape)
                elif len(image.shape) != 4:
                    raise ValueError(f"Unexpected image shape: {image.shape}. Expected (D, H, W) or (C, D, H, W).")    
                 # 转为 PyTorch 张量
                image = torch.tensor(image, dtype=torch.float32)  # 转为浮点张量    
                image_tensor.append(image)
            # Setup stopping criteria
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # Generate model output
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True
                )

            # Decode and format the output
            outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)].strip()
            print(outputs)
            # Save answer to file
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": idx,
                "information": information,
                # "prompt": cur_prompt,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_name,
                "human_answer": human_answer,
                "metadata": {}
            }) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/home/ly/LLMs/Cervical-RG/ckpt/longllava_all_finetune_15epoch_3D_without_stage3171_Our_CLIPandprojector_6epoch')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="/home/ly/LLMs/Cervical-RG/json_data/test_data/test_wangbo.json")
    parser.add_argument("--answers-file", type=str, default="/home/ly/LLMs/Cervical-RG/Answer/test_wangbo.json")
    parser.add_argument("--conv-mode", type=str, default='jamba')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2768)
    args = parser.parse_args()

    eval_model(args)
