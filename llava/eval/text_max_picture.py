import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
from PIL import Image
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def eval_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16  # or bfloat16, float16, float32
    # Disable default torch initialization
    disable_torch_init()

    # Load model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, vision_tower, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Load questions
    with open(args.question_file) as f:
        questions = json.load(f)

    # Prepare to save results
    results_file = args.results_file
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # Initialize variables for testing
    max_images = 0  # Track the maximum number of images successfully processed
    images_num = 1  # Start with 1 image and incrementally increase

    while True:
        try:
            # Select a subset of images for testing
            image_file = '/home/zwding/hospital_data_3D/zhongzong/10336016/T2A.npy' # Assuming each question has one image
            image = np.load(image_file)
            if len(image.shape) == 3:  # If (D, H, W), add channel dimension
                    image = np.expand_dims(image, axis=0)  # Convert to (C, D, H, W), where C=1
            elif len(image.shape) != 4:
                    raise ValueError(f"Unexpected image shape: {image.shape}. Expected (D, H, W) or (C, D, H, W).")
            image = torch.tensor(image, dtype=torch.float32).to(device)  # Convert to tensor and move to device
            # Prepare inputs
            image_tensor = []
            for i in range(images_num):
                image_tensor.append(image)

            # Stack images into a single batch
            image_tensor = torch.stack(image_tensor)
            image_token = '<image>'
            # Prepare text input
            conv = conv_templates[args.conv_mode].copy()
            qs = "Describe these images." + image_token * len(image_tensor)
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

            # Generate output
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

            # If successful, update max_images
            max_images = images_num
            print(f"Successfully processed {images_num} images.")

            # Increment batch size for the next test
            images_num += 1

        except Exception as e:
            # If an error occurs, stop testing and log the result
            print(f"Failed to process {images_num} images. Error: {e}")
            break

    # Save the result
    with open(results_file, "w") as f:
        json.dump({"max_images": max_images}, f, indent=4)
    print(f"Testing complete. Maximum number of images processed: {max_images}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/home/zwding/ly/LongLLaVA3D/ckpts/all_finetune_15epoch_3D_without_stage3171_Our_CLIPandprojector_6epoch_COT_format')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="/home/zwding/ly/LongLLaVA3D/json_data/test_data/3D_stage_QA_test_our_diagnosis-withimg.json")
    parser.add_argument("--results-file", type=str, default="/home/zwding/ly/LongLLaVA3D/results/max_images_test_result.json")
    parser.add_argument("--conv-mode", type=str, default='jamba')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=2768)
    args = parser.parse_args()

    eval_model(args)