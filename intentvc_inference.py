from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import os
import json
from llamafactory.hparams.model_args import ProcessorArguments
from llamafactory.data.mm_plugin import get_mm_plugin
import argparse
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inference script for Qwen2.5-VL model.")
    parser.add_argument("--model", default='Qwen/Qwen2.5-VL-7B-Instruct', type=str, help="Model name or path.")
    parser.add_argument("--dataset_path", default='IntentVCDatasets/IntentVC', type=str, help="Path to the video file.")
    parser.add_argument("--video_path1", default='data/video', type=str, help="Path to the first video file.")
    parser.add_argument("--video_path2", default='data/video_small', type=str, help="Path to the second video file.")
    # decoding arguments
    parser.add_argument("--num_beams", default=3, type=int, help="Number of beams for beam search.")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling.")
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for sampling.")

    # misc
    parser.add_argument("--old_gpu", action="store_true", help="GPU older than 3090, use float32 instead of float16.")
    return parser.parse_args()

def inference(path1: str, path2: str, subdir: str, model, processor, proc_args, mm_plugin, args) -> list[str]:

    # Process video
    video_data = mm_plugin._regularize_videos(
        videos=[path1, path2],
        image_max_pixels=proc_args.video_max_pixels,
        image_min_pixels=proc_args.video_min_pixels,
        video_fps=proc_args.video_fps,
        video_maxlen=proc_args.video_maxlen,
        # is_vid_aug=False # mm_plugin.py 中如果没有加 augmentation 的话这里就不用加
    )

    # Create messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_data["videos"][0]},
                {"type": "video", "video": video_data["videos"][1]},
                {
                    "type": "text",
                    "text": f"Describe this video with the intention-oriented object: {subdir}.The following is a reference format for two descriptive sentences: 1.a [color] [object] [action] in [environmental context]. 2.the [object] [performs action] while [surrounding context]."
                }
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        videos=video_data["videos"],
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=args.do_sample, temperature=args.temperature, num_beams=args.num_beams)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    torch.cuda.empty_cache()
    return output_text

if __name__ == "__main__":
    args = parse_args()

    # Process all subdirectories
    subdirs = [name for name in os.listdir(args.dataset_path) 
              if os.path.isdir(os.path.join(args.dataset_path, name))]

    # Initialize model and processor
    if args.old_gpu:
        # 2080Ti
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map="cuda"
        )
    else:
        # > 30 series
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="cuda"
        )

    processor = AutoProcessor.from_pretrained(args.model)

    # Create processor arguments and plugin
    proc_args = ProcessorArguments()
    mm_plugin = get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>")

    total_num = len(subdirs) * 6  # 6 videos per subdir
    pbar = tqdm(total=total_num, desc="Inference Progress")
    result = {}
    for subdir in subdirs:
        for i in range(15, 21):
            try:
                video_path1 = f"data/video/{subdir}-{i}.mp4"
                video_path2 = f"data/video_small/{subdir}-{i}.mp4"
                result_list = inference(video_path1, video_path2, subdir, model, processor, proc_args, mm_plugin, args)
                result[f"{subdir}-{i}"] = result_list[0] if result_list else ""
                pbar.update(1)
            except Exception as e:
                print(f"Error processing {subdir}-{i}: {str(e)}")
                continue
    pbar.close()

    # Split and save results
    public_result = {k: v for k, v in result.items() if int(k.split('-')[-1]) in range(15, 18)}
    private_result = {k: v for k, v in result.items() if int(k.split('-')[-1]) in range(18, 21)}

    # Update public results
    public_file_path = 'result_public.json'
    try:
        with open(public_file_path, 'r', encoding='utf-8') as f:
            public_data = json.load(f)
        public_data['captions'].update(public_result)
        with open(public_file_path, 'w', encoding='utf-8') as f:
            json.dump(public_data, f, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        print(f"{public_file_path} 文件未找到")

    # Update private results
    private_file_path = 'result_private.json'
    try:
        with open(private_file_path, 'r', encoding='utf-8') as f:
            private_data = json.load(f)
        private_data['captions'].update(private_result)
        with open(private_file_path, 'w', encoding='utf-8') as f:
            json.dump(private_data, f, ensure_ascii=False, indent=4)
    except FileNotFoundError:
        print(f"{private_file_path} 文件未找到")