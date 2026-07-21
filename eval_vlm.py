import time
import argparse
import os
import warnings
import torch
import random
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_vlm import skyRopeVLM, VLMConfig
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_form, trust_remote_code=True)
    if 'model' in args.load_form:
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = (f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth')
        model = skyRopeVLM(
            VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)),
            vision_model_path="./model/siglip2-base-p32-256-ve"
        )
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = skyRopeVLM.get_vision_model("./model/siglip2-base-p32-256-ve")
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer, model.processor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="skyRope-V Chat")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称前缀（pretrain_vlm, sft_vlm）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--image_dir', default='./dataset/eval_images/', type=str, help="测试图像目录")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    args = parser.parse_args()

    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, True, skip_special_tokens=True)

