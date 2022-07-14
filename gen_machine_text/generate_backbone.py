import sys
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser
import time
from tqdm import tqdm
from constants import BOS_TOKEN, PAD_TOKEN

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--backbone', type=str, required=True, help='load ckpt from file if given')
    parser.add_argument('--output', type=str, default=None)

    parser.add_argument('--topk', type=int, default=None, help='top k for sampling')
    parser.add_argument('--topp', type=float, default=None, help='top p for sampling')
    parser.add_argument('--num', type=int, default=10)

    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--show_special_tokens', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForCausalLM.from_pretrained(args.backbone, return_dict=True, pad_token_id=tokenizer.encode(PAD_TOKEN)[0]).to(args.device)
    model.eval()

    if args.output:
        f = open(args.output, 'w')
    else:
        f = sys.stdout

    print(f'# model path: {args.backbone}', file=f)
    t1 = time.time()
    for _ in tqdm(range(args.num)):
        prefix = BOS_TOKEN
        input_ids = tokenizer.encode(prefix, return_tensors='pt').to(args.device)
        topk_output = model.generate(
            input_ids, 
            do_sample=True, 
            max_length=args.max_length, 
            top_k=args.topk,
            top_p=args.topp
        )
        print(tokenizer.decode(topk_output[0], skip_special_tokens=not args.show_special_tokens), file=f)

    t2 = time.time()
    avg_time = (t2-t1) / args.num
    print(f'average time per sentence: {avg_time} seconds')
    f.close()
