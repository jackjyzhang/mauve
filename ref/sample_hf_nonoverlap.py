import sys
sys.path.append('/home/gridsan/jzhang2/repos/mauve')
from datasets import load_from_disk
from util import load_file_by_line, write_file_by_line

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, required=True, help='path to hf dataset (NOT dict!)')
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('-n', '--num', type=int, default=10)
    parser.add_argument('-e', '--exclude_file', type=str, default=None, help='path to file to exclude')
    parser.add_argument('-s', '--seed', type=int, default=0)

    args = parser.parse_args()

    ds = load_from_disk(args.dataset_path)
    ds.shuffle(seed=args.seed)
    exclude_texts = set(load_file_by_line(args.exclude_file))

    texts = []
    for i in range(len(ds)):
        text = ds[i]['text']
        if not text in exclude_texts:
            texts.append(text)
        if len(texts) == args.num: break
    if len(texts) < args.num:
        print(f'WARNING: not enough text to sample... requested: {args.num}, actual: {len(texts)}')
    write_file_by_line(args.output_path, texts)