import argparse
import os
from src.mauve.compute_mauve import compute_mauve
import matplotlib.pyplot as plt 

'''partially adapted from nl-command'''

def load_file_by_line(path):
    '''
    each line is an example of ref/gen text
    '''
    texts = []
    with open(path, 'r') as f:
        for _, line in enumerate(f.readlines()):
            line = line.strip()
            if not line.startswith('#'):
                texts.append(line)
    return texts

def eval_mauve(ref, gen):
    if ref is None:
        return -1
    gen = [text.replace('[BOS]', '').replace('<|endoftext|>', '') for text in gen] # get rid of BOS, EOS
    gen = [text for text in gen if len(text) > 0]
    print(f'number of mauve generations: {len(gen)}')
    if len(ref) < len(gen): print('WARNING: MAUVE #reference < #generated! They should be the same!')
    if len(ref) > len(gen):
        print('MAUVE #reference > #generated, truncating reference to have length #generated')
        ref = ref[:len(gen)]
    out = compute_mauve(p_text=ref, q_text=gen, device_id=0, max_text_length=512, verbose=False)
    print(f'MAUVE={out.mauve}')
    return out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference', type=str, required=True, help='reference for mauve, should be SAME LENGTH as generation')
    parser.add_argument('-g', '--generation', type=str, nargs='+', required=True)
    parser.add_argument('-d', '--save_dir', type=str, default=None, help='save dir, default to dir of last generation file')
    parser.add_argument('-s', '--output_suffix', type=str, default='', help='output name mauve_{suffix}.csv and div_{suffix}.png')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    ref = load_file_by_line(args.reference)
    plt.figure(dpi=160)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.autoscale(enable=True, tight=True)
    plt.grid()

    results = 'name,mauve\n'
    for gen_path in args.generation:
        if not os.path.exists(gen_path):
            print(f'WARNING: path {gen_path} does not exist, skipping...')
            continue
        if os.path.splitext(gen_path)[1] != '.txt':
            print(f'WARNING: {gen_path} is not a text file, skipping...')
            continue
        basename = os.path.splitext(os.path.basename(gen_path))[0]
        print(f'file basename: {basename}')
        gen = load_file_by_line(gen_path)
        out = eval_mauve(ref, gen)
        results += f'{basename},{out.mauve}\n'
        plt.plot(out.divergence_curve[:, 1], out.divergence_curve[:, 0], label=basename)

    save_dir = os.path.dirname(gen_path) if args.save_dir is None else args.save_dir # use directory of last generation file by default
    print(results)
    if len(args.output_suffix) > 0: args.output_suffix = '_' + args.output_suffix
    with open(os.path.join(save_dir, f'mauve{args.output_suffix}.csv'), 'w') as f:
        print(results, file=f)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'div{args.output_suffix}.png'))
