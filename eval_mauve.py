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
        basename = os.path.splitext(os.path.basename(gen_path))[0]
        print(f'file basename: {basename}')
        gen = load_file_by_line(gen_path)
        out = eval_mauve(ref, gen)
        results += f'{basename},{out.mauve}\n'
        plt.plot(out.divergence_curve[:, 1], out.divergence_curve[:, 0], label=basename)

    print(results)
    plt.legend()
    plt.savefig('div.png')
