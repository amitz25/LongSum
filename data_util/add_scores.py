import argparse
import json
import os
from glob import glob
import multiprocessing
import tqdm

scores_dir = os.path.join('arxiv-release', 'scores')
res_dir = os.path.join('arxiv-release', 'chunked_scored')
chunked_dir = os.path.join('arxiv-release', 'chunked')


def gen_res(path):
    basename = os.path.basename(path)
    res_path = os.path.join(res_dir, basename)

    orig_lines = open(os.path.join(chunked_dir, basename), 'r').read().splitlines()
    score_lines = open(os.path.join(scores_dir, basename), 'r').read().splitlines()
    res_lines = []

    for i, line in enumerate(score_lines):
        line = json.loads(line)
        orig_line = json.loads(orig_lines[i])
        assert line['article_id'] == orig_line['article_id'], "Inconsistent lines!"
        orig_line['similarity_scores'] = line['scores']
        res_lines.append(str(orig_line))

    open(res_path, 'w').write('\n'.join(res_lines))


def main():
    assert os.path.exists(scores_dir), "%s doesn't exist!" % scores_dir
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    paths = glob(os.path.join(scores_dir, "*"))
    pool = multiprocessing.Pool()

    res = list(tqdm.tqdm(pool.imap(gen_res, paths), total=len(paths)))

if __name__ == '__main__':
    main()