import argparse
import json


def split_file(path, output_path):
    file_count = 0
    lines = []
    f = open(path, 'r')
    for line in f:
        parsed_line = json.loads(line)
        if len(parsed_line['section_names']) <= 3:
            continue

        lines.append(line)
        if len(lines) >= 67:
            open("%s_%d.txt" % (output_path, file_count), "w").write(''.join(lines))
            lines = []
            file_count += 1


def main():
    split_file('arxiv-release/train.txt', 'arxiv-release/chunked/train')


if __name__ == '__main__':
    main()