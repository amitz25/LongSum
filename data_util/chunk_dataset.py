import argparse


def split_file(path, output_path):
    file_count = 0
    lines = []
    f = open(path, 'r')
    for line in f:
        lines.append(line)
        if len(lines) >= 67:
            open("%s_%d.txt" % (output_path, file_count), "w").write(''.join(lines))
            lines = []
            file_count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    split_file(args.input_path, args.output_path)


if __name__ == '__main__':
    main()