from Constants import *
import sys

if __name__ == '__main__':
    input_file_name = sys.argv[1]
    with open(input_file_name) as f:
        all_lines = f.readlines()
    test_dir_path = all_lines[0].strip()
    question_ids = []
    for line in all_lines[1:]:
        question_ids.append(line.strip())