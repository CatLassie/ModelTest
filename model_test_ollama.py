import os
import sys
import logging
import argparse
import torch
from ollama import chat

from config import Config

######## ARG PARSE ########

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--model_name', '-m', type=str,
                    help='model to load', default='deepseek-r1:1.5b')
parser.add_argument('--input_file', '-f', type=str, help='input file')
parser.add_argument('--input_sequence', '-s', type=str, help='input sequence')
args = parser.parse_args()
model_name: str = args.model_name
input_file: str = args.input_file
input_sequence: str = args.input_sequence


######## CONFIG ########

conf = Config()

######## LOGGING ########

log_lvls = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}

lg = logging.getLogger(__name__)
handlers = [logging.StreamHandler(sys.stdout)]
logging.basicConfig(level=log_lvls[conf.log_lvl], handlers=handlers)

lg.info('\nSTART LOGGING\n')
lg.info(f'\nParsed command line arguments: {args}\n')


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


######## MAIN ########


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    input = None

    if input_file != None:
        input = read_file(input_file)

    if input_sequence != None:
        input = input_sequence

    if input == None:
        raise Exception("No input!")

    print('\nINPUT:\n', input)

    response = chat(
        model=model_name,
        messages=[
            {'role': 'user', 'content': input}
        ],
        options = {
            'temperature': 0.0,
            'top_p': 1.0
        }
    )

    output = response['message']['content']

    print('\nOUTPUT:\n', output)


if __name__ == '__main__':
    main()
