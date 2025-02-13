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
parser.add_argument('--input_sequence', '-s', type=str,
                    help='input sequence', default='Hello there!')
args = parser.parse_args()
model_name: str = args.model_name
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


######## MAIN ########


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)

    print('\nINPUT:\n', input_sequence)

    response = chat(
        model=model_name,
        messages=[
            {'role': 'user', 'content': input_sequence}
        ]
    )

    output_sequence = response['message']['content']

    print('\nOUTPUT:\n', output_sequence)


if __name__ == '__main__':
    main()
