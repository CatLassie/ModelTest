import os
import sys
import logging
import argparse
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

from config import Config

######## ARG PARSE ########

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument("--model_name", "-m", type=str,
                    required=True, help='model to load')
parser.add_argument("--sequence", "-s", type=str,
                    required=True, help='input sequence')
args = parser.parse_args()
model_name: str = args.model_name
sequence: str = args.sequence


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
    model_full_path = os.path.join(conf.model_base_path, model_name)
    if not os.path.exists(model_full_path):
        raise Exception("Model does not exist!")

    tokenizer = AutoTokenizer.from_pretrained(model_full_path)
    model = AutoModel.from_pretrained(model_full_path)
    # model = AutoModelForSequenceClassification.from_pretrained(model_full_path)

    model_input = tokenizer(sequence, return_tensors='pt')
    print('input', model_input)

    model_output = model(**model_input)
    print('output', model_output)
    # print('output', model_output.logits)


if __name__ == '__main__':
    main()
