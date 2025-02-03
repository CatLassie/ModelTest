import os
import sys
import logging
import argparse
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

from config import Config

######## ARG PARSE ########

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument("--model_name", "-m", type=str,
                    required=True, help='model to save')
args = parser.parse_args()
model_name: str = args.model_name

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
    Path(model_full_path).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lg.info(f'\nTokenizer downloaded successfully')
    tokenizer.save_pretrained(model_full_path)
    lg.info(f'\nTokenizer saved successfully')

    model = AutoModel.from_pretrained(model_name)
    lg.info(f'\nModel downloaded successfully')
    model.save_pretrained(model_full_path)
    lg.info(f'\nModel saved successfully')


if __name__ == '__main__':
    main()
