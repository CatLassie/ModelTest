import os
import sys
import logging
import argparse
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, Qwen2ForCausalLM

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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device', device)

    tokenizer = AutoTokenizer.from_pretrained(model_full_path)
    # model = AutoModel.from_pretrained(model_full_path)
    # model = AutoModelForSequenceClassification.from_pretrained(model_full_path)
    model = AutoModelForCausalLM.from_pretrained(model_full_path)


    model.to(device)

    print('\nINPUT:\n', sequence)
    model_input = tokenizer(sequence, return_tensors='pt').to(device)
    print('\ninput sequence type and length:', type(sequence), len(sequence))
    print('model_input ids shape:', model_input["input_ids"].shape)
    # decoded_input = tokenizer.decode(model_input["input_ids"][0], skip_special_tokens=True)
    # print('decoded_input:', decoded_input)

    # model_output = model(**model_input)

    with torch.no_grad():
        model_output = model.generate(**model_input)
    print('model_output shape:', model_output.shape)
    # print('output', model_output.logits)

    output = tokenizer.decode(model_output[0], skip_special_tokens=True)
    print("output type and length:", type(output), len(output))
    print('\nOUTPUT:\n', output)


if __name__ == '__main__':
    main()
