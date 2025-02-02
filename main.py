import sys
import logging
import argparse

from config import Config

######## ARG PARSE ########

parser = argparse.ArgumentParser(description='arguments')

args = parser.parse_args()

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
    print('main()')


if __name__ == '__main__':
    main()
