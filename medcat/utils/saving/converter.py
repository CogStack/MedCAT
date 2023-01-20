import argparse
import os

import logging

from medcat.cat import CAT


logger = logging.getLogger(__name__)


def run_conversion(modelpack: str, format: str, target: str, overwrite: bool) -> None:
    cat = CAT.load_model_pack(modelpack)
    cat.create_model_pack(os.path.dirname(
        target), os.path.basename(target), cdb_format=format)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelpack', help='The model pack to use',
                        type=str)
    parser.add_argument('format', help='The target format. '
                        'Either "dill" or "json" can be specified.', type=str)
    parser.add_argument('target', help='The target folder.', type=str)
    parser.add_argument('--silent', '-s', help='Make the operation silent (i.e ignore console output)',
                        action='store_true')
    parser.add_argument('--verbose', '-debug', help='Enable debug/verbose mode',
                        action='store_true')
    parser.add_argument('--overwrite', help='Whether to overwrite save file',
                        action='store_true')
    args = parser.parse_args()
    if not args.silent:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel('INFO')
    if args.verbose:
        from medcat.utils.saving import logger as saving_logger
        saving_logger.setLevel('DEBUG')
        saving_logger.addHandler(logging.StreamHandler())
    run_conversion(args.modelpack, args.format,
                   args.target, args.overwrite)


if __name__ == "__main__":
    main()
