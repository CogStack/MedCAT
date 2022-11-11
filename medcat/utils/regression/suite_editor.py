import argparse
import logging
from pathlib import Path
from typing import Optional

from medcat.utils.regression.editing import combine_yamls


logger = logging.getLogger(__name__)


def main(base_file: str, add_file: str, new_file: Optional[str] = None,
         ignore_identicals: bool = True) -> None:
    logger.info(
        "Starting to add to %s from %s", base_file, add_file)
    res_file = combine_yamls(base_file, add_file, new_file=new_file,
                             ignore_identicals=ignore_identicals)
    logger.debug("Combination successful")
    logger.info("Saved combined data to %s", res_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'file', help='The base regression YAML file', type=Path)
    parser.add_argument(
        'add_file', help='The additional regression YAML file', type=Path)
    parser.add_argument('--newfile', help='The target file for the combination '
                        '(otherwise, the base file is used)', type=Path, required=False)
    parser.add_argument('--silent', '-s', help='Make the operation silent (i.e ignore console output)',
                        action='store_true')
    parser.add_argument('--verbose', '-debug', help='Enable debug/verbose mode',
                        action='store_true')
    parser.add_argument(
        '--include-identicals',
        help='Write down identical cases (they are only written down once by default)',
        action='store_true')
    args = parser.parse_args()
    if not args.silent:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel('INFO')
    if args.verbose:
        from checking import logger as checking_logger
        checking_logger.addHandler(logging.StreamHandler())
        checking_logger.setLevel('DEBUG')
    main(args.file, args.add_file, new_file=args.newfile,
         ignore_identicals=not args.include_identicals)
