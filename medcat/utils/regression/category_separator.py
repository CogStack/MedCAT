import argparse
import logging
from pathlib import Path

from medcat.utils.regression.categoryseparation import separate_categories, StrategyType


logger = logging.getLogger(__name__)


def _prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # category_yaml: str, strategy_type: StrategyType,
    # regression_suite_yaml: str, target_file_prefix: str, overwrite: bool = False
    parser.add_argument(
        'categories', help='The categories YAML file', type=Path)
    parser.add_argument('regressionsuite',
                        help='The regression suite YAML file', type=Path)
    parser.add_argument(
        'targetprefix', help='The target YAML file prefix', type=Path)
    parser.add_argument(
        '--strategy', help='The strategy to be used for separation (FIRST or ALL)',
        default='ALL', type=str)
    parser.add_argument('--silent', '-s', help='Make the operation silent (i.e ignore console output)',
                        action='store_true')
    parser.add_argument('--verbose', '-debug', help='Enable debug/verbose mode',
                        action='store_true')
    parser.add_argument(
        '--overwrite', help='Overwrite the target file if it exists', action='store_true')
    return parser.parse_args()


def main():
    args = _prepare_args()
    if not args.silent:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel('INFO')
    if args.verbose:
        from categoryseparation import logger as checking_logger
        checking_logger.addHandler(logging.StreamHandler())
        checking_logger.setLevel('DEBUG')
    strategy = StrategyType[args.strategy.upper()]
    separate_categories(args.categories, strategy, args.regressionsuite,
                        args.targetprefix, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
