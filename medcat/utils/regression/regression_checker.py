
import argparse
from pathlib import Path
import logging
import time

from sys import exit as sys_exit
from typing import Optional

from medcat.cat import CAT
from checking import RegressionChecker, TranslationLayer

logger = logging.getLogger(__name__)


def main(model_pack_dir: Path, test_suite_file: Path, max_failures=0, total: Optional[int] = None) -> None:
    """Check test suite against the specifeid model pack.

    Args:
        model_pack_dir (Path): The path to the model pack
        test_suite_file (Path): The path to the test suite YAML
    """
    logger.info('Loading RegressionChecker from yaml: %s', test_suite_file)
    rc = RegressionChecker.from_yaml(test_suite_file)
    logger.info('Loading model pack from file: %s', model_pack_dir)
    cat: CAT = CAT.load_model_pack(str(model_pack_dir))
    logger.info('Checking the current status')
    st = time.time()
    # TODO - use total
    s, f = rc.check_model(cat, TranslationLayer.from_CDB(cat.cdb), total=total)
    logger.info('Checking took %s seconds', time.time() - st)
    logger.info('\tSuccessful:\t%d\n\tFailed:\t\t%d', s, f)
    if f > max_failures:
        logger.warn('Found too many failures (%s)', f)
        sys_exit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modelpack', help='The model pack against which to check',
                        type=Path)
    parser.add_argument('test_suite', help='YAML formatted file containing the regression test suite'
                        'The default value (and exampe) is at `configs/default_regression_tests.yml`',
                        default=Path(
                            'configs', 'default_regression_tests.yml'),
                        nargs='?',
                        type=Path)
    parser.add_argument('--maxfail', '--mf', help='Number of maximum failures allowed (defaults to 0)',
                        type=int, default='0')
    parser.add_argument('--silent', '-s', help='Make the operation silent (i.e ignore console output)',
                        action='store_true')
    parser.add_argument('--verbose', '-debug', help='Enable debug/verbose mode',
                        action='store_true')
    parser.add_argument('--total', '-t', help='Set the total number of (sub)cases that will be tested.'
                        'This will enable using a progress bar', type=int, default=None)
    args = parser.parse_args()
    if not args.silent:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel('INFO')
    if args.verbose:
        from checking import logger as checking_logger
        checking_logger.addHandler(logging.StreamHandler())
        checking_logger.setLevel('DEBUG')
    main(args.modelpack, args.test_suite,
         max_failures=args.maxfail, total=args.total)
