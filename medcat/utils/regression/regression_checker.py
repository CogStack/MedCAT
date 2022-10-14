
import argparse
from pathlib import Path
import logging
import time

from sys import exit as sys_exit
from typing import Optional

from medcat.cat import CAT
from medcat.utils.regression.checking import RegressionChecker, TranslationLayer
from medcat.utils.regression.results import MultiDescriptor

logger = logging.getLogger(__name__)


def main(model_pack_dir: Path, test_suite_file: Path, max_failures=0,
         total: Optional[int] = None, report: bool = False,
         phrases: bool = False) -> None:
    """Check test suite against the specifeid model pack.

    Args:
        model_pack_dir (Path): The path to the model pack
        test_suite_file (Path): The path to the test suite YAML
        total (Optional[int]): The total number of (sub)cases to be tested (for progress bar)
        report (bool): Whether to use a more comprehensive report
        phrases (bool): Whether to show per-phrase information in a report
    """
    logger.info('Loading RegressionChecker from yaml: %s', test_suite_file)
    rc = RegressionChecker.from_yaml(str(test_suite_file), report=report)
    logger.info('Loading model pack from file: %s', model_pack_dir)
    cat: CAT = CAT.load_model_pack(str(model_pack_dir))
    logger.info('Checking the current status')
    st = time.time()
    # s, f = rc.check_model(cat, TranslationLayer.from_CDB(cat.cdb), total=total)
    res = rc.check_model(cat, TranslationLayer.from_CDB(cat.cdb), total=total)
    if report and isinstance(res, MultiDescriptor):
        logger.info(res.get_report(phrases_separately=phrases))
        return
    s, f = res  # tuple
    logger.info('Checking took %s seconds', time.time() - st)
    logger.info('\tSuccessful:\t%d\n\tFailed:\t\t%d', s, f)
    if f > max_failures:
        logger.warning('Found too many failures (%s)', f)
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
    parser.add_argument('--total', '-t', help='Set the total number of (sub)cases that will be tested. '
                        'This will enable using a progress bar. '
                        'If unknown, a large-ish number might still be beneficial to show progress.', type=int, default=None)
    parser.add_argument('--report', help='Show a more comprehensive report instead of a simple '
                        'total of success and failures', action='store_true')
    parser.add_argument('--phrases', '-p', help='Include per-phrase information in report '
                        '(only applicable if --report was passed)', action='store_true')
    args = parser.parse_args()
    if not args.silent:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel('INFO')
    if args.verbose:
        from checking import logger as checking_logger
        checking_logger.addHandler(logging.StreamHandler())
        checking_logger.setLevel('DEBUG')
    main(args.modelpack, args.test_suite,
         max_failures=args.maxfail, total=args.total, report=args.report,
         phrases=args.phrases)
