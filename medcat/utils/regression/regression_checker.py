
import argparse
from pathlib import Path
import logging

from typing import Optional

from medcat.cat import CAT
from medcat.utils.regression.checking import RegressionChecker, TranslationLayer

logger = logging.getLogger(__name__)


def main(model_pack_dir: Path, test_suite_file: Path,
         total: Optional[int] = None,
         phrases: bool = False, hide_empty: bool = False,
         hide_failures: bool = False) -> None:
    """Check test suite against the specifeid model pack.

    Args:
        model_pack_dir (Path): The path to the model pack
        test_suite_file (Path): The path to the test suite YAML
        total (Optional[int]): The total number of (sub)cases to be tested (for progress bar)
        phrases (bool): Whether to show per-phrase information in a report
        hide_empty (bool): Whether to hide empty cases in a report
        hide_failures (bool): Whether to hide failures in a report
    """
    logger.info('Loading RegressionChecker from yaml: %s', test_suite_file)
    rc = RegressionChecker.from_yaml(str(test_suite_file))
    logger.info('Loading model pack from file: %s', model_pack_dir)
    cat: CAT = CAT.load_model_pack(str(model_pack_dir))
    logger.info('Checking the current status')
    res = rc.check_model(cat, TranslationLayer.from_CDB(cat.cdb), total=total)
    logger.info(res.get_report(phrases_separately=phrases,
                hide_empty=hide_empty, show_failures=not hide_failures))


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
    parser.add_argument('--silent', '-s', help='Make the operation silent (i.e ignore console output)',
                        action='store_true')
    parser.add_argument('--verbose', '-debug', help='Enable debug/verbose mode',
                        action='store_true')
    parser.add_argument('--total', '-t', help='Set the total number of (sub)cases that will be tested. '
                        'This will enable using a progress bar. '
                        'If unknown, a large-ish number might still be beneficial to show progress.', type=int, default=None)
    parser.add_argument('--phrases', '-p', help='Include per-phrase information in report',
                        action='store_true')
    parser.add_argument('--noempty', help='Hide empty cases in report',
                        action='store_true')
    parser.add_argument('--hidefailures', help='Hide failed cases in report',
                        action='store_true')
    args = parser.parse_args()
    if not args.silent:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel('INFO')
    if args.verbose:
        from medcat.utils.regression.checking import logger as checking_logger
        from medcat.utils.regression.converting import logger as converting_logger
        console = logging.StreamHandler()
        checking_logger.addHandler(console)
        checking_logger.setLevel('DEBUG')
        converting_logger.addHandler(console)
        converting_logger.setLevel('DEBUG')
    main(args.modelpack, args.test_suite, total=args.total,
         phrases=args.phrases, hide_empty=args.noempty, hide_failures=args.hidefailures)
