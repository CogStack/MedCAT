import argparse
import json
from pathlib import Path
import logging

from typing import Optional

from medcat.cat import CAT
from medcat.utils.regression.checking import RegressionSuite, TranslationLayer
from medcat.utils.regression.results import Strictness

logger = logging.getLogger(__name__)


DEFAULT_TEST_SUITE_PATH = Path('configs', 'default_regression_tests.yml')


def main(model_pack_dir: Path, test_suite_file: Path,
         phrases: bool = False, hide_empty: bool = False,
         examples_strictness_str: str = 'STRICTEST',
         jsonpath: Optional[Path] = None, overwrite: bool = False,
         jsonindent: Optional[int] = None,
         strictness_str: str = 'NORMAL',
         max_phrase_length: int = 80,
         use_mct_export: bool = False,
         mct_export_yaml_path: Optional[str] = None) -> None:
    """Check test suite against the specifeid model pack.

    Args:
        model_pack_dir (Path): The path to the model pack
        test_suite_file (Path): The path to the test suite YAML
        phrases (bool): Whether to show per-phrase information in a report
        hide_empty (bool): Whether to hide empty cases in a report
        examples_strictness_str (str): The example strictness string. Defaults to STRICTEST.
            NOTE: If you set this to 'None', examples wille be omitted.
        jsonpath (Optional[Path]): The json path to save the report to (if specified)
        overwrite (bool): Whether to overwrite the file if it exists. Defaults to False
        jsonindent (int): The indentation for json objects. Defaults to 0
        strictness_str (str): The strictness name. Defaults to NORMAL.
        max_phrase_length (int): The maximum phrase length in examples. Defualts to 80.
        use_mct_export (bool): Whether to use a MedCATtrainer export as input. Defaults to False.
        mct_export_yaml_path (str): The (optional) path the converted MCT export should be saved as YAML at.
            If not set (or None), the MCT export is not saved in YAML format. Defaults to None.

    Raises:
        ValueError: If unable to overwrite file or folder does not exist.
    """
    if jsonpath and jsonpath.exists() and not overwrite:
        # check before doing anything so as to not waste time on the tests
        raise ValueError(
            f'Unable to write to existing file {str(jsonpath)} pass --overwrite to overwrite the file')
    if jsonpath and not jsonpath.parent.exists():
        raise ValueError(
            f'Need to specify a file in an existing directory, folder not found: {str(jsonpath)}')
    logger.info('Loading RegressionChecker from yaml: %s', test_suite_file)
    if not use_mct_export:
        rc = RegressionSuite.from_yaml(str(test_suite_file))
    else:
        rc = RegressionSuite.from_mct_export(str(test_suite_file))
        if mct_export_yaml_path:
            logger.info('Writing MCT export in YAML to %s', str(mct_export_yaml_path))
            with open(mct_export_yaml_path, 'w') as f:
                f.write(rc.to_yaml())
    logger.info('Loading model pack from file: %s', model_pack_dir)
    cat: CAT = CAT.load_model_pack(str(model_pack_dir))
    logger.info('Checking the current status')
    res = rc.check_model(cat, TranslationLayer.from_CDB(cat.cdb))
    strictness = Strictness[strictness_str]
    if jsonpath:
        logger.info('Writing to %s', str(jsonpath))
        jsonpath.write_text(json.dumps(res.dict(), indent=jsonindent))
    else:
        if examples_strictness_str in ("None", "N/A"):
            examples_strictness = None
        else:
            examples_strictness = Strictness[examples_strictness_str]
        logger.info(res.get_report(phrases_separately=phrases,
                    hide_empty=hide_empty, examples_strictness=examples_strictness,
                    strictness=strictness, phrase_max_len=max_phrase_length))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('modelpack', help='The model pack against which to check',
                        type=Path)
    parser.add_argument('test_suite', help='YAML formatted file containing the regression test suite'
                        f'The default value (and exampe) is at `{DEFAULT_TEST_SUITE_PATH}`',
                        default=DEFAULT_TEST_SUITE_PATH, nargs='?', type=Path)
    parser.add_argument('--silent', '-s', help='Make the operation silent (i.e ignore console output)',
                        action='store_true')
    parser.add_argument('--verbose', '-debug', help='Enable debug/verbose mode',
                        action='store_true')
    parser.add_argument('--phrases', '-p', help='Include per-phrase information in report',
                        action='store_true')
    parser.add_argument('--noempty', help='Hide empty cases in report',
                        action='store_true')
    parser.add_argument('--example-strictness', help='The strictness of examples. Set to None to disable. '
                        'This defaults to STRICTEST to show all non-identical examples. ',
                        choices=[strictness.name for strictness in Strictness] + ["None"],
                        default=Strictness.STRICTEST.name)
    parser.add_argument('--jsonfile', help='Save report to a json file',
                        type=Path)
    parser.add_argument('--overwrite', help='Whether to overwrite save file',
                        action='store_true')
    parser.add_argument('--jsonindent', help='The json indent',
                        type=int, default=None)
    parser.add_argument('--strictness', help='The strictness to consider success.',
                        choices=[strictness.name for strictness in Strictness],
                        default=Strictness.NORMAL.name)
    parser.add_argument('--max-phrase-length', help='The maximum phrase length in examples.',
                        type=int, default=80)
    parser.add_argument('--from-mct-export', help='Whether to load the regression suite from '
                        'a MedCATtrainer export (.json) instead of a YAML format (default).',
                        action='store_true')
    parser.add_argument('--mct-export-yaml', help='The YAML file path to safe a convert MCT '
                        'export as. Only useful alongside `--from-mct-export` option and an '
                        'MCT export passed as the test suite.',
                        type=str, default=None)
    args = parser.parse_args()
    if not args.silent:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel('INFO')
    if args.verbose:
        from medcat.utils.regression import logger as regr_logger
        regr_logger.setLevel('DEBUG')
        regr_logger.addHandler(logging.StreamHandler())
    main(args.modelpack, args.test_suite,
         phrases=args.phrases, hide_empty=args.noempty, examples_strictness_str=args.example_strictness,
         jsonpath=args.jsonfile, overwrite=args.overwrite, jsonindent=args.jsonindent,
         strictness_str=args.strictness, max_phrase_length=args.max_phrase_length,
         use_mct_export=args.from_mct_export, mct_export_yaml_path=args.mct_export_yaml)
