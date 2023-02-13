import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional
import json

from medcat.cat import CAT
from medcat.utils.regression.converting import ContextSelector, PerSentenceSelector, PerWordContextSelector, medcat_export_json_to_regression_yml


logger = logging.getLogger(__name__)


def get_model_card_from_file(model_card_file: str) -> dict:
    with open(model_card_file) as f:
        return json.load(f)


def get_model_card_from_model(model_zip: str) -> dict:
    logger.info(f"Loading model from {model_zip} to find the model card - this may take a while")
    cat = CAT.load_model_pack(model_zip)
    return cat.get_model_card(as_dict=True)


def main(mct_export: str, target: str, overwrite: bool = False,
         words: Optional[List[int]] = None, model_card_file: Optional[str] = None,
         model_file: Optional[str] = None) -> None:
    if not overwrite and os.path.isfile(target):
        raise ValueError("Not able to overwrite an existingfile, "
                         "pass '--overwrite' to force an overwrite")
    logger.info(
        "Starting to convert export JSON to YAML from file %s", mct_export)
    cont_sel: ContextSelector
    if not words:
        cont_sel = PerSentenceSelector()
    else:
        cont_sel = PerWordContextSelector(*words)
    if model_card_file:
        model_card = get_model_card_from_file(model_card_file)
    elif model_file:
        model_card = get_model_card_from_model(model_file)
    else:
        logger.warn("Creating regression suite with no model-card / metadata")
        logger.warn("Please consider passing --modelcard <model card json> or")
        logger.warn("--model <model zip> to find the model card associated with the regression suite")
        logger.warn("This will help better understand where and how the regression suite was generated")
        model_card = None
    yaml = medcat_export_json_to_regression_yml(mct_export, cont_sel=cont_sel, model_card=model_card)
    logger.debug("Conversion successful")
    logger.info("Saving writing data to %s", target)
    with open(target, 'w') as f:
        f.write(yaml)
    logger.debug("Done saving")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'file', help='The MedCATtrainer export file', type=Path)
    parser.add_argument('target', help='The Target YAML file', type=Path)
    parser.add_argument(
        '--modelcard', help='The ModelCard json file', type=Path)
    parser.add_argument(
        '--model', help='The Model to read model card from', type=Path)
    parser.add_argument('--silent', '-s', help='Make the operation silent (i.e ignore console output)',
                        action='store_true')
    parser.add_argument('--verbose', '-debug', help='Enable debug/verbose mode',
                        action='store_true')
    parser.add_argument(
        '--overwrite', help='Overwrite the target file if it exists', action='store_true')
    parser.add_argument(
        '--words', help='Select the number of words to select from before and after the concept',
        nargs=2, type=int)
    args = parser.parse_args()
    if not args.silent:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel('INFO')
    if args.verbose:
        from checking import logger as checking_logger
        checking_logger.addHandler(logging.StreamHandler())
        checking_logger.setLevel('DEBUG')
    main(args.file, args.target, overwrite=args.overwrite, words=args.words,
        model_card_file=args.modelcard, model_file=args.model)
