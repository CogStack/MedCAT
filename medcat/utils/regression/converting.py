
import json
import logging
import tqdm

from medcat.utils.regression.checking import RegressionCase, RegressionChecker
from medcat.utils.regression.targeting import FilterOptions, FilterStrategy, FilterType, TypedFilter


logger = logging.getLogger(__name__)


def medcat_export_json_to_regression_yml(mct_export_file: str) -> str:
    with open(mct_export_file, 'r') as f:
        data = json.load(f)
    test_cases = []
    for project in tqdm.tqdm(data['projects']):
        name = project['name']
        docs = project['documents']
        for doc in tqdm.tqdm(docs):
            text = doc['text']
            for ann in tqdm.tqdm(doc['annotations']):
                target_name = ann['value']
                start, end = ann['start'], ann['end']
                in_text_name = text[start: end]
                if target_name != in_text_name:
                    logging.warn('Could not convert annotation since the text was not '
                                 f' equal to the name, ignoring:\n{ann}')
                    break
                fo = FilterOptions(
                    strategy=FilterStrategy.ANY, onlyprefnames=False)
                filt = TypedFilter(type=FilterType.NAME,
                                   values=[target_name, ])
                phrase = text[:start] + '%s' + text[end:]
                rc = RegressionCase(name=f'{name.replace(" ", "-").replace(" ", "-")}-'
                                    f'{target_name.replace(" ", "-")}', options=fo, filters=[filt, ], phrases=[phrase, ])
                test_cases.append(rc)
    checker = RegressionChecker(cases=test_cases)
    return checker.to_yaml()
