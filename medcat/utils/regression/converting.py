
import json
import logging
import re
from typing import List
import tqdm

from medcat.utils.regression.checking import RegressionCase, RegressionChecker
from medcat.utils.regression.targeting import FilterOptions, FilterStrategy, FilterType, TypedFilter


logger = logging.getLogger(__name__)


class ContextSelector:

    def _splitter(self, text: str) -> List[str]:
        text = re.sub(' +', ' ', text)  # remove duplicate spaces
        # remove 1-letter words that are not a valid character
        return [word for word in text.split() if (
            len(word) > 1 or re.match('\w', word))]

    def get_context(self, text: str, start: int, end: int) -> str:
        pass  # should be overwritten by subclass


class PerWordContextSelector(ContextSelector):

    def __init__(self, words_before: int, words_after: int) -> None:
        self.words_before = words_before
        self.words_after = words_after

    def get_context(self, text: str, start: int, end: int) -> str:
        words_before = self._splitter(text[:start])
        words_after = self._splitter(text[end:])
        concept = text[start:end]
        # TODO - better joining?
        return ' '.join(words_before[-self.words_before:] + [concept] + words_after[:self.words_after])


class PerSentenceSelector(ContextSelector):
    stoppers = '\.+|\?+|!+'

    def get_context(self, text: str, start: int, end: int) -> str:
        text_before = text[:start]
        r_last_stopper = re.search(self.stoppers, text_before[::-1])
        if r_last_stopper:
            last_stopper = len(text_before) - r_last_stopper.start()
            context_before = text_before[last_stopper:]
        else:  # concept in first sentence
            context_before = text_before
        text_after = text[end:]
        first_stopper = re.search(self.stoppers, text_after)
        if first_stopper:
            context_after = text_after[:first_stopper.start()]
        else:  # concept in last sentence
            context_after = text_after
        concept = text[start: end]
        return (context_before + concept + context_after).strip()


def medcat_export_json_to_regression_yml(mct_export_file: str, ) -> str:
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
