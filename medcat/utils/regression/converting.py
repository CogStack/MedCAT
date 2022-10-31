
import json
import logging
import re
from typing import List
import tqdm

from medcat.utils.regression.checking import RegressionCase, RegressionChecker
from medcat.utils.regression.targeting import FilterOptions, FilterStrategy, FilterType, TypedFilter


logger = logging.getLogger(__name__)


class ContextSelector:
    """Describes how the context of a concept is found.
    A sub-class should be used as this one has no implementation.
    """

    def _splitter(self, text: str) -> List[str]:
        text = re.sub(' +', ' ', text)  # remove duplicate spaces
        # remove 1-letter words that are not a valid character
        return [word for word in text.split() if (
            len(word) > 1 or re.match('\w', word))]

    def get_context(self, text: str, start: int, end: int) -> str:
        """Get the context of a concept within a larger body of text.
        The concept is specifiedb by its start and end indices.

        Args:
            text (str): The larger text
            start (int): The starting index
            end (int): The ending index

        Returns:
            str: The select contexts
        """
        pass  # should be overwritten by subclass


class PerWordContextSelector(ContextSelector):
    """Context selector that selects a number of words
    from either side of the concept, regardless of punctuation.

    Args:
        words_before (int): Number of words to select from before concept
        words_after (int): Number of words to select from after concepts
    """

    def __init__(self, words_before: int, words_after: int) -> None:
        """_summary_

        """
        self.words_before = words_before
        self.words_after = words_after

    def get_context(self, text: str, start: int, end: int) -> str:
        words_before = self._splitter(text[:start])
        words_after = self._splitter(text[end:])
        concept = text[start:end]
        # TODO - better joining?
        return ' '.join(words_before[-self.words_before:] + [concept] + words_after[:self.words_after])


class PerSentenceSelector(ContextSelector):
    """Context selector that selects a sentence as context.
    Sentences are said to end with either ".", "?" or "!".
    """
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


def medcat_export_json_to_regression_yml(mct_export_file: str,
                                         cont_sel: ContextSelector = PerSentenceSelector()) -> str:
    """Extract regression test cases from a MedCATtrainer export yaml.
    This is done based on the context selector specified.

    Args:
        mct_export_file (str): The MCT export file path
        cont_sel (ContextSelector, optional): The context selector. Defaults to PerSentenceSelector().

    Returns:
        str: _description_
    """
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
                context = cont_sel.get_context(text, start, end)
                phrase = context.replace(target_name, '%')
                rc = RegressionCase(name=f'{name.replace(" ", "-").replace(" ", "-")}-'
                                    f'{target_name.replace(" ", "-")}', options=fo, filters=[filt, ], phrases=[phrase, ])
                test_cases.append(rc)
    checker = RegressionChecker(cases=test_cases)
    return checker.to_yaml()
