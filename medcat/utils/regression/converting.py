import json
import logging
from abc import ABC, abstractmethod
import re
from typing import List, Optional, Set
import tqdm

from medcat.utils.regression.checking import RegressionCase, RegressionChecker, MetaData
from medcat.utils.regression.results import ResultDescriptor
from medcat.utils.regression.targeting import FilterOptions, FilterStrategy, FilterType, TypedFilter


logger = logging.getLogger(__name__)


class ContextSelector(ABC):
    """Describes how the context of a concept is found.
    A sub-class should be used as this one has no implementation.
    """

    def _splitter(self, text: str) -> List[str]:
        text = re.sub(' +', ' ', text)  # remove duplicate spaces
        # remove 1-letter words that are not a valid character
        return [word for word in text.split() if (
            len(word) > 1 or re.match(r'\w', word))]

    def make_replace_safe(self, text: str) -> str:
        """Make the text replace-safe.
        That is, wrap all '%' as '%%' so that the `text % replacement` syntax
        can be used for an inserted part (and that part only).

        Args:
            text (str): The text to use

        Returns:
            str: The replace-safe text
        """
        return text.replace(r'%', r'%%')

    @abstractmethod
    def get_context(self, text: str, start: int, end: int, leave_concept: bool = False) -> str:
        """Get the context of a concept within a larger body of text.
        The concept is specifiedb by its start and end indices.

        Args:
            text (str): The larger text
            start (int): The starting index
            end (int): The ending index
            leave_concept (bool): Whether to leave the concept or replace it by '%s'. Defaults to False

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

    def get_context(self, text: str, start: int, end: int, leave_concept: bool = False) -> str:
        words_before = self._splitter(text[:start])
        words_after = self._splitter(text[end:])
        if leave_concept:
            concept = text[start:end]
        else:
            concept = '%s'
        before = ' '.join(words_before[-self.words_before:])
        before = self.make_replace_safe(before)
        after = ' '.join(words_after[:self.words_after])
        after = self.make_replace_safe(after)
        return f'{before} {concept} {after}'


class PerSentenceSelector(ContextSelector):
    """Context selector that selects a sentence as context.
    Sentences are said to end with either ".", "?" or "!".
    """
    stoppers = r'\.+|\?+|!+'

    def get_context(self, text: str, start: int, end: int, leave_concept: bool = False) -> str:
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
        if leave_concept:
            concept = text[start: end]
        else:
            concept = '%s'
        context_before = self.make_replace_safe(context_before)
        context_after = self.make_replace_safe(context_after)
        return (context_before + concept + context_after).strip()


class UniqueNamePreserver:
    """Used to preserver unique names in a set
    """

    def __init__(self) -> None:
        self.unique_names: Set[str] = set()

    def name2nrgen(self, name: str, nr: int) -> str:
        """The method to generate name and copy-number combinations.

        Args:
            name (str): The base name
            nr (int): The number of the copy

        Returns:
            str: The combined name
        """
        return f'{name}-{nr}'

    def get_unique_name(self, orig_name: str, dupe_nr: int = 0) -> str:
        """Get the unique name of dupe number (at least) as high as specified.

        Args:
            orig_name (str): The original / base name
            dupe_nr (int, optional): The number of the copy to start from. Defaults to 0.

        Returns:
            str: The unique name
        """
        if dupe_nr == 0:
            cur_name = orig_name
        else:
            cur_name = self.name2nrgen(orig_name, dupe_nr)
        if cur_name not in self.unique_names:
            self.unique_names.add(cur_name)
            return cur_name
        return self.get_unique_name(orig_name, dupe_nr + 1)


def get_matching_case(cases: List[RegressionCase], filters: List[TypedFilter]) -> Optional[RegressionCase]:
    """Get a case that matches a set of filters (if one exists) from within a list.

    Args:
        cases (List[RegressionCase]): The list to look in
        filters (List[TypedFilter]): The filters to compare to

    Returns:
        Optional[RegressionCase]: The regression case (if found) or None
    """
    for case in cases:
        if case.filters == filters:
            return case
    return None


def medcat_export_json_to_regression_yml(mct_export_file: str,
                                         cont_sel: ContextSelector = PerSentenceSelector(),
                                         model_card: Optional[dict] = None) -> str:
    """Extract regression test cases from a MedCATtrainer export yaml.
    This is done based on the context selector specified.

    Args:
        mct_export_file (str): The MCT export file path
        cont_sel (ContextSelector, optional): The context selector. Defaults to PerSentenceSelector().
        model_card (Optional[dict]): The optional model card for generating metadata

    Returns:
        str: Extracted regression cases in YAML form
    """
    with open(mct_export_file) as f:
        data = json.load(f)
    fo = FilterOptions(strategy=FilterStrategy.ALL, onlyprefnames=False)
    test_cases: List[RegressionCase] = []
    unique_names = UniqueNamePreserver()
    for project in tqdm.tqdm(data['projects']):
        proj_name = project['name']
        docs = project['documents']
        for doc in tqdm.tqdm(docs):
            text = doc['text']
            for ann in tqdm.tqdm(doc['annotations']):
                target_name = ann['value']
                target_cui = ann['cui']
                start, end = ann['start'], ann['end']
                in_text_name = text[start: end]
                if target_name != in_text_name:
                    logging.warn('Could not convert annotation since the text was not '
                                 f' equal to the name, ignoring:\n{ann}')
                    break
                name_filt = TypedFilter(type=FilterType.NAME,
                                        values=[target_name, ])
                cui_filt = TypedFilter(type=FilterType.CUI,
                                       values=[target_cui, ])
                context = cont_sel.get_context(text, start, end)
                phrase = context
                case_name = unique_names.get_unique_name(f'{proj_name.replace(" ", "-")}-'
                                                         f'{target_name.replace(" ", "~")}')
                cur_filters = [name_filt, cui_filt]
                added_to_existing = False
                for prev_rc in test_cases:
                    if prev_rc.filters == cur_filters:
                        prev_rc.phrases.append(phrase)
                        added_to_existing = True
                if not added_to_existing:
                    rc = RegressionCase(name=case_name, options=fo,
                                        filters=cur_filters, phrases=[
                                            phrase, ],
                                        report=ResultDescriptor(name=case_name))
                    test_cases.append(rc)
    if model_card:
        metadata = MetaData.from_modelcard(model_card)
    else:
        metadata = MetaData.unknown()
    checker = RegressionChecker(cases=test_cases, metadata=metadata)
    return checker.to_yaml()
