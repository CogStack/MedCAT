from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, cast
import pydantic

from medcat.utils.regression.targeting import TranslationLayer


class FailReason(str, Enum):
    CONCEPT_NOT_ANNOTATED = 'CONCEPT_NOT_ANNOTATED'
    """The concept was not annotated by the model"""
    INCORRECT_CUI_FOUND = 'INCORRECT_CUI_FOUND'
    """A different CUI with the same name was found"""
    INCORRECT_SPAN_BIG = 'INCORRECT_SPAN_BIG'
    """The concept was a part of an annotation made by the model"""
    INCORRECT_SPAN_SMALL = 'INCORRECT_SPAN_SMALL'
    """Only a part of the concept was annotated"""
    CUI_NOT_FOUND = 'CUI_NOT_FOUND'
    """The CUI was not found in the context database"""
    CUI_PARENT_FOUND = 'CUI_PARENT_FOUND'
    """The CUI annotated was the parent of the concept"""
    CUI_CHILD_FOUND = 'CUI_CHILD_FOUND'
    """The CUI annotated was a child of the concept"""
    NAME_NOT_FOUND = 'NAME_NOT_FOUND'
    """The name specified was not found in the context database"""
    UNKNOWN = 'UNKNOWN'
    """Unknown reason for failure"""


class FailDescriptor(pydantic.BaseModel):
    cui: str
    name: str
    reason: FailReason
    extra: str = ''

    @classmethod
    def get_reason_for(cls, cui: str, name: str, res: dict, translation: TranslationLayer) -> 'FailDescriptor':
        """Get the fail reason for the failure of finding the specifeid CUI and name
        where the resulting entities are presented.

        Args:
            cui (str): The cui that was expected
            name (str): The name that was expected
            res (dict): The entities that were annotated
            translation (TranslationLayer): The translation layer

        Returns:
            FailDescriptor: The corresponding fail descriptor
        """
        def format_matching(matches: List[Tuple[str, str]]) -> str:
            return 'Found: ' + ', '.join(f'{mcui}|{mname}' for mcui, mname in matches)
        fail_reason: FailReason = FailReason.UNKNOWN  # should never remain unknown
        extra: str = ''
        if cui not in translation.cui2names:
            fail_reason = FailReason.CUI_NOT_FOUND
        elif name not in translation.name2cuis:
            fail_reason = FailReason.NAME_NOT_FOUND
            extra = f'Names for concept: {translation.cui2names[cui]}'
        else:
            ents = res['entities']
            found_cuis = [ents[nr]['cui'] for nr in ents]
            found_names = [ents[nr]['source_value'] for nr in ents]
            found_children = translation.get_children_of(found_cuis, cui)
            found_parents = translation.get_parents_of(found_cuis, cui)
            if found_children:
                fail_reason = FailReason.CUI_CHILD_FOUND
                w_name = [(ccui, found_names[found_cuis.index(ccui)])
                          for ccui in found_children]
                extra = format_matching(w_name)
            elif found_parents:
                fail_reason = FailReason.CUI_PARENT_FOUND
                w_name = [(ccui, found_names[found_cuis.index(ccui)])
                          for ccui in found_parents]
                extra = format_matching(w_name)
            else:
                found_cuis_names = list(zip(found_cuis, found_names))

                def get_matching(condition: Callable[[str, str], bool]):
                    return [(found_cui, found_name)
                            for found_cui, found_name in found_cuis_names
                            if condition(found_cui, found_name)]
                name = name.lower()
                same_names = get_matching(
                    lambda _, fname: fname.lower() == name)
                bigger_span = get_matching(
                    lambda _, fname: name in fname.lower())
                smaller_span = get_matching(
                    lambda _, fname: fname.lower() in name)
                if same_names:
                    extra = format_matching(same_names)
                    fail_reason = FailReason.INCORRECT_CUI_FOUND
                elif bigger_span:
                    extra = format_matching(bigger_span)
                    fail_reason = FailReason.INCORRECT_SPAN_BIG
                elif smaller_span:
                    extra = format_matching(smaller_span)
                    fail_reason = FailReason.INCORRECT_SPAN_SMALL
                else:
                    fail_reason = FailReason.CONCEPT_NOT_ANNOTATED
        return FailDescriptor(cui=cui, name=name, reason=fail_reason, extra=extra)


class SingleResultDescriptor(pydantic.BaseModel):
    name: str
    """The name of the part that was checked"""
    success: int = 0
    """Number of successes"""
    fail: int = 0
    """Number of failures"""
    failures: List[FailDescriptor] = []
    """The description of failures"""

    def report_success(self, cui: str, name: str, success: bool, fail_reason: Optional[FailDescriptor]) -> None:
        """Report a test case and its successfulness

        Args:
            cui (str): The CUI being checked
            name (str): The name being checked
            success (bool): Whether or not the check was successful
            fail_reason (Optional[FailDescriptor]): The reason for the failure (if applicable)
        """
        if success:
            self.success += 1
        else:
            self.fail += 1
            self.failures.append(cast(FailDescriptor, fail_reason))

    def get_report(self) -> str:
        """Get the report associated with this descriptor

        Returns:
            str: The report string
        """
        total = self.success + self.fail
        return f"""Tested "{self.name}" for a total of {total} cases:
        Success:    {self.success:10d} ({100 * self.success / total if total > 0 else 0}%)
        Failure:    {self.fail:10d} ({100 * self.fail / total if total > 0 else 0}%)"""


class ResultDescriptor(SingleResultDescriptor):
    per_phrase_results: Dict[str, SingleResultDescriptor] = {}

    def report(self, cui: str, name: str, phrase: str, success: bool, fail_reason: Optional[FailDescriptor]) -> None:
        """Report a test case and its successfulness

        Args:
            cui (str): The CUI being checked
            name (str): The name being checked
            phrase (str): The phrase being checked
            success (bool): Whether or not the check was successful
            fail_reason (Optional[FailDescriptor]): The reason for the failure (if applicable)
        """
        super().report_success(cui, name, success, fail_reason)
        if phrase not in self.per_phrase_results:
            self.per_phrase_results[phrase] = SingleResultDescriptor(
                name=phrase)
        self.per_phrase_results[phrase].report_success(
            cui, name, success, fail_reason)

    def get_report(self, phrases_separately: bool = False) -> str:
        """Get the report associated with this descriptor

        Args:
            phrases_separately (bool): Whether to output descriptor for each phrase separately

        Returns:
            str: The report string
        """
        sr = super().get_report()
        if not phrases_separately:
            return sr
        children = '\n'.join([srd.get_report()
                             for srd in self.per_phrase_results.values()])
        return sr + '\n\t\t' + children.replace('\n', '\n\t\t')


class MultiDescriptor(pydantic.BaseModel):
    name: str
    """The name of the collection being checked"""
    parts: List[ResultDescriptor] = []
    """The parts kept track of"""

    @property
    def success(self) -> int:
        """The total number of successes.
        """
        return sum(part.success for part in self.parts)

    @property
    def fail(self) -> int:
        """The total number of failures.
        """
        return sum(part.fail for part in self.parts)

    def get_report(self, phrases_separately: bool,
                   hide_empty: bool = False, show_failures: bool = True) -> str:
        """Get the report associated with this descriptor

        Args:
            phrases_separately (bool): Whether to include per-phrase information
            hide_empty (bool): Whether to hide empty cases
            show_failures (bool): Whether to show failures

        Returns:
            str: The report string
        """
        del_out = []  # delegation
        all_failures: List[FailDescriptor] = []
        total_s, total_f = 0, 0
        nr_of_empty = 0
        for part in self.parts:
            total_s += part.success
            total_f += part.fail
            if hide_empty and part.success == part.fail == 0:
                nr_of_empty += 1
                continue
            cur_add = '\t' + \
                part.get_report(phrases_separately=phrases_separately).replace(
                    '\n', '\n\t\t')
            del_out.append(cur_add)
            all_failures.extend(part.failures)
        total_total = total_s + total_f
        delegated = '\n\t'.join(del_out)
        empty_text = ''
        if hide_empty:
            empty_text = f' A total of {nr_of_empty} cases did not match any CUIs and/or names.'
        failures = ''
        if show_failures and all_failures:
            failure_types = {}
            for fd in all_failures:
                if fd.reason not in failure_types:
                    failure_types[fd.reason] = 0
                failure_types[fd.reason] += 1
            failures = '\nFailures:\n' + \
                '\n'.join(
                    [f'{ft}: {occurances}' for ft, occurances in failure_types.items()])
            failures += '\nDetailed:\n' + '\n'.join(
                [f'CUI: {repr(descriptor.cui)}, name: {repr(descriptor.name)}, '
                 f'reason: {descriptor.reason}{" (%s)"%descriptor.extra if descriptor.extra else ""}'
                 for descriptor in all_failures])
        return f"""A total of {len(self.parts)} parts were kept track of within the group "{self.name}".
And a total of {total_total} (sub)cases were checked.{empty_text}
        Total success:  {total_s:10d} ({100 * total_s / total_total if total_total > 0 else 0}%)
        Total failure:  {total_f:10d} ({100 * total_f / total_total if total_total > 0 else 0}%)
        {delegated}{failures}"""
