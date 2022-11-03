
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pydantic

from medcat.utils.regression.targeting import TranslationLayer

# TODO - add more comprohensive report
# i.e save to file and include information such as which name-cui pairs
# failed to work and/or which ones weren't found


class FailReason(Enum):
    CONCEPT_NOT_ANNOTATED = 1
    """The concept was not annotated by the model"""
    INCORRECT_SPAN_BIG = 2
    """The concept was a part of an annotation made by the model"""
    INCORRECT_SPAN_SMALL = 3
    """Only a part of the concept was annotated"""
    INCORRECT_CUI_NOT_FOUND = 4
    """The CUI was not found in the context database"""
    INCORRECT_CUI_FOUND_PARENT = 5
    """The CUI annotated was the parent of the concept"""
    INCORRECT_CUI_FOUND_CHILD = 6
    """The CUI annotated was a child of the concept"""
    INCORRECT_NAME_NOT_FOUND = 7
    """The name specified was not found in the context database"""
    UNKNOWN = -1
    """Unknown reason for failure"""

    @classmethod
    def get_reason_for(cls, cui: str, name: str, res: dict, translation: TranslationLayer) -> 'FailReason':
        fail_reason: FailReason = FailReason.UNKNOWN  # should never remain unknown
        if cui not in translation.cui2names:
            fail_reason = FailReason.INCORRECT_CUI_NOT_FOUND
        elif name not in translation.name2cuis:
            fail_reason = FailReason.INCORRECT_NAME_NOT_FOUND
        else:
            ents = res['entities']
            found_cuis = [ents[nr]['cui'] for nr in ents]
            found_child = translation.has_child_of(found_cuis, cui)
            found_parent = translation.has_parent_of(found_cuis, cui)
            if found_child:
                fail_reason = FailReason.INCORRECT_CUI_FOUND_CHILD
            elif found_parent:
                fail_reason = FailReason.INCORRECT_CUI_FOUND_PARENT
            else:
                found_names = [ents[nr]['source_value'].lower()
                               for nr in ents]
                name = name.lower()
                if any(name in found_name for found_name in found_names):
                    fail_reason = FailReason.INCORRECT_SPAN_BIG
                elif any(found_name in name for found_name in found_names):
                    fail_reason = FailReason.INCORRECT_SPAN_SMALL
                else:
                    fail_reason = FailReason.CONCEPT_NOT_ANNOTATED
        return fail_reason


class SingleResultDescriptor(pydantic.BaseModel):
    name: str
    """The name of the part that was checked"""
    success: int = 0
    """Number of successes"""
    fail: int = 0
    """Number of failures"""
    failures: List[Tuple[str, str, FailReason]] = []
    """The description of failures"""

    def report_success(self, cui: str, name: str, success: bool, fail_reason: Optional[FailReason]) -> None:
        """Report a test case and its successfulness

        Args:
            cui (str): The CUI being checked
            name (str): The name being checked
            success (bool): Whether or not the check was successful
            fail_reason (Optional[FailReason]): The reason for the failure (if applicable)
        """
        if success:
            self.success += 1
        else:
            self.fail += 1
            self.failures.append((cui, name, fail_reason))

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

    def report(self, cui: str, name: str, phrase: str, success: bool, fail_reason: Optional[FailReason]) -> None:
        """Report a test case and its successfulness

        Args:
            cui (str): The CUI being checked
            name (str): The name being checked
            phrase (str): The phrase being checked
            success (bool): Whether or not the check was successful
            fail_reason (Optional[FailReason]): The reason for the failure (if applicable)
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
        all_failures = []
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
            for _, _, ft in all_failures:
                if ft not in failure_types:
                    failure_types[ft] = 0
                failure_types[ft] += 1
            failures = '\nFailures:\n' + \
                '\n'.join(
                    [f'{ft}: {occurances}' for ft, occurances in failure_types.items()])
            failures += '\nDetailed:\n' + '\n'.join(
                [f'CUI: {repr(cui)}, name: {repr(name)}, reason: {reason}'
                 for cui, name, reason in all_failures])
        return f"""A total of {len(self.parts)} parts were kept track of within the group "{self.name}".
And a total of {total_total} (sub)cases were checked.{empty_text}
        Total success:  {total_s:10d} ({100 * total_s / total_total if total_total > 0 else 0}%)
        Total failure:  {total_f:10d} ({100 * total_f / total_total if total_total > 0 else 0}%)
        {delegated}{failures}"""
