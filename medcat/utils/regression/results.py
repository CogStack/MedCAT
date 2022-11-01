
from typing import Dict, List
import pydantic


class SingleResultDescriptor(pydantic.BaseModel):
    name: str
    """The name of the part that was checked"""
    success: int = 0
    """Number of successes"""
    fail: int = 0
    """Number of failures"""

    def report_success(self, success: bool) -> None:
        """Report a test case and its successfulness

        Args:
            success (bool): Whether or not the check was successful
        """
        if success:
            self.success += 1
        else:
            self.fail += 1

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

    def report(self, cui: str, name: str, phrase: str, success: bool) -> None:
        """Report a test case and its successfulness

        Args:
            cui (str): The CUI being checked
            name (str): The name being checked
            phrase (str): The phrase being checked
            success (bool): Whether or not the check was successful
        """
        super().report_success(success)
        if phrase not in self.per_phrase_results:
            self.per_phrase_results[phrase] = SingleResultDescriptor(
                name=phrase)
        self.per_phrase_results[phrase].report_success(success)

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

    def get_report(self, phrases_separately: bool, hide_empty: bool=False) -> str:
        """Get the report associated with this descriptor

        Args:
            phrases_separately (bool): Whether to include per-phrase information
            hide_empty (bool): Whether to hide empty cases

        Returns:
            str: The report string
        """
        del_out = []  # delegation
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
        total_total = total_s + total_f
        delegated = '\n\t'.join(del_out)
        empty_text = ''
        if hide_empty:
            empty_text = f' A total of {nr_of_empty} cases did not match any CUIs and/or names.'
        return f"""A total of {len(self.parts)} parts were kept track of within the group "{self.name}".
And a total of {total_total} (sub)cases were checked.{empty_text}
        Total success:  {total_s:10d} ({100 * total_s / total_total if total_total > 0 else 0}%)
        Total failure:  {total_f:10d} ({100 * total_f / total_total if total_total > 0 else 0}%)
        {delegated}"""
