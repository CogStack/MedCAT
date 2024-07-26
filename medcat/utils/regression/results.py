from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set
import pydantic

from medcat.utils.regression.targeting import TranslationLayer


class Finding(Enum):
    # same CUIs
    IDENTICAL = auto()
    BIGGER_SPAN_RIGHT = auto()
    BIGGER_SPAN_LEFT = auto()
    BIGGER_SPAN_BOTH = auto()
    SMALLER_SPAN = auto()
    SPAN_OVERLAP = auto()  # neither start NOR end match expectation, but there is some overlap
    # slightly different CUIs
    FOUND_DIR_PARENT = auto()
    FOUND_DIR_GRANDPARENT = auto()
    FOUND_ANY_CHILD = auto()
    # TODO - anything else?
    FAIL = auto()

    @classmethod
    def determine(cls, exp_cui: str, exp_start: int, exp_end: int,
                  tl: TranslationLayer, found_entities: Dict[str, Dict[str, Any]],
                  strict_only: bool = False,
                  check_children: bool = True, check_parent: bool = True,
                  check_grandparent: bool = True
                  ) -> 'Finding':
        return FindingDeterminer(exp_cui, exp_start, exp_end,
                                 tl, found_entities, strict_only,
                                 check_parent, check_grandparent, check_children).determine()


class FindingDeterminer:

    def __init__(self, exp_cui: str, exp_start: int, exp_end: int,
                    tl: TranslationLayer, found_entities: Dict[str, Dict[str, Any]],
                    strict_only: bool = False,
                    check_children: bool = True, check_parent: bool = True,
                    check_grandparent: bool = True,) -> None:
        self.exp_cui = exp_cui
        self.exp_start = exp_start
        self.exp_end = exp_end
        self.tl = tl
        self.found_entities = found_entities
        self.strict_only = strict_only
        self.check_children = check_children
        self.check_parent = check_parent
        self.check_grandparent = check_grandparent
        # helper for children to avoid infinite recursion
        self._checked_children: Set[str] = set()

    def _determine_raw(self, start: int, end: int) -> Optional[Finding]:
        if end < start:
            raise MalformedFinding(f"The end ({end}) is smaller than the start ({start})")
        elif self.exp_end < self.exp_start:
            raise MalformedFinding(f"The expected end ({self.exp_end}) is "
                                   f"smaller than the expected start ({self.exp_start})")
        if self.strict_only:
            if start == self.exp_start and end == self.exp_end:
                return Finding.IDENTICAL
            return None
        if start < self.exp_start:
            if end < self.exp_start:
                return None
            elif end < self.exp_end:
                return Finding.SPAN_OVERLAP  # TODO - distinguish[overlap]?
            elif end == self.exp_end:
                return Finding.BIGGER_SPAN_LEFT
            return Finding.BIGGER_SPAN_BOTH
        elif start == self.exp_start:
            if end < self.exp_end:
                return Finding.SMALLER_SPAN # TODO - distinguish[smaller]?
            elif end == self.exp_end:
                return Finding.IDENTICAL
            return Finding.BIGGER_SPAN_RIGHT
        elif start > self.exp_start and start <= self.exp_end:
            if end < self.exp_end:
                return Finding.SMALLER_SPAN # TODO - distinguish[smaller]?
            elif end == self.exp_end:
                return Finding.SMALLER_SPAN # TODO - distinguish[smaller]?
            return Finding.SPAN_OVERLAP  # TODO - distinguish[overlap]?
        # if start > exp_end -> no match
        return None

    def _get_strict(self) -> Optional[Finding]:
        if not self.found_entities:
            return Finding.FAIL
        for entity in self.found_entities.values():
            start, end, cui = entity['start'], entity['end'], entity['cui']
            if cui == self.exp_cui:
                raw_find = self._determine_raw(start, end)
                if raw_find:
                    return raw_find
        if self.strict_only:
            return Finding.FAIL
        return None

    def _check_parents(self) -> Optional['Finding']:
        parents = self.tl.get_direct_parents(self.exp_cui)
        for parent in parents:
            finding = Finding.determine(parent, self.exp_start, self.exp_end,
                                        self.tl,
                                        self.found_entities,
                                        check_children=False,
                                        check_parent=self.check_grandparent,
                                        check_grandparent=False)
            if finding is Finding.IDENTICAL:
                return Finding.FOUND_DIR_PARENT
            if finding is Finding.FOUND_DIR_PARENT:
                return Finding.FOUND_DIR_GRANDPARENT
        return None

    def _check_children(self) -> Optional['Finding']:
        children = self.tl.get_direct_children(self.exp_cui)
        for child in children:
            finding = Finding.determine(child, self.exp_start, self.exp_end,
                                        self.tl,
                                        self.found_entities,
                                        check_children=True,
                                        check_parent=False,
                                        check_grandparent=False)
            if finding in (Finding.IDENTICAL, Finding.FOUND_ANY_CHILD):
                return Finding.FOUND_ANY_CHILD
            self._checked_children.add(child)
        return None

    def determine(self) -> 'Finding':
        finding = self._get_strict()
        if finding is not None:
            return finding
        if self.check_parent:
            finding = self._check_parents()
            if finding is not None:
                return finding
        if self.check_children:
            self._checked_children.add(self.exp_cui)
            finding = self._check_children()
            if finding is not None:
                return finding
        return Finding.FAIL


class Strictness(Enum):
    STRICTEST = auto()
    STRICT = auto()
    NORMAL = auto()
    LAX = auto()


STRICTNESS_MATRIX: Dict[Strictness, Set[Finding]] = {
    Strictness.STRICTEST: {Finding.IDENTICAL},
    Strictness.STRICT: {Finding.IDENTICAL, Finding.FOUND_ANY_CHILD},
    Strictness.NORMAL: {
        Finding.IDENTICAL, Finding.FOUND_ANY_CHILD,
        Finding.BIGGER_SPAN_RIGHT, Finding.BIGGER_SPAN_LEFT,
        Finding.BIGGER_SPAN_BOTH,
        Finding.SMALLER_SPAN, Finding.SPAN_OVERLAP
    },
    Strictness.LAX: {
        Finding.IDENTICAL, Finding.FOUND_ANY_CHILD,
        Finding.BIGGER_SPAN_RIGHT, Finding.BIGGER_SPAN_LEFT,
        Finding.BIGGER_SPAN_BOTH,
        Finding.SMALLER_SPAN, Finding.SPAN_OVERLAP,
        Finding.FOUND_DIR_PARENT, Finding.FOUND_DIR_GRANDPARENT,
    }
}


class SingleResultDescriptor(pydantic.BaseModel):
    name: str
    """The name of the part that was checked"""
    findings: Dict[Finding, int] = {}
    """The description of failures"""

    def report_success(self, cui: str, name: str, finding: Finding) -> None:
        """Report a test case and its successfulness

        Args:
            cui (str): The CUI being checked
            name (str): The name being checked
            finding (Finding): Whether or not the check was successful
        """
        if finding not in self.findings:
            self.findings[finding] = 0
        self.findings[finding] += 1

    def get_report(self) -> str:
        """Get the report associated with this descriptor

        Returns:
            str: The report string
        """
        total = sum(self.findings.values())
        ret_vals = [f"Tested '{self.name}' for a total of {total} cases:"]
        ret_vals.extend([
            f"{f.name:24s}:{self.findings[f]:10d} ({100 * self.findings[f] / total if total > 0 else 0}%)"
            # NOTE iterating over Finding so the order is the same as in the enum
            for f in Finding if f in self.findings
        ])
        return "\n".join(ret_vals)


class ResultDescriptor(SingleResultDescriptor):
    per_phrase_results: Dict[str, SingleResultDescriptor] = {}

    def report(self, cui: str, name: str, phrase: str, finding: Finding) -> None:
        """Report a test case and its successfulness

        Args:
            cui (str): The CUI being checked
            name (str): The name being checked
            phrase (str): The phrase being checked
            finding (Finding): To what extent the concept was recognised
        """
        super().report_success(cui, name, finding)
        if phrase not in self.per_phrase_results:
            self.per_phrase_results[phrase] = SingleResultDescriptor(
                name=phrase)
        self.per_phrase_results[phrase].report_success(
            cui, name, finding)

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
    def findings(self) -> Dict[Finding, int]:
        """The total findings.

        Returns:
            Dict[Finding, int]: The total number of sucesses.
        """
        totals: Dict[Finding, int] = {}
        for part in self.parts:
            for f, val in part.findings.items():
                if f not in totals:
                    totals[f] = val
                else:
                    totals[f] += val
        return totals

    def get_report(self, phrases_separately: bool,
                   hide_empty: bool = False, show_failures: bool = True,
                   strictness: Strictness = Strictness.NORMAL) -> str:
        """Get the report associated with this descriptor

        Args:
            phrases_separately (bool): Whether to include per-phrase information
            hide_empty (bool): Whether to hide empty cases
            show_failures (bool): Whether to show failures
            strictness (Strictness): The strictness of the success / fail overview.
                Defaults to Strictness.NORMAL.

        Returns:
            str: The report string
        """
        del_out = []  # delegation
        total_findings: Dict[Finding, int] = {}
        total_s, total_f = 0, 0
        allowed_findings = STRICTNESS_MATRIX[strictness]
        total_total = 0
        nr_of_empty = 0
        for part in self.parts:
            for f, val in part.findings.items():
                if f not in total_findings:
                    total_findings[f] = val
                else:
                    total_findings[f] += val
                total_total += val
                if f in allowed_findings:
                    total_s += val
                else:
                    total_f += val
            if hide_empty and len(part.findings) == 0:
                nr_of_empty += 1
                continue
            cur_add = '\t' + \
                part.get_report(phrases_separately=phrases_separately).replace(
                    '\n', '\n\t\t')
            del_out.append(cur_add)
        delegated = '\n\t'.join(del_out)
        empty_text = ''
        if hide_empty:
            empty_text = f' A total of {nr_of_empty} cases did not match any CUIs and/or names.'
        ret_vals = [f"""A total of {len(self.parts)} parts were kept track of within the group "{self.name}".
And a total of {total_total} (sub)cases were checked.{empty_text}"""]
        allowed_fingings_str = [f.name for f in allowed_findings]
        ret_vals.extend([
            f"At the strictness level of {strictness} (allowing {allowed_fingings_str}):",
            f"The number of total successful (sub) cases: {total_s} "
            f"({100 * total_s/total_total if total_total > 0 else 0:5.2f}%)",
            f"The number of total failing (sub) cases   : {total_f} "
            f"({100 * total_f/total_total if total_total > 0 else 0:5.2f}%"
        ])
        ret_vals.extend([
            f"{f.name:24s}:{self.findings[f]:10d} "
            f"({100 * self.findings[f] / total_total if total_total > 0 else 0:5.2f}%)"
            # NOTE iterating over Finding so the order is the same as in the enum
            for f in Finding if f in self.findings
        ])
        return "\n".join(ret_vals) + f"\n{delegated}"


class MalformedFinding(ValueError):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
