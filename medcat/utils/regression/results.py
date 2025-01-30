from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Iterable, Tuple, cast
import json
import pydantic

from medcat.utils.regression.targeting import TranslationLayer, FinalTarget
from medcat.utils.regression.utils import limit_str_len, add_doc_strings_to_enum


class Finding(Enum):
    """Describes whether or how the finding verified.

    The idea is that we know where we expect the entity to be recognised
    and the enum constants describe how the recognition compared to the
    expectation.

    In essence, we want to know the relative positions of the two pairs of
    numbers (character numbers):
    - Expected Start, Expected End
    - Recognised Start, Recognised End

    We can model this as 4 numbers on the number line. And we want to know
    their position relative to each other.
    For example, if the expected positions are marked with * and recognised
    positions with #, we may have something like:
    ___*__#_______#*______________
    Which would indicate that there is a partial, but smaller span recognised.
    """
    # same CUIs
    IDENTICAL = auto()
    """The CUI and the span recognised are identical to what was expected."""
    BIGGER_SPAN_RIGHT = auto()
    """The CUI is the same, but the recognised span is longer on the right.

    If we use the notation from the class doc string, e.g:
    _*#__*__#"""
    BIGGER_SPAN_LEFT = auto()
    """The CUI is the same, but the recognised span is longer on the left.

    If we use the notation from the class doc string, e.g:
    _#_*__*#_"""
    BIGGER_SPAN_BOTH = auto()
    """The CUI is the same, but the recognised span is longer on both sides.

    If we use the notation from the class doc string, e.g:
    _#__*__*__#_"""
    SMALLER_SPAN = auto()
    """The CUI is the same, but the recognised span is smaller.

    If we use the notation from the class doc string, e.g:
    _*_#_#_*_ (neither start nor end match)
    _*#_#_*__ (start matches, but end is before expected)
    _*__#_#*_ (end matches, but start is after expected)"""
    PARTIAL_OVERLAP = auto()
    """The CUI is the same, but the span overlaps partially.

    If we use the notation from the class doc string, e.g:
    _*_#__*_#_ (starts between expected start and end, but ends beyond)
    _#_*_#_*__ (start before expected start, but ends between expected start and end)"""
    # slightly different CUIs
    FOUND_DIR_PARENT = auto()
    """The recognised CUI is a parent of the expected CUI but the span is an exact match."""
    FOUND_DIR_GRANDPARENT = auto()
    """The recognised CUI is a grandparent of the expected CUI but the span is an exact match."""
    FOUND_ANY_CHILD = auto()
    """The recognised CUI is a child of the expected CUI but the span is an exact match."""
    FOUND_CHILD_PARTIAL = auto()
    """The recognised CUI is a child yet the match is only partial (smaller/bigger/partial)."""
    FOUND_OTHER = auto()
    """Found another CUI in the same span."""
    FAIL = auto()
    """The concept was not recognised in any meaningful way."""

    def has_correct_cui(self) -> bool:
        """Whether the finding found the correct concept.

        Returns:
            bool: Whether the correct concept was found.
        """
        return self in (
            Finding.IDENTICAL, Finding.BIGGER_SPAN_RIGHT, Finding.BIGGER_SPAN_LEFT,
            Finding.BIGGER_SPAN_BOTH, Finding.SMALLER_SPAN, Finding.PARTIAL_OVERLAP
        )

    @classmethod
    def determine(cls, exp_cui: str, exp_start: int, exp_end: int,
                  tl: TranslationLayer, found_entities: Dict[str, Dict[str, Any]],
                  strict_only: bool = False,
                  check_children: bool = True, check_parent: bool = True,
                  check_grandparent: bool = True
                  ) -> Tuple['Finding', Optional[str]]:
        """Determine the finding type based on the input

        Args:
            exp_cui (str): Expected CUI.
            exp_start (int): Expected span start.
            exp_end (int): Expected span end.
            tl (TranslationLayer): The translation layer.
            found_entities (Dict[str, Dict[str, Any]]): The entities found by the model.
            strict_only (bool): Whether to use a strict-only mode (either identical or fail). Defaults to False.
            check_children (bool): Whether to check the children. Defaults to True.
            check_parent (bool): Whether to check for parent(s). Defaults to True.
            check_grandparent (bool): Whether to check for grandparent(s). Defaults to True.

        Returns:
            Tuple['Finding', Optional[str]]: The type of finding determined, and the alternative.
        """
        return FindingDeterminer(exp_cui, exp_start, exp_end,
                                 tl, found_entities, strict_only,
                                 check_children, check_parent, check_grandparent).determine()


# NOTE: add doc strings to enum constants
add_doc_strings_to_enum(Finding)


class FindingDeterminer:
    """A helper class to determine the type of finding.

    This is mostly useful to split the responsibilities of
    looking at children/parents as well as to keep track of
    the already-checked children to avoid infinite recursion
    (which could happen in - e.g - a SNOMED model).

    Args:
        exp_cui (str): The expected CUI.
        exp_start (int): The expected span start.
        exp_end (int): The expected span end.
        tl (TranslationLayer): The translation layer.
        found_entities (Dict[str, Dict[str, Any]]): The entities found by the model.
        strict_only (bool): Whether to use strict-only mode (either identical or fail). Defaults to False.
        check_children (bool): Whether or not to check the children. Defaults to True.
        check_parent (bool): Whether to check for parent(s). Defaults to True.
        check_grandparent (bool): Whether to check for granparent(s). Defaults to True.
    """

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
        """Determines the raw SPAN-ONLY finding.

        I.e this assumes the concept is appropriate.
        It will return None if there is no overlapping span.

        Args:
            start (int): The start of the span.
            end (int): The end of the span.

        Raises:
            MalformedFinding: If the start is greater than the end.
            MalformedFinding: If the expected start is greater than the expected end.

        Returns:
            Optional[Finding]: The finding, if a match is found.
        """
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
                return Finding.PARTIAL_OVERLAP  # TODO - distinguish[overlap]?
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
            return Finding.PARTIAL_OVERLAP  # TODO - distinguish[overlap]?
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

    def _check_parents(self) -> Optional[Tuple[Finding, Optional[str]]]:
        parents = self.tl.get_direct_parents(self.exp_cui)
        for parent in parents:
            finding, wcui = Finding.determine(parent, self.exp_start, self.exp_end,
                                              self.tl,
                                              self.found_entities,
                                              check_children=False,
                                              check_parent=self.check_grandparent,
                                              check_grandparent=False)
            if finding is Finding.IDENTICAL:
                return Finding.FOUND_DIR_PARENT, parent
            if finding is Finding.FOUND_DIR_PARENT:
                return Finding.FOUND_DIR_GRANDPARENT, wcui
        return None

    def _check_children(self) -> Optional[Tuple[Finding, Optional[str]]]:
        children = self.tl.get_direct_children(self.exp_cui)
        for child in children:
            finding, wcui = Finding.determine(child, self.exp_start, self.exp_end,
                                              self.tl,
                                              self.found_entities,
                                              check_children=True,
                                              check_parent=False,
                                              check_grandparent=False)
            if finding in (Finding.IDENTICAL, Finding.FOUND_ANY_CHILD):
                alt_cui = child if finding == Finding.IDENTICAL else wcui
                return Finding.FOUND_ANY_CHILD, alt_cui
            elif finding.has_correct_cui():
                # i.e a partial match with same CUI
                return Finding.FOUND_CHILD_PARTIAL, child
            elif finding is Finding.FOUND_CHILD_PARTIAL:
                return finding, wcui
            self._checked_children.add(child)
        return None

    def _descr_cui(self, cui: Optional[str]) -> Optional[str]:
        if cui is None:
            return None
        return f"{cui} ({self.tl.get_preferred_name(cui)})"

    def _find_diff_cui(self) -> Optional[Tuple[Finding, str]]:
        for entity in self.found_entities.values():
            start, end, cui = entity['start'], entity['end'], entity['cui']
            if start == self.exp_start and end == self.exp_end:
                return Finding.FOUND_OTHER, cui
        return None

    def determine(self) -> Tuple[Finding, Optional[str]]:
        """Determine the finding based on the given information.

        First, the strict check is done (either identical or not).
        Then, parents are checked (if required).
        After that, children are checked (if required).

        Returns:
            Tuple[Finding, Optional[str]]: The appropriate finding, and the alternative (if applicable).
        """
        finding, cui = self._determine()
        # NOTE: the point of this wrapper method is to add the preferred name
        #       to the CUI in one place and one place only
        return finding, self._descr_cui(cui)

    def _determine(self) -> Tuple[Finding, Optional[str]]:
        finding = self._get_strict()
        if finding is not None:
            return finding, None
        if self.check_parent:
            fpar = self._check_parents()
            if fpar is not None:
                return fpar
        if self.check_children:
            self._checked_children.add(self.exp_cui)
            fch = self._check_children()
            if fch is not None:
                return fch
        fdcui = self._find_diff_cui()
        return fdcui or (Finding.FAIL, None)


class Strictness(Enum):
    """The total strictness on which to judge the results."""
    STRICTEST = auto()
    """The strictest option which only allows identical findings."""
    STRICT = auto()
    """A strict option which allows identical or children."""
    NORMAL = auto()
    """Normal strictness also allows partial overlaps on target concept and children."""
    LENIENT = auto()
    """Lenient stictness also allows parents and grandparents."""
    ANYTHING = auto()
    """Anything stricness allows ANY finding.

    This would generally only be relevant when disabling examples
    for results descriptors."""


STRICTNESS_MATRIX: Dict[Strictness, Set[Finding]] = {
    Strictness.STRICTEST: {Finding.IDENTICAL},
    Strictness.STRICT: {Finding.IDENTICAL, Finding.FOUND_ANY_CHILD},
    Strictness.NORMAL: {
        Finding.IDENTICAL, Finding.FOUND_ANY_CHILD, Finding.FOUND_CHILD_PARTIAL,
        Finding.BIGGER_SPAN_RIGHT, Finding.BIGGER_SPAN_LEFT,
        Finding.BIGGER_SPAN_BOTH,
        Finding.SMALLER_SPAN, Finding.PARTIAL_OVERLAP
    },
    Strictness.LENIENT: {
        Finding.IDENTICAL, Finding.FOUND_ANY_CHILD,
        Finding.BIGGER_SPAN_RIGHT, Finding.BIGGER_SPAN_LEFT,
        Finding.BIGGER_SPAN_BOTH,
        Finding.SMALLER_SPAN, Finding.PARTIAL_OVERLAP,
        Finding.FOUND_DIR_PARENT, Finding.FOUND_DIR_GRANDPARENT,
    },
    Strictness.ANYTHING: set(Finding),
}


class SingleResultDescriptor(pydantic.BaseModel):
    """The result descriptor.

    This class is responsible for keeping track of all the
    findings (i.e how many were found to be identical) as
    well as the examples of the finding on a per-target
    basis for further analysis.
    """
    name: str
    """The name of the part that was checked"""
    findings: Dict[Finding, int] = {}
    """The description of failures"""
    examples: List[Tuple[FinalTarget, Tuple[Finding, Optional[str]]]] = []
    """The examples of non-perfect alignment."""

    def report_success(self, target: FinalTarget, found: Tuple[Finding, Optional[str]]) -> None:
        """Report a test case and its successfulness.

        Args:
            target (FinalTarget): The target configuration
            found (Tuple[Finding, Optional[str]]): Whether or not the check was successful
        """
        finding, _ = found
        if finding not in self.findings:
            self.findings[finding] = 0
        self.findings[finding] += 1
        self.examples.append((target, found))

    def get_report(self) -> str:
        """Get the report associated with this descriptor

        Returns:
            str: The report string
        """
        total = sum(self.findings.values())
        ret_vals = [f"Tested '{self.name}' for a total of {total} cases:"]
        ret_vals.extend([
            f"{f.name:24s}:{self.findings[f]:10d} ({100 * self.findings[f] / total if total > 0 else 0:5.2f}%)"
            # NOTE iterating over Finding so the order is the same as in the enum
            for f in Finding if f in self.findings
        ])
        return "\n".join(ret_vals)

    def model_dump(self, **kwargs) -> dict:
        if 'strictness' in kwargs:
            kwargs = kwargs.copy() # so if used elsewhere, keeps the kwarg
            strict_raw = kwargs.pop('strictness')
            if isinstance(strict_raw, Strictness):
                strictness = strict_raw
            elif isinstance(strict_raw, str):
                strictness = Strictness[strict_raw]
            else:
                raise ValueError(f"Unknown stircntess specified: {strict_raw}")
        else:
            strictness = Strictness.NORMAL
        # avoid serialising multiple times
        if 'exclude' in kwargs and kwargs['exclude'] is not None:
            exclude: set = kwargs['exclude']
        else:
            exclude = set()
            kwargs['exclude'] = exclude
        exclude.update(('findings', 'examples'))
        serialized_dict = {
            key.name: value for key, value in self.findings.items()
        }
        serialized_examples = [
            (ft.model_dump(**kwargs), (f[0].name, f[1])) for ft, f in self.examples
            # only count if NOT in strictness matrix (i.e 'failures')
            if f[0] not in STRICTNESS_MATRIX[strictness]
        ]
        model_dict = cast(pydantic.BaseModel, super()).model_dump(**kwargs)
        model_dict['findings'] = serialized_dict
        model_dict['examples'] = serialized_examples
        return model_dict

    def json(self, **kwargs) -> str:
        d = self.model_dump(**kwargs)
        return json.dumps(d)


class ResultDescriptor(SingleResultDescriptor):
    """The overarching result descriptor that handles multiple phrases.

    This class keeps track of the results on a per-phrase basis and
    can be used to get the overall report and/or iterate over examples.
    """
    per_phrase_results: Dict[str, SingleResultDescriptor] = {}

    def report(self, target: FinalTarget, finding: Tuple[Finding, Optional[str]]) -> None:
        """Report a test case and its successfulness

        Args:
            target (FinalTarget): The final targe configuration
            finding (Tuple[Finding, Optional[str]]): To what extent the concept was recognised
        """
        phrase = target.final_phrase
        super().report_success(target, finding)
        if phrase not in self.per_phrase_results:
            self.per_phrase_results[phrase] = SingleResultDescriptor(
                name=phrase)
        self.per_phrase_results[phrase].report_success(target, finding)

    def iter_examples(self, strictness_threshold: Strictness
                      ) -> Iterable[Tuple[FinalTarget, Tuple[Finding, Optional[str]]]]:
        """Iterate suitable examples.

        The strictness threshold at which to include examples.

        Any finding that is assumed to be "correct enough" according to
        the strictness matrix for this threshold will be withheld from
        examples.

        In simpler terms, if the finding is NOT in the strictness matrix
        for this strictness, the example is recorded.

        NOTE: To disable example keeping, set the threshold to Strictness.ANYTHING.

        Args:
            strictness_threshold (Strictness): The strictness threshold.

        Yields:
            Iterable[Tuple[FinalTarget, Tuple[Finding, Optional[str]]]]: The placeholder, phrase, finding, CUI, and name.
        """
        phrases = sorted(self.per_phrase_results.keys())
        for phrase in phrases:
            srd = self.per_phrase_results[phrase]
            # sort by finding 1st, found CUI 2nd, and used name 3rd
            sorted_examples = sorted(
                srd.examples, key=lambda tf: (tf[1][0].name, str(tf[1][1]), tf[0].name))
            for target, finding in sorted_examples:
                if finding[0] not in STRICTNESS_MATRIX[strictness_threshold]:
                    yield target, finding

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

    def model_dump(self, **kwargs) -> dict:
        if 'exclude' in kwargs and kwargs['exclude'] is not None:
            exclude: set = kwargs['exclude']
        else:
            exclude = set()
            kwargs['exclude'] = exclude
        # NOTE: ignoring here so that examples are only present in the per phrase part
        exclude.update(('examples', 'per_phrase_results'))
        d = cast(pydantic.BaseModel, super()).model_dump(**kwargs)
        if 'examples' in d:
            # NOTE: I don't really know why, but the examples still
            #       seem to be a part of the resulting dict, so I need
            #       to explicitly remove them
            del d['examples']
        # NOTE: need to propagate here manually so the strictness keyword
        #       makes sense and doesn't cause issues due being to unexpected keyword
        per_phrase_results = {
            phrase: res.model_dump(**kwargs) for phrase, res in
            sorted(self.per_phrase_results.items(), key=lambda it: it[0])
        }
        d['per_phrase_results'] = per_phrase_results
        return d


class MultiDescriptor(pydantic.BaseModel):
    """The descriptor of results over multiple different results (parts).

    The idea is that this would likely be used with a regression suite
    and it would incorporate all the different regression cases it describes.
    """
    name: str
    """The name of the collection being checked"""
    parts: List[ResultDescriptor] = []
    """The parts kept track of"""

    @property
    def findings(self) -> Dict[Finding, int]:
        """The total findings.

        Returns:
            Dict[Finding, int]: The total number of successes.
        """
        totals: Dict[Finding, int] = {}
        for part in self.parts:
            for f, val in part.findings.items():
                if f not in totals:
                    totals[f] = val
                else:
                    totals[f] += val
        return totals

    def iter_examples(self, strictness_threshold: Strictness
                      ) -> Iterable[Tuple[FinalTarget, Tuple[Finding, Optional[str]]]]:
        """Iterate over all relevant examples.

        Only examples that are not in the strictness matrix for the specified
        threshold will be used.

        Args:
            strictness_threshold (Strictness): The threshold of avoidance.

        Yields:
            Iterable[Tuple[FinalTarget, Tuple[Finding, Optional[str]]]]: The examples
        """
        for descr in self.parts:
            yield from descr.iter_examples(strictness_threshold=strictness_threshold)

    def _get_part_report(self, part: ResultDescriptor, allowed_findings: Set[Finding],
                         total_findings: Dict[Finding, int],
                         hide_empty: bool,
                         examples_strictness: Optional[Strictness],
                         phrases_separately: bool,
                         phrase_max_len: int,
                         ) -> Tuple[str, int, int, int]:
        if hide_empty and len(part.findings) == 0:
            return '', 0, 0, 0
        total_total, total_s, total_f = 0, 0, 0
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
        cur_add = '\t' + \
            part.get_report(phrases_separately=phrases_separately).replace(
                '\n', '\n\t\t')
        if examples_strictness is not None:
            latest_phrase = ''
            for target, found in part.iter_examples(strictness_threshold=examples_strictness):
                finding, ocui = found
                if latest_phrase == '':
                    # add header only if there's failures to include
                    cur_add += f"\n\t\tExamples at {examples_strictness} strictness"
                if latest_phrase != target.final_phrase:
                    short_phrase = limit_str_len(target.final_phrase, max_length=phrase_max_len,
                                                 keep_front=phrase_max_len // 2,
                                                 keep_rear=phrase_max_len // 2 - 10)
                    cur_add += f"\n\t\tWith phrase: {repr(short_phrase)}"
                    latest_phrase = target.final_phrase
                found_cui_descr = f' [{ocui}]' if ocui else ''
                cur_add += (f'\n\t\t\t{finding.name}{found_cui_descr} for '
                            f'placeholder {target.placeholder} '
                            f'with CUI {repr(target.cui)} and name {repr(target.name)}')
        return cur_add, total_total, total_s, total_f

    def calculate_report(self, phrases_separately: bool = False,
                         hide_empty: bool = False,
                         examples_strictness: Optional[Strictness] = Strictness.STRICTEST,
                         strictness: Strictness = Strictness.NORMAL,
                         phrase_max_len: int = 80) -> Tuple[int, int, int, str, int]:
        """Calculate some of the major parts of the report.

        Args:
            phrases_separately (bool): Whether to include per-phrase information
            hide_empty (bool): Whether to hide empty cases
            examples_strictness (Optional[Strictness.STRICTEST]): What level of strictness to show for examples.
                Set to None to disable examples. Defaults to Strictness.STRICTEST.
            strictness (Strictness): The strictness of the success / fail overview.
                Defaults to Strictness.NORMAL.
            phrase_max_len (int): The maximum length of the phrase in examples. Defaults to 80.

        Returns:
            Tuple[int, int, int, int, str]: The total number of examples, the total successes, the total failures,
                the delegated part, and the number of empty
        """
        del_out = []  # delegation
        total_findings: Dict[Finding, int] = {}
        total_s, total_f = 0, 0
        allowed_findings = STRICTNESS_MATRIX[strictness]
        total_total = 0
        nr_of_empty = 0
        for part in self.parts:
            (cur_add, total_total_add,
             total_s_add, total_f_add) = self._get_part_report(
                 part, allowed_findings, total_findings, hide_empty,
                 # NOTE: using STRICTEST strictness for examples means
                 #       that all but IDENTICAL examples will be shown
                 examples_strictness, phrases_separately, phrase_max_len)
            if hide_empty and total_total_add == 0:
                nr_of_empty += 1
            else:
                total_total += total_total_add
                total_s += total_s_add
                total_f += total_f_add
                del_out.append(cur_add)
        delegated = '\n'.join(del_out)
        return total_total, total_s, total_f, delegated, nr_of_empty

    def get_report(self, phrases_separately: bool,
                   hide_empty: bool = False,
                   examples_strictness: Optional[Strictness] = Strictness.STRICTEST,
                   strictness: Strictness = Strictness.NORMAL,
                   phrase_max_len: int = 80) -> str:
        """Get the report associated with this descriptor

        Args:
            phrases_separately (bool): Whether to include per-phrase information
            hide_empty (bool): Whether to hide empty cases
            examples_strictness (Optional[Strictness.STRICTEST]): What level of strictness to show for examples.
                Set to None to disable examples. Defaults to Strictness.STRICTEST.
            strictness (Strictness): The strictness of the success / fail overview.
                Defaults to Strictness.NORMAL.
            phrase_max_len (int): The maximum length of the phrase in examples. Defaults to 80.

        Returns:
            str: The report string
        """
        (total_total, total_s, total_f,
         delegated, nr_of_empty) = self.calculate_report(phrases_separately=phrases_separately,
                                                         hide_empty=hide_empty,
                                                         examples_strictness=examples_strictness,
                                                         strictness=strictness,
                                                         phrase_max_len=phrase_max_len)
        empty_text = ''
        allowed_findings = STRICTNESS_MATRIX[strictness]
        if hide_empty:
            empty_text = f' A total of {nr_of_empty} cases did not match any CUIs and/or names.'
        ret_vals = [f"""A total of {len(self.parts)} parts were kept track of within the group "{self.name}".
                    And a total of {total_total} (sub)cases were checked.{empty_text}"""]
        allowed_fingings_str = sorted([f.name for f in allowed_findings])
        ret_vals.extend([
            f"At the strictness level of {strictness} (allowing {allowed_fingings_str}):",
            f"The number of total successful (sub) cases: {total_s} "
            f"({100 * total_s/total_total if total_total > 0 else 0:5.2f}%)",
            f"The number of total failing (sub) cases   : {total_f} "
            f"({100 * total_f/total_total if total_total > 0 else 0:5.2f}%)"
        ])
        ret_vals.extend([
            f"{f.name:24s}:{self.findings[f]:10d} "
            f"({100 * self.findings[f] / total_total if total_total > 0 else 0:5.2f}%)"
            # NOTE iterating over Finding so the order is the same as in the enum
            for f in Finding if f in self.findings
        ])
        return "\n".join(ret_vals) + f"\n{delegated}"

    def model_dump(self, **kwargs) -> dict:
        if 'strictness' in kwargs:
            strict_raw = kwargs.pop('strictness')
            if isinstance(strict_raw, Strictness):
                strictness = strict_raw
            elif isinstance(strict_raw, str):
                strictness = Strictness[strict_raw]
            else:
                raise ValueError(f"Unknown stircntess specified: {strict_raw}")
        else:
            strictness = Strictness.NORMAL
        out_dict = cast(pydantic.BaseModel, super()).model_dump(exclude={'parts'}, **kwargs)
        out_dict['parts'] = [part.model_dump(strictness=strictness) for part in self.parts]
        return out_dict


class MalformedFinding(ValueError):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
