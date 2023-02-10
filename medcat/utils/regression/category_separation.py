from abc import ABC, abstractmethod
from enum import auto, Enum
import os
from typing import Any, List, Dict, Optional, Set
import yaml
import string
import random
import logging

import pydantic

from medcat.utils.regression.checking import RegressionChecker, RegressionCase, FilterType, TypedFilter, MetaData


logger = logging.getLogger(__name__)


class CategoryDescription(pydantic.BaseModel):
    """A descriptor for a category.

    Args:
        target_cuis (Set[str]): The set of target CUIs
        target_names (Set[str]): The set of target names
        target_tuis (Set[str]): The set of target type IDs
        anything_goes (bool): Matches any CUI/NAME/TUI. Defaults to False
    """
    target_cuis: Set[str]
    target_names: Set[str]
    target_tuis: Set[str]
    allow_everything: bool = False

    def _get_required_filter(self, case: RegressionCase, target_filter: FilterType) -> Optional[TypedFilter]:
        for filter in case.filters:
            if filter.type == target_filter:
                return filter
        return None

    def _has_specific_from(self, case: RegressionCase, targets: Set[str], target_filter: FilterType):
        if self.allow_everything:
            return True
        filter = self._get_required_filter(case, target_filter)
        if filter is None:
            return False  # No such filter
        for val in filter.values:
            if val in targets:
                return True
        return False

    def has_cui_from(self, case: RegressionCase) -> bool:
        """Check if the description has a CUI from the specified regression case.

        Args:
            case (RegressionCase): The regression case to check

        Returns:
            bool: True if the description has a CUI from the regression case
        """
        return (self._has_specific_from(case, self.target_cuis, FilterType.CUI) or
                self._has_specific_from(case, self.target_cuis, FilterType.CUI_AND_CHILDREN))

    def has_name_from(self, case: RegressionCase) -> bool:
        """Check if the description has a name from the specified regression case.

        Args:
            case (RegressionCase): The regression case to check

        Returns:
            bool: True if the description has a name from the regression case
        """
        return self._has_specific_from(case, self.target_names, FilterType.NAME)

    def has_tui_from(self, case: RegressionCase) -> bool:
        """Check if the description has a target ID/TUI from the specified regression case.

        Args:
            case (RegressionCase): The regression case to check

        Returns:
            bool: True if the description has a target ID/TUI from the regression case
        """
        return self._has_specific_from(case, self.target_tuis, FilterType.TYPE_ID)

    def __hash__(self) -> int:
        return hash((tuple(self.target_cuis), tuple(self.target_names), tuple(self.target_tuis)))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CategoryDescription):
            return False
        return (self.target_cuis == other.target_cuis
                and self.target_names == other.target_names
                and self.target_tuis == other.target_tuis)

    @classmethod
    def anything_goes(cls) -> 'CategoryDescription':
        s: Set[str] = set()
        return CategoryDescription(target_cuis=s, target_tuis=s, target_names=s, allow_everything=True)


class Category(ABC):
    """The category base class.

    A category defines which regression cases fit in it.

    Args:
        name (str): The name of the category
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def fits(self, case: RegressionCase) -> bool:
        """Check if a particular regression case fits in this category"""


class AllPartsCategory(Category):
    """Represents a category which only fits a regression case if it matches all parts of category description.

    That is, in order for a regression case to match, it would need to match a CUI, a name and a TUI
    specified in the category description.

    Args:
        name (str): The name of the category
        descr (CategoryDescription): The description of the category
    """

    def __init__(self, name: str, descr: CategoryDescription) -> None:
        super().__init__(name)
        self.description = descr

    def fits(self, case: RegressionCase) -> bool:
        return (self.description.has_cui_from(case) and self.description.has_name_from(case)
                and self.description.has_tui_from(case))

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, AllPartsCategory):
            return False
        return __o.description == self.description

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.description))

    def __str__(self) -> str:
        return f"AllPartsCategory with: {self.description}"

    def __repr__(self) -> str:
        return f"<{str(self)}>"


class AnyPartOfCategory(Category):
    """Represents a category which fits a regression case that matches any part of its category desription.

    That is, any case that matches either a CUI, a name or a TUI within the category description, will fit.

    Args:
        name (str): The name of the category
        descr (CategoryDescription): The description of the category
    """

    def __init__(self, name: str, descr: CategoryDescription) -> None:
        super().__init__(name)
        self.description = descr

    def fits(self, case: RegressionCase) -> bool:
        return (self.description.has_cui_from(case) or self.description.has_name_from(case)
                or self.description.has_tui_from(case))

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, AnyPartOfCategory):
            return False
        return __o.description == self.description

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.description))

    def __str__(self) -> str:
        return f"AnyPartOfCategory with: {self.description}"

    def __repr__(self) -> str:
        return f"<{str(self)}>"


class SeparationObserver:
    """Keeps track of which case is separate into which category/categories.

    It also keeps track of which cases have been observed as separated and
    into which category.
    """

    def __init__(self) -> None:
        self.reset()

    def observe(self, case: RegressionCase, category: Category) -> None:
        """Observe the specified regression case in the specified category.

        Args:
            case (RegressionCase): The regression case to observe
            category (Category): The category to link the case tos
        """
        if category not in self.separated:
            self.separated[category] = set()
        self.separated[category].add(case)
        if case not in self.cases:
            self.cases[case] = set()
        self.cases[case].add(category)

    def has_observed(self, case: RegressionCase) -> bool:
        """Check if the case has already been observed.

        Args:
            case (RegressionCase): The case to check

        Returns:
            bool: True if the case had been observed, False otherwise
        """
        return case in self.cases

    def reset(self) -> None:
        """Allows resetting the state of the observer."""
        self.separated: Dict[Category, Set[RegressionCase]] = {}
        self.cases: Dict[RegressionCase, Set[Category]] = {}


class StrategyType(Enum):
    """Describes the types of strategies one can can employ for strategy."""
    FIRST = auto
    ALL = auto


class SeparatorStrategy(ABC):
    """The strategy according to which the separation takes place.

    The separation strategy relies on the mutable separation observer instance.
    """

    def __init__(self, observer: SeparationObserver) -> None:
        self.observer = observer

    @abstractmethod
    def can_separate(self, case: RegressionCase) -> bool:
        """Check if the separator strategy can separate the specified regression case

        Args:
            case (RegressionCase): The regression case to check

        Returns:
            bool: True if the strategy allows separation, False otherwise
        """

    @abstractmethod
    def separate(self, case: RegressionCase, category: Category) -> None:
        """Separate the regression case

        Args:
            case (RegressionCase): The regression case to separate
            category (Category): The category to separate to
        """

    def reset(self) -> None:
        """Allows resetting the state of the separator strategy."""
        self.observer.reset()


class SeparateToFirst(SeparatorStrategy):
    """Separator strategy that separates each case to its first match.

    That is to say, any subsequently matching categories are ignored.
    This means that no regression case gets duplicated.
    It also means that the number of cases in all categories will be the
    same as the initial number of cases.
    """

    def can_separate(self, case: RegressionCase) -> bool:
        return not self.observer.has_observed(case)

    def separate(self, case: RegressionCase, category: Category) -> None:
        if self.observer.has_observed(case):
            raise ValueError(f"Case {case} has already been observed")
        self.observer.observe(case, category)


class SeparateToAll(SeparatorStrategy):
    """A separator strateg that allows separation to all matching categories.

    This means that when one regression case fits into multiple categories,
    it will be saved in each such category. I.e the some cases may be
    duplicated.
    """

    def can_separate(self, case: RegressionCase) -> bool:
        return True

    def separate(self, case: RegressionCase, category: Category) -> None:
        self.observer.observe(case, category)


def get_random_str(length=8):
    return ''.join(random.choices(string.ascii_letters, k=length))


class RegressionCheckerSeparator(pydantic.BaseModel):
    """Regression checker separtor.

    It is able to separate cases in a regression checker
    into multiple different sets of regression cases
    based on the given list of categories and the specified
    strategy.

    Args:
        categories(List[Category]): The categories to separate into
        strategy(SeparatorStrategy): The strategy for separation
        overflow_category(bool): Whether to use an overflow category for cases that don't fit in other categoreis. Defaults to False.
    """

    categories: List[Category]
    strategy: SeparatorStrategy
    overflow_category: bool = False

    class Config:
        arbitrary_types_allowed = True

    def _attempt_category_for(self, cat: Category, case: RegressionCase):
        if cat.fits(case) and self.strategy.can_separate(case):
            self.strategy.separate(case, cat)

    def find_categories_for(self, case: RegressionCase):
        """Find the categories for a specific regression case

        Args:
            case (RegressionCase): The regression case to check
        """
        for cat in self.categories:
            self._attempt_category_for(cat, case)
        if not self.strategy.observer.has_observed(case) and self.overflow_category:
            anything_goes = AnyPartOfCategory(
                f'overflow-{get_random_str()}', descr=CategoryDescription.anything_goes())
            self.categories.append(anything_goes)
            self._attempt_category_for(anything_goes, case)
            logger.info(
                "Created overflow category since not all cases fit in specified categories")
            logger.info("The overflow category is named: %s",
                        anything_goes.name)
            if not self.strategy.observer.has_observed(case):
                raise ValueError("Anything-goes category should be sufficient")

    def separate(self, checker: RegressionChecker) -> None:
        """Separate the specified regression checker into multiple sets of cases.

        Each case may be associated with either no, one, or multiple categories.
        The specifics depends on `allow_overflow` and `strategy`.

        Args:
            checker(RegressionChecker): The input regression checker
        """
        for case in checker.cases:
            self.find_categories_for(case)

    def save(self, prefix: str, metadata: MetaData, overwrite: bool = False) -> None:
        """Save the results of the separation in different files.

        This needs to be called after the `separate` method has been called.

        Each separated category (that has any cases registered to it) will
        be saved in a separate file with the specified predix and the category name.

        Args:
            prefix (str): The prefix for the saved file(s)
            metadata (MetaData): The metadata for the regression suite
            overwrite (bool, optional): Whether to overwrite file(s) if/when needed. Defaults to False.

        Raises:
            ValueError: If the method is called before separation or no separtion was done
            ValueError: If a file already exists and is not allowed to be overwritten
        """
        if not self.strategy.observer.separated:  # empty
            raise ValueError("Need to do separation before saving!")
        for category, cases in self.strategy.observer.separated.items():
            rc = RegressionChecker(list(cases), metadata=metadata)
            yaml_str = rc.to_yaml()
            yaml_file_name = f"{prefix}_{category.name}.yml"
            if not overwrite and os.path.exists(yaml_file_name):
                raise ValueError(f"File already exists: {yaml_file_name}. "
                                 "Pass overwrite=True to overwrite")
            logger.info("Writing %d cases to %s", len(cases), yaml_file_name)
            with open(yaml_file_name, 'w') as f:
                f.write(yaml_str)


def get_strategy(strategy_type: StrategyType) -> SeparatorStrategy:
    """Get the separator strategy from the strategy type.

    Args:
        strategy_type (StrategyType): The type of strategy

    Raises:
        ValueError: If an unknown strategy is provided

    Returns:
        SeparatorStrategy: The resulting separator strategys
    """
    observer = SeparationObserver()
    if strategy_type == StrategyType.FIRST:
        return SeparateToFirst(observer)
    elif strategy_type == StrategyType.ALL:
        return SeparateToAll(observer)
    else:
        raise ValueError(f"Unknown strategy type {strategy_type}")


def get_separator(categories: List[Category], strategy_type: StrategyType,
                  overflow_category: bool = False) -> RegressionCheckerSeparator:
    """Get the regression checker separator for the list of categories and the specified strategy.

    Args:
        categories (List[Category]): The list of categories to include
        strategy_type (StrategyType): The strategy for separation
        overflow_category (bool): Whether to use an overflow category for items that don't go in other categories. Defaults to False.

    Returns:
        RegressionCheckerSeparator: The resulting separator
    """
    strategy = get_strategy(strategy_type)
    return RegressionCheckerSeparator(categories=categories, strategy=strategy, overflow_category=overflow_category)


def get_description(cat_description: dict) -> CategoryDescription:
    """Get the description from its dict representation.

    The dict is expected to have the following keys:
    'cuis', 'tuis', and 'names'
    Each one should have a list of strings as their values.

    Args:
        cat_description (dict): The dict representation

    Returns:
        CategoryDescription: The resulting category description
    """
    cuis = set(cat_description['cuis'])
    names = set(cat_description['names'])
    tuis = set(cat_description['tuis'])
    return CategoryDescription(target_cuis=cuis, target_names=names, target_tuis=tuis)


def get_category(cat_name: str, cat_description: dict) -> Category:
    """Get the category of the specified name from the dict.

    The dict is expected to be in the form:
        type: <category type> # either any or all
        cuis: []  # list of CUIs in category
        names: [] # list of names in category
        tuis: []  # list of type IDs in category

    Args:
        cat_name (str): The name of the category
        cat_description (dict): The dict describing the category

    Raises:
        ValueError: If an unknown type is specified.

    Returns:
        Category: The resulting category
    """
    description = get_description(cat_description)
    cat_type = cat_description['type']
    if cat_type.lower() in ('any', 'anyparts', 'anypartsof'):
        return AnyPartOfCategory(cat_name, description)
    elif cat_type.lower() in ('all', 'allparts'):
        return AllPartsCategory(cat_name, description)
    else:
        raise ValueError(
            f"Unknown category type: {cat_type} for category '{cat_name}'")


def read_categories(yaml_file: str) -> List[Category]:
    """Read categories from a YAML file.

    The yaml is assumed to be in the format:
    categories:
      category-name:
        type: <category type>
        cuis: [<target cui 1>, <target cui 2>, ...]
        names: [<target name 1>, <target name 2>, ...]
        tuis: [<target tui 1>, <target tui 2>, ...]
      other-category-name:
        ... # and so on

    Args:
        yaml_file (str): The yaml file location

    Returns:
        List[Category]: The resulting categories
    """
    with open(yaml_file) as f:
        d = yaml.safe_load(f)
    cat_part = d['categories']
    return [get_category(cat_name, cat_part[cat_name]) for cat_name in cat_part]


def separate_categories(category_yaml: str, strategy_type: StrategyType,
                        regression_suite_yaml: str, target_file_prefix: str, overwrite: bool = False,
                        overflow_category: bool = False) -> None:
    """Separate categories based on simple input.

    The categories are read from the provided file and
    the regression suite from its corresponding yaml.
    The separated regression suites are saved in accordance
    to the defined prefix.

    Args:
        category_yaml (str): The name of the YAML file describing the categories
        strategy_type (StrategyType): The strategy for separation
        regression_suite_yaml (str): The regression suite YAML
        target_file_prefix (str): The target file prefix
        overwrite (bool, optional): Whether to overwrite file(s) if/when needed. Defaults to False.
        overflow_category (bool): Whether to use an overflow category for items that don't go in other categories. Defaults to False.
    """
    separator = get_separator(read_categories(
        category_yaml), strategy_type, overflow_category)
    checker = RegressionChecker.from_yaml(regression_suite_yaml)
    separator.separate(checker)
    metadata = checker.metadata  # TODO - allow using different metadata?
    separator.save(target_file_prefix, metadata, overwrite=overwrite)
