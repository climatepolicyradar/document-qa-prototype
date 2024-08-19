
import pandas as pd

from typing import Callable, Optional, Union


def aggregate_and_print_results(
        df: pd.DataFrame,
        evals: pd.DataFrame,
        filter_func: Callable,
        attributes_to_breakdown: dict[str, str], 
        title: Optional[str],
        update_evals: bool = False,
        markdown: bool = False
    ) -> pd.DataFrame:
    """
    Aggregates and prints the results for a given set of attributes based on the evals and qa-pairs dataframes
    
    It applies a filter function by which it chooses the 'positives' from the evals dataset. These are usually the violations
    on some axis of evaluation.
    
    
    Args:
    - df: the qa-pairs dataframe
    - evals: the evals dataframe
    - filter_func: filter for the evals dataframe (usually the eval threshold for the axis)
    - attributes_to_breakdown: A dictionary with the attributes to breakdown and the aggregation type (ratio or count)
    - title: title for the print block
    - update_evals: whether to return the updated (filtered) evals dataframe
    - markdown: whether to print the results in markdown format
    
    Returns:
    - The updated evals dataframe if update_evals is True, otherwise the evals dataframe -- this is used for filtering
      purposes given the sequential nature of the analysis
    """
    if title:
        print(f"{title}\n\n")
    
    positives = evals[filter_func(evals)]
    positive_df = df[df["id"].isin(positives.index)]

    _df = df[df["id"].isin(evals.index)]

    print(f"Total number of positives: {len(positive_df)} out of {len(_df)}, ({len(positive_df) / len(_df) * 100:.2f}%)")

    for attribute, aggregation in attributes_to_breakdown.items():
        print(f"\n{attribute} as {aggregation}:")
        printable = breakdown_for_attribute(positive_df, _df, evals, attribute, aggregation, markdown)
        print(f"{printable}\n")
    
    if update_evals:
        return evals[~evals.index.isin(positives.index)]
    else:
        return evals
    


def breakdown_for_attribute(
        positive_df: pd.DataFrame,
        df: pd.DataFrame,
        evals: pd.DataFrame,
        attribute: str,
        aggregation: str,
        markdown: bool = False
    ) -> Union[pd.DataFrame, str]:
    """Creates a breakdown of the counts or ratios based on the eval results for a given attribute"""

    _counts = positive_df[attribute].value_counts()

    if aggregation == "count":
        printable =  _counts
    elif aggregation == "ratio":
        printable = _counts / df[df["id"].isin(evals.index)][attribute].value_counts()
    else:
        raise ValueError("Invalid aggregation type")
    
    printable = pd.DataFrame(printable.sort_values(ascending=False)).T

    return printable if not markdown else printable.to_markdown()
