from __future__ import annotations

import pytest

from submission.experiments.phase3c_qasper_validation import (
    compile_gold_groups,fast_group_coverage,
)
from submission.experiments.phase3b_full_qasper_train import best_annotator_coverage


@pytest.mark.parametrize("context,groups",[
    (
        "The method improves accuracy across three separate validation datasets.",
        [["The method improves accuracy across three separate validation datasets."],
         ["Another method contains completely unrelated details for comparison."]],
    ),
    (
        "We measured three alternative data sources from the paper.",
        [["We measured three alternative data sources from the paper.",
          "Another sentence which is not available in this document."],
         ["No matching phrase here in this original annotation."]],
    ),
    (
        "",
        [["This is an annotation with five or more words."]],
    ),
    (
        "Another short passage on a new result.",
        [["short"], ["Another short passage on a new result."]],
    ),
])
def test_compiled_coverage_matches_phase3b_legacy_metric(context,groups):
    new=fast_group_coverage(context,compile_gold_groups(groups))
    legacy=best_annotator_coverage(context,groups)
    assert new==legacy
