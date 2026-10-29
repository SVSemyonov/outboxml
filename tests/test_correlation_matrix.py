from unittest.mock import patch

import numpy as np
import pandas as pd

from outboxml.analysis_tools import CorrelationMatrix


def test_correlations_are_calculated_without_worker_processes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    data = pd.DataFrame({"A": x, "A_COPY": x * 2 + 1, "B": rng.normal(size=300),
                         "C": rng.choice(["x", "y"], size=300)})

    with patch("phik.phik.Parallel", side_effect=RuntimeError("worker pool used")):
        to_drop = CorrelationMatrix(data=data, feature_importance_list=["C", "B", "A_COPY", "A"],
                                    features_numerical=["A", "A_COPY", "B"], threshold=0.9).result()

    assert to_drop == ["A_COPY"]
