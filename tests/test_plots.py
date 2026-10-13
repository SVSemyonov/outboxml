import pandas as pd
import pytest

from outboxml.plots import CompareModelsPlot, MLPlot


def test_mlplot_unknown_plot_type_raises_value_error():
    plot = MLPlot(model_name_1="first", y_graph=pd.DataFrame(), features_numerical=[], features_categorical=[],
                  show=False)

    with pytest.raises(ValueError, match="Unknown plot type: 7"):
        plot.make(plot_type=7, cut_min_value=0.01, cut_max_value=0.9, samples=100, cohort_base="model")


def test_compare_models_plot_unknown_plot_type_raises_value_error():
    plot = CompareModelsPlot(model_name="first", df1=pd.DataFrame(), df2=pd.DataFrame(),
                             features_categorical=[], features_numerical=[], show=False)

    with pytest.raises(ValueError, match="Unknown plot type: 7"):
        plot.make(plot_type=7)
