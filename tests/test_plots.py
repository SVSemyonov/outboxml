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


def test_plotly_params_are_applied_to_traces():
    from outboxml.plots import PlotlyWrapper

    wrapper = PlotlyWrapper(model_name="first", plotly_params={
        "names": {"model": "M"}, "colors": {"fact": "rgba(0,0,0,1)"}, "mode": {"fact": "markers"},
        "title": "My title", "unknown": 1})
    wrapper.add_figure([1, 2], [3, 4], name="model")
    wrapper.add_figure([1, 2], [5, 6], name="fact")
    figure = wrapper.prepare_plot()

    model_trace, fact_trace = figure.data
    assert model_trace.name == "M"
    assert (fact_trace.marker.color, fact_trace.mode, fact_trace.name) == ("rgba(0,0,0,1)", "markers", "Fact")
    assert figure.layout.title.text == "My title"
