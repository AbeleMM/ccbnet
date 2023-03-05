from typing import cast

from matplotlib.axes import Axes
from pgmpy.utils import get_example_model
from util import BENCHMARK_PIVOT_COL, ExpWriter, benchmark_multi


def plot_model_results(
        model_name: str,
        in_out_inf_vars: bool,
        exp_writer: ExpWriter | None = None) -> None:
    model = get_example_model(model_name)
    res = benchmark_multi(model, 4, 2000, 50000, True, in_out_inf_vars, 42)

    for metric in [col for col in res.columns if col != BENCHMARK_PIVOT_COL]:
        pivoted_res = res.pivot(columns=BENCHMARK_PIVOT_COL, values=metric)
        axes = cast(Axes, pivoted_res.plot(kind="bar", title=model_name, ylabel=metric))
        if exp_writer:
            name = f"{model_name}_{'inout' if in_out_inf_vars else 'halves'}_{metric}"
            exp_writer.save_fig(axes, name)


def main() -> None:
    exp_writer = ExpWriter()
    nets = [
        "asia", "cancer", "earthquake", "sachs", "survey",
        "alarm", "child", "insurance", "water"
    ]
    for net_name in nets:
        for in_out_inf_vars in [True, False]:
            plot_model_results(net_name, in_out_inf_vars, exp_writer)


if __name__ == "__main__":
    main()
