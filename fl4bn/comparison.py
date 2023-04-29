from typing import cast

from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from pgmpy.utils import get_example_model
from util import BENCHMARK_PIVOT_COL, ExpWriter, benchmark_multi


def plot_model_results(
        model_name: str,
        in_out_inf_vars: bool,
        exp_writer: ExpWriter | None = None) -> None:
    model = get_example_model(model_name)
    res = benchmark_multi(model, 4, 2000, 50000, True, in_out_inf_vars, True, 42)

    if not exp_writer:
        print('inout' if in_out_inf_vars else 'halves')
        print(res)
        return

    for metric in [col for col in res.columns if col != BENCHMARK_PIVOT_COL]:
        pivoted_res = res.pivot(columns=BENCHMARK_PIVOT_COL, values=metric)
        axes = cast(
            Axes,
            pivoted_res.plot(kind="barh", title=model_name, xlabel=metric, width=0.9)
        )
        for container in cast(list[BarContainer], getattr(axes, "containers")):
            axes.bar_label(container, label_type="center", fontsize="xx-small")
        # axes.legend(mode="expand")
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
