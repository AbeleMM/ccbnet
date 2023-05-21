from typing import cast

from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from pgmpy.utils import get_example_model
from util import BENCHMARK_PIVOT_COL, ExpWriter, benchmark_multi


def plot_model_results(
        model_name: str,
        exp_writer: ExpWriter | None = None) -> None:
    model = get_example_model(model_name)
    res = benchmark_multi(ref_model=model,
                          nr_clients=4,
                          test_counts=2000,
                          samples_factor=500,
                          include_learnt=True,
                          in_out_inf_vars=True,
                          rand_inf_vars=True,
                          r_seed=42)

    for scenario, d_f in res.items():
        if not exp_writer:
            print(scenario)
            print(d_f)
            continue

        for metric in [col for col in d_f.columns if col != BENCHMARK_PIVOT_COL]:
            pivoted_res = d_f.pivot(columns=BENCHMARK_PIVOT_COL, values=metric)
            axes = cast(
                Axes,
                pivoted_res.plot(kind="barh", title=model_name, xlabel=metric, width=0.9)
            )
            for container in cast(list[BarContainer], getattr(axes, "containers")):
                axes.bar_label(container, label_type="center", fontsize="xx-small")
            # axes.legend(mode="expand")
            if exp_writer:
                name = f"{model_name}_{scenario}_{metric}"
                exp_writer.save_fig(axes, name)


def main() -> None:
    exp_writer = ExpWriter()
    nets = [
        "asia", "cancer", "earthquake", "sachs", "survey",
        "alarm", "child", "insurance", "water"
    ]
    for net_name in nets:
        plot_model_results(net_name, exp_writer)


if __name__ == "__main__":
    main()
