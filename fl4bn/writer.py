from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from experiment import BENCHMARK_PIVOT_COL
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from pandas import DataFrame


class OutTarget(Enum):
    TXT = "txt"
    MD = "md"
    PNG = "PNG"
    SVG = "SVG"
    NONE = ""


class Writer():
    def __init__(self, out_target=OutTarget.NONE) -> None:
        self.out_target = out_target
        if self.out_target != OutTarget.NONE:
            time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            self.res_dir = Path(__file__).parents[1] / "out" / time_str
            self.res_dir.mkdir(parents=True, exist_ok=True)

    def write(self, net_name: str, nr_clients: int, scenario: str, d_f: DataFrame) -> None:
        net_clients_str = f"{net_name}_{nr_clients}_{scenario}"
        print(net_clients_str)
        print(d_f)
        match self.out_target:
            case OutTarget.TXT:
                d_f.to_string((self.res_dir / net_clients_str).with_suffix(
                    f".{self.out_target.value}"))
            case OutTarget.MD:
                d_f.to_markdown((self.res_dir / net_clients_str).with_suffix(
                    f".{self.out_target.value}"))
            case OutTarget.PNG | OutTarget.SVG:
                self.save_fig(net_clients_str, d_f)

    def save_fig(self, net_clients_str: str, d_f: DataFrame) -> None:
        for metric in [col for col in d_f.columns if col != BENCHMARK_PIVOT_COL]:
            pivoted_res = d_f.pivot(columns=BENCHMARK_PIVOT_COL, values=metric)
            axes = cast(
                Axes,
                pivoted_res.plot(kind="barh", title=net_clients_str, xlabel=metric, width=0.9)
            )
            for container in cast(list[BarContainer], getattr(axes, "containers")):
                axes.bar_label(container, label_type="center", fontsize="xx-small")

            fig = axes.get_figure()
            fig.savefig(
                str((self.res_dir / f"{net_clients_str}_{metric}").with_suffix(
                f".{self.out_target.value}")),
                bbox_inches="tight"
            )
            plt.close(fig)
