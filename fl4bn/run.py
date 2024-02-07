from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from experiment import benchmark_multi
from pgmpy.utils import get_example_model

WRITE = True


@dataclass(frozen=True)
class RunCfg:
    """Data class specifying the config for a set of experiments"""
    nets: list[str]
    net_to_nr_clients: dict[str, list[int]]
    eq_weights: bool
    overlap_ratios: list[float]
    connected: bool


def main() -> None:
    cfg_name = "related"
    run_cfg = RunCfg(
        ["asia", "child", "alarm", "insurance", "win95pts"],
        defaultdict(lambda: [2, 4, 8]),
        True,
        [0.1, 0.3, 0.5],
        True
    )

    # cfg_name = "random"
    # run_cfg = RunCfg(
    #     ["asia", "child", "alarm", "insurance"],
    #     defaultdict(lambda: [2, 4]),
    #     True,
    #     [0.1, 0.3],
    #     False
    # )

    # cfg_name = "weighting"
    # run_cfg = RunCfg(
    #     ["asia", "child", "alarm", "insurance"],
    #     defaultdict(lambda: [2, 4]),
    #     False,
    #     [0.1, 0.3],
    #     False
    # )

    # cfg_name = "large"
    # run_cfg = RunCfg(
    #     ["andes", "pigs", "link", "munin2"],
    #     {"andes": [16], "pigs": [32], "link": [64], "munin2": [128]},
    #     True,
    #     [0.1],
    #     True
    # )

    if WRITE:
        time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        res_dir = Path(__file__).parents[1] / "out" / time_str
        res_dir.mkdir(parents=True, exist_ok=True)
    else:
        res_dir = None

    dfs: list[pd.DataFrame] = []

    for net_name in run_cfg.nets:
        model = get_example_model(net_name)
        model.name = net_name

        for client_count in run_cfg.net_to_nr_clients[net_name]:

            res = benchmark_multi(
                ref_bn=model,
                nr_clients=client_count,
                overlap_ratios=run_cfg.overlap_ratios,
                samples_factor=500 if run_cfg.eq_weights else 250,
                test_counts=2000,
                connected=run_cfg.connected,
                eq_weights=run_cfg.eq_weights,
                r_seed=1
            )
            res["net"] = net_name
            res["nr_clients"] = client_count
            res = res.reset_index()
            dfs.append(res)
            print(res)

    df = pd.concat(dfs, ignore_index=True)
    print(df)

    if res_dir:
        df.to_csv(res_dir / f"{cfg_name}.csv")


if __name__ == "__main__":
    main()
