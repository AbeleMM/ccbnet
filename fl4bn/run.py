from experiment import benchmark_multi
from pgmpy.utils import get_example_model
from writer import OutTarget, Writer


def main() -> None:
    writer = Writer(OutTarget.NONE)
    nets = ["asia", "sachs", "child", "alarm", "insurance", "win95pts"]
    client_counts = [2, 4, 8]
    for net_name in nets:
        model = get_example_model(net_name)
        model.name = net_name
        for client_count in client_counts:
            res = benchmark_multi(
                ref_model=model,
                nr_clients=client_count,
                overlap_ratios=[0.1, 0.3, 0.6],
                samples_factor=500,
                test_counts=2000,
                include_learnt=False,
                in_out_inf_vars=True,
                rand_inf_vars=True,
                r_seed=42
            )
            for scenario, d_f in res.items():
                writer.write(net_name, client_count, scenario, d_f)


if __name__ == "__main__":
    main()
