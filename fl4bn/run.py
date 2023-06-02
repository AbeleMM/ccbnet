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
                ref_bn=model,
                nr_clients=client_count,
                overlap_ratios=[0.1, 0.3, 0.6],
                samples_factor=500,
                test_counts=2000,
                connected=True,
                r_seed=42
            )
            writer.write(net_name, client_count, res)


if __name__ == "__main__":
    main()
