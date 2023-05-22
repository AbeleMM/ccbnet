import numpy as np
from experiment import benchmark_multi
from pgmpy.utils import get_example_model
from writer import OutTarget, Writer


def main() -> None:
    writer = Writer(OutTarget.NONE)
    nets = ["sachs"]
    client_counts = [4]
    for net_name in nets:
        model = get_example_model(net_name)
        for client_count in client_counts:
            res = benchmark_multi(
                ref_model=model,
                nr_clients=client_count,
                overlap_ratios=np.arange(0.0, 0.6, 0.1).tolist(),
                samples_factor=500,
                test_counts=2000,
                include_learnt=True,
                in_out_inf_vars=True,
                rand_inf_vars=True,
                r_seed=42
            )
            for scenario, d_f in res.items():
                writer.write(net_name, client_count, scenario, d_f)

if __name__ == "__main__":
    main()
