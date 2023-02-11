from util import benchmark


def run(name: str) -> None:
    benchmark_res = benchmark(name, 4, 2000, r_seed=42)

    for method, res in benchmark_res.items():
        print(method, res)


if __name__ == "__main__":
    for net in ["asia", "alarm", "child", "hailfinder"]:
        print(net)
        run(net)
        print()
