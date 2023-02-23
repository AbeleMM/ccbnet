from util import benchmark

if __name__ == "__main__":
    for net in ["asia", "alarm", "child"]:
        print(f"| {net} |\n")
        benchmark(net, 4, 2000, overlap=0.0, r_seed=42)
        print("---\n")
