import sys
import re
import os
from tqdm import tqdm
import io
import tokenize
import keyword
import pandas as pd
import gc
import statistics
import pprint
import argparse


DEBUG = False


def main():
    args = get_hyperparameters()
    args_dict = vars(args)
    print(pprint.pformat(args_dict))

    global COLUMN_PREFIX
    COLUMN_PREFIX = args.column_prefix

    # Uncomment to run single-top evaluation
    # evaluate_top1()
    evaluate_top_metrics(args)



def get_hyperparameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--column_prefix", default=r'', type=str)
    parser.add_argument("--mode", default=r'', type=str)
    args = parser.parse_args()
    return args


def evaluate_top_metrics(args):
    """
    For each code example, we have 5 candidate fast-code versions.
    We use public IO pass-rate > 0.999 as a filter, then pick the one
    with the smallest public IO latency. We then collect the corresponding
    private IO pass-rate and execution time for that candidate.
    """
    # Load CSV
    df = pd.read_csv(args.dataset_path)

    # Initialize 5 lists of lists for the 5 candidates
    public_io_pass_rates = [[] for _ in range(5)]
    public_io_times      = [[] for _ in range(5)]
    private_io_pass_rates = [[] for _ in range(5)]
    private_io_times      = [[] for _ in range(5)]
    avg_log_probs        = [[] for _ in range(5)]

    # Populate the lists from dataframe columns
    for idx in range(5):
        i = idx + 1
        public_io_pass_rates[idx] = df[f"{COLUMN_PREFIX}__Predict_Fast_code_{i}__Public_IO_pass_rate_(%)"].tolist()
        public_io_times[idx]      = df[f"{COLUMN_PREFIX}__Predict_Fast_code_{i}__Public_IO_time(ms)"].tolist()
        private_io_pass_rates[idx] = df[f"{COLUMN_PREFIX}__Predict_Fast_code_{i}__IO_pass_rate_(%)"].tolist()
        private_io_times[idx]      = df[f"{COLUMN_PREFIX}__Predict_Fast_code_{i}__time(ms)"].tolist()
        avg_log_probs[idx]        = df[f"{COLUMN_PREFIX}__avg_log_probs_{i}"].tolist()

    # Ensure numeric types
    assert all(isinstance(x, float) for x in [
        public_io_pass_rates[0][0], public_io_times[0][0],
        private_io_pass_rates[0][0], private_io_times[0][0]
    ]), "Data values must be floats!"

    # Select best candidates using public IO metric
    if args.mode == 'top1':
        best_private_io_rates, best_private_io_times = select_top5_by_public_io(
            public_io_pass_rates, public_io_times,
            private_io_pass_rates, private_io_times
        )
    elif args.mode == 'top3':
        best_private_io_rates, best_private_io_times = select_top5_by_public_io_top3(
            public_io_pass_rates, public_io_times,
            private_io_pass_rates, private_io_times
        )
    elif args.mode == 'top5':
        best_private_io_rates, best_private_io_times = select_top5_by_private_io_top5(
            private_io_pass_rates, private_io_times
        )

    # Baseline slow-code metrics
    baseline_io_rates = df['input__IO_pass_rate_(%)'].tolist()
    baseline_times    = df['input__time(ms)'].tolist()

    assert len(baseline_io_rates) == len(best_private_io_rates) == len(baseline_times) == len(best_private_io_times), "Length mismatch!"

    print(f"\n\n### Baseline IO pass rates, average pass rate: {average_pass_rate(baseline_io_rates)}")
    print(f"\n\n### Optimized IO pass rates, average pass rate: {average_pass_rate(best_private_io_rates)}")
    print("\n\n### Baseline vs Optimized execution times:")
    output_optimization_metrics(
        baseline_io_rates, best_private_io_rates,
        baseline_times, best_private_io_times
    )

def evaluate_top1():
    """Evaluate only the top-1 predicted code version."""
    df = pd.read_csv(DATASET_PATH)
    baseline_io_rates = df['input__IO_pass_rate_(%)'].tolist()
    baseline_times    = df['input__time(ms)'].tolist()

    optimized_io_rates = df[f"{COLUMN_PREFIX}__Predict_Fast_code__IO_pass_rate_(%)"].tolist()
    optimized_times    = df[f"{COLUMN_PREFIX}__Predict_Fast_code__time(ms)"].tolist()

    print(f"Baseline average IO pass rate: {average_pass_rate(baseline_io_rates)}")
    print(f"Optimized average IO pass rate: {average_pass_rate(optimized_io_rates)}")
    print("\nBaseline vs Optimized execution times:")
    output_optimization_metrics(
        baseline_io_rates, optimized_io_rates,
        baseline_times, optimized_times
    )

def select_top5_by_max_prob(io_rates, io_times, log_probs):
    """
    Select, for each example, the candidate with highest average log-probability
    among the 5 and return its IO pass rate and execution time.
    """
    best_rates = []
    best_times = []
    num_examples = len(io_rates[0])
    for j in range(num_examples):
        max_idx = max(range(5), key=lambda i: log_probs[i][j])
        best_rates.append(io_rates[max_idx][j])
        best_times.append(io_times[max_idx][j])
    return best_rates, best_times

def select_top5_by_max_prob_top3(io_rates, io_times, log_probs):
    """
    From the 5 candidates, first filter those whose private IO pass rate > 0.999,
    then pick the one among the first three candidates with smallest execution time.
    """
    best_rates = []
    best_times = []
    num_examples = len(io_rates[0])
    for j in range(num_examples):
        # Candidate indices with pass rate > 0.999 among first 3
        candidates = [i for i in range(3) if io_rates[i][j] > 0.999]
        if candidates:
            chosen = min(candidates, key=lambda i: io_times[i][j])
        else:
            chosen = 0
        best_rates.append(io_rates[chosen][j])
        best_times.append(io_times[chosen][j])
    return best_rates, best_times

def select_top5_by_public_io(pub_rates, pub_times, priv_rates, priv_times):
    """
    For each example, filter candidates whose public IO pass-rate > 0.999,
    then pick the one with the smallest public IO time, and return its
    private IO pass-rate and private execution time.
    """
    best_rates = []
    best_times = []
    num_examples = len(priv_rates[0])
    for j in range(num_examples):
        candidates = [i for i in range(5) if pub_rates[i][j] > 0.999]
        if candidates:
            chosen = min(candidates, key=lambda i: pub_times[i][j])
        else:
            chosen = 0
        best_rates.append(priv_rates[chosen][j])
        best_times.append(priv_times[chosen][j])
    return best_rates, best_times

def select_top5_by_public_io_top3(pub_rates, pub_times, priv_rates, priv_times):
    """
    Variant: choose the top 3 by public IO time among candidates with pass-rate > 0.999,
    then among those 3, pick the one with private IO pass-rate > 0.999 and min private time.
    """
    best_rates = []
    best_times = []
    num_examples = len(priv_rates[0])
    for j in range(num_examples):
        # Step 1: candidates with public pass-rate > 0.999
        candidates = [i for i in range(5) if pub_rates[i][j] > 0.999]
        # Step 2: pick smallest three public IO times
        if candidates:
            top3 = sorted(candidates, key=lambda i: pub_times[i][j])[:3]
        else:
            top3 = [0]
        # Step 3: among top3, choose private pass-rate > 0.999 and smallest private time
        valid = [i for i in top3 if priv_rates[i][j] > 0.999]
        if valid:
            chosen = min(valid, key=lambda i: priv_times[i][j])
        else:
            chosen = 0
        best_rates.append(priv_rates[chosen][j])
        best_times.append(priv_times[chosen][j])
    return best_rates, best_times

def select_top5_by_private_io_top5(priv_rates, priv_times):
    """
    Among the 5 candidates, filter those with private pass-rate > 0.999,
    then pick the one with smallest private time.
    """
    best_rates = []
    best_times = []
    num_examples = len(priv_rates[0])
    for j in range(num_examples):
        candidates = [i for i in range(5) if priv_rates[i][j] > 0.999]
        if candidates:
            chosen = min(candidates, key=lambda i: priv_times[i][j])
        else:
            chosen = 0
        best_rates.append(priv_rates[chosen][j])
        best_times.append(priv_times[chosen][j])
    return best_rates, best_times

def average_pass_rate(rates_list, threshold=0.999):
    """Compute the fraction of rates above a threshold."""
    count_pass = sum(1 for x in rates_list if x > threshold)
    return count_pass / len(rates_list)

def output_optimization_metrics(slow_rates, fast_rates, slow_times, fast_times):
    """Compute and print time-speedups and percentage improvements."""
    assert len(slow_rates) == len(fast_rates) == len(slow_times) == len(fast_times), "List length mismatch!"
    time_improvements = []
    speedups = []
    for i in tqdm(range(len(slow_times))):
        if slow_rates[i] < 0.999:
            continue
        if slow_times[i] >= 1_234_567_890:
            continue
        if fast_rates[i] < 0.999 or fast_times[i] >= 1_234_567_890:
            time_improvements.append(0)
            speedups.append(1)
        else:
            slow_t = slow_times[i]
            fast_t = fast_times[i]
            if fast_t < slow_t:
                improvement = round(((slow_t - fast_t) / fast_t) * 100, 2)
                ratio = round(slow_t / fast_t, 2)
                time_improvements.append(improvement)
                speedups.append(ratio)
            else:
                time_improvements.append(0)
                speedups.append(1)

    overall_improvement_pct = round(
        sum(1 for x in time_improvements if x > 10) / len(time_improvements) * 100, 2
    )
    overall_speedup = round(statistics.mean(speedups) * 100, 2)

    print(f"### Evaluated examples: {len(time_improvements)}")
    print(f"### Overall >10% improvement rate: {overall_improvement_pct}%")
    print(f"### Overall average speedup*100: {overall_speedup}%")

if __name__ == '__main__':
    main()
