import subprocess
import argparse
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def divisors(n):
    # get factors and their counts
    factors = {}
    nn = n
    i = 2
    while i*i <= nn:
        while nn % i == 0:
            factors[i] = factors.get(i, 0) + 1
            nn //= i
        i += 1
    if nn > 1:
        factors[nn] = factors.get(nn, 0) + 1

    primes = list(factors.keys())

    # generates factors from primes[k:] subset
    def generate(k):
        if k == len(primes):
            yield 1
        else:
            rest = generate(k+1)
            prime = primes[k]
            for factor in rest:
                prime_to_i = 1
                # prime_to_i iterates prime**i values, i being all possible exponents
                for _ in range(factors[prime] + 1):
                    yield factor * prime_to_i
                    prime_to_i *= prime

    # in python3, `yield from generate(0)` would also work
    for factor in generate(0):
        yield factor

def sorted_divisors(n, reverse=False):
    return sorted(divisors(n), reverse=reverse)

def multi_run(filter_inputs, filter_entries, filter_hashes, bits_per_input, dim1_block_size, dim2_block_size, max_bleach, save_option, num_repeats):
    # subprocess.run(['make', 'clean'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    # subprocess.run(['make', 'trainer'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    accuracies = []
    bleaches = []
    for i in range(num_repeats):
        result = single_run(filter_inputs, filter_entries, filter_hashes, bits_per_input, dim1_block_size, dim2_block_size, max_bleach, save_option)
        # print("\t", result)
        split_result = result.split(',')
        accuracy = float(split_result[3])
        bleach = int(split_result[4])
        accuracies.append(accuracy)
        bleaches.append(bleach)

    best_accuracy = max(accuracies)
    mean_accuracy = sum(accuracies) / len(accuracies)
    std_accuracy = (sum([(x - mean_accuracy) ** 2 for x in accuracies]) / len(accuracies)) ** 0.5

    best_result = {
        'dim1_block_size': dim1_block_size,
        'dim2_block_size': dim2_block_size,
        'bleach': bleaches[accuracies.index(max(accuracies))],
        'best_accuracy': best_accuracy,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
    }

    return best_result

def single_run(filter_inputs, filter_entries, filter_hashes, bits_per_input, dim1_block_size, dim2_block_size, max_bleach, save_option):
    trainer_output = subprocess.run(['./trainer', str(filter_inputs), str(filter_entries), str(filter_hashes), str(bits_per_input), str(dim1_block_size), str(dim2_block_size), str(max_bleach), str(save_option)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    filtered_output = [line for line in trainer_output.stdout.decode().split('\n') if 'test_accuracy' in line]
    assert(len(filtered_output) == 1)
    return filtered_output[0]

# Big blocks first
def get_block_sizes(n, bits=1, min=None, max=None):
    bits_total = n * bits
    if min is None: min = 1
    if max is None: max = bits_total
    return [block_size for block_size in sorted_divisors(bits_total, reverse=True) if block_size >= min and block_size <= max]

def print_res(res):
    print(f"{res['dim1_block_size']} x {res['dim2_block_size']} (bleach: {res['bleach']})")
    print(f"\t{bcolors.OKGREEN}-> {res['best_accuracy']:2.2f}{bcolors.ENDC} (mean {res['mean_accuracy']:2.2f}) (std: {res['std_accuracy']:1.2f})")

def perf1d_single(label, num_repeat, filter_inputs, filter_entries, filter_hashes, bits_per_input, max_bleach, save_option, min_block_size=None, reorder_first=False):
    if reorder_first:
        block_sizes = get_block_sizes(n=784, bits=1, min=min_block_size)
    else:
        block_sizes = get_block_sizes(n=784, bits=bits_per_input, min=min_block_size)
    print(f"{label}")
    print(f"1d block sizes {block_sizes}")
    for i in block_sizes:
        best_res = multi_run(filter_inputs, filter_entries, filter_hashes, bits_per_input, i, 1, max_bleach, save_option, num_repeat)
        print_res(best_res)

def perf1d(num_repeats=1, min_block_size=None, reorder_first=False):
    perf1d_single("MNIST-Small", num_repeats, 28, 1024, 2, 2, 13, 0, min_block_size, reorder_first)
    perf1d_single("MNIST-Medium", num_repeats, 28, 2048, 2, 3, 11, 0, min_block_size, reorder_first)
    perf1d_single("MNIST-Large", num_repeats, 49, 8192, 4, 6, 11, 0, min_block_size, reorder_first)

def perf2d_single(label, num_repeats, filter_inputs, filter_entries, filter_hashes, bits_per_input, max_bleach, save_option, min_block_size=None, reorder_first=False):
    if reorder_first:
        col_block_sizes = get_block_sizes(n=28, bits=1, min=min_block_size)
        row_block_sizes = get_block_sizes(n=28, bits=1, min=min_block_size)
    else:
        col_block_sizes = get_block_sizes(n=28, bits=1, min=min_block_size)
        row_block_sizes = get_block_sizes(n=28, bits=bits_per_input, min=min_block_size)
    print(f"{label}")
    print(f"2d block sizes {col_block_sizes}, {row_block_sizes}")
    for i in col_block_sizes:
        for j in row_block_sizes:
            best_res = multi_run(filter_inputs, filter_entries, filter_hashes, bits_per_input, i, j, max_bleach, save_option, num_repeats)
            print_res(best_res)

def perf2d(num_repeats=1, min_block_size=None, reorder_first=False):
    perf2d_single("MNIST-Small", num_repeats, 28, 1024, 2, 2, 11, 0, min_block_size, reorder_first)
    perf2d_single("MNIST-Medium", num_repeats, 28, 2048, 2, 3, 11, 0, min_block_size)
    perf2d_single("MNIST-Large", num_repeats, 49, 8192, 4, 6, 11, 0, min_block_size)

def read_arguments():
    parser = argparse.ArgumentParser(description="Accuracy benchmarking Cbthowen models")
    parser.add_argument("--reps", default=1, required=False, type=int,\
            help="Number of repeats to run for each block size")
    args = parser.parse_args()
    return args

def remake(encoding="STRIDED_ENCODING", reorder="REORDER_FIRST"):
    subprocess.run(['make', 'clean'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    env = os.environ.copy()
    env["ENCODING"] = encoding
    env["REORDER"] = reorder
    subprocess.run(['make', 'trainer'], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"\n======== make trainer ENCODING={encoding} REORDER={reorder} ========\n")

def main():
    args = read_arguments()

    # remake("STRIDED_ENCODING", "REORDER_SECOND")
    # perf1d(args.reps, 100, False)
    # remake("LOCAL_STRIDED_ENCODING", "REORDER_SECOND")
    # perf1d(args.reps, 100, False)
    remake("LOCAL_ENCODING", "REORDER_SECOND")
    # perf1d(args.reps, 200, False)

    perf2d(args.reps, 7, False)

if __name__ == "__main__":
    main()
