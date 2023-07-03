import subprocess
import argparse

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

def sorted_divisors(n):
    return sorted(divisors(n))

def multi_run(filter_inputs, filter_entries, filter_hashes, bits_per_input, block_size_div, max_bleach, save_option, num_repeats):
    subprocess.run(['make', 'clean'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run(['make', 'trainer'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    best_result = ""
    best_accuracy = 0.0
    for i in range(num_repeats):
        result = single_run(filter_inputs, filter_entries, filter_hashes, bits_per_input, block_size_div, max_bleach, save_option)
        accuracy = float(result.split(',')[4])

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_result = result

    return best_result

def single_run(filter_inputs, filter_entries, filter_hashes, bits_per_input, block_size_div, max_bleach, save_option):
    trainer_output = subprocess.run(['./trainer', str(filter_inputs), str(filter_entries), str(filter_hashes), str(bits_per_input), str(block_size_div), str(max_bleach), str(save_option)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    filtered_output = [line for line in trainer_output.stdout.decode().split('\n') if 'test_accuracy' in line]
    assert(len(filtered_output) == 1)
    return filtered_output[0]

def perf(num_repeats=1):
    block_size_divs = sorted_divisors(784 * 2)
    print("MNIST-Small")
    for i in block_size_divs:
        line = multi_run(28, 1024, 2, 2, i, 12, 0, num_repeats)
        print(line)
            
    block_size_divs = sorted_divisors(784 * 6)
    print("MNIST-Large")
    for i in block_size_divs:
        line = multi_run(49, 8192, 4, 6, i, 12, 1, num_repeats)
        print(line)

def read_arguments():
    parser = argparse.ArgumentParser(description="Accuracy benchmarking Cbthowen models")
    parser.add_argument("--reps", default=1, required=False, type=int,\
            help="Number of repeats to run for each block size")
    args = parser.parse_args()
    return args

def main():
    args = read_arguments()

    perf(args.reps)

if __name__ == "__main__":
    main()
