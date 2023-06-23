#include "data_loader.h"
#include "data_manager.h"

void binarize_and_write(size_t num_samples, size_t bits_per_input, u8_matrix_t dataset, char* output_path) {
    printf("Binarizing infimnist dataset with %zu bits per input\n", bits_per_input);
    u8_matrix_t binarized_dset;
    matrix_u8_init(&binarized_dset, num_samples, MNIST_IM_SIZE * bits_per_input);
    binarize_matrix(binarized_dset, dataset, MNIST_IM_SIZE, num_samples, bits_per_input);

    printf("Saving binarized infimnist\n");
    write_dataset(output_path, binarized_dset, num_samples, MNIST_IM_SIZE * bits_per_input);
}

int main(int argc, char *argv[]) {                              
    if(argc < 5) {
        printf("Error: usage: %s dataset num_samples bits_per_input output_path\n", argv[0]);
        printf("\tExample usage: %s infimnist 60000 2 ./binarized_mnist_train.dat\n", argv[0]);
        printf("\tPossible datasets: mnist_test, infimnist\n");
        printf("\tPossible num_samples: 1-10K (mnist_test), 1-8M (infimnist)\n");

        return 1;
    }

    size_t num_samples = atoi(argv[2]);
    size_t bits_per_input = atoi(argv[3]);
    char* dataset_name = argv[1];
    char* output_path = argv[4];

    if(dataset_name[0] == 'm') {
        if(num_samples > 10000) {
            printf("Error: mnist_test has only 10K samples\n");
            return 1;
        }

        printf("Loading mnist_test\n");
        u8_matrix_t mnist_patterns;
        matrix_u8_init(&mnist_patterns, num_samples, MNIST_IM_SIZE);
        unsigned char* mnist_labels = calloc(num_samples, sizeof(*mnist_labels));
        load_mnist_test(mnist_patterns, mnist_labels, num_samples);

        binarize_and_write(num_samples, bits_per_input, mnist_patterns, output_path);
    } else if(dataset_name[0] == 'i') {
        if(num_samples > 8100000) {
            printf("Error: infimnist has only 8M1 samples\n");
            return 1;
        }

        printf("Loading infimnist\n"); 
        u8_matrix_t infimnist_patterns;
        matrix_u8_init(&infimnist_patterns, num_samples, MNIST_IM_SIZE);
        unsigned char* infimnist_labels = calloc(num_samples, sizeof(*infimnist_labels));
        load_infimnist(infimnist_patterns, infimnist_labels, num_samples);

        binarize_and_write(num_samples, bits_per_input, infimnist_patterns, output_path);
    } else {
        printf("Error: unknown dataset %s\n", dataset_name);
        return 1;
    }
}
