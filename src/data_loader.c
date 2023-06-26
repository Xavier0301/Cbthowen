#include "data_loader.h"

u8_matrix_t train_images; // Of shape (#Samples, 784) -> flattened
u8_matrix_t test_images; // Of shape (#Samples, 784) -> flattened
uint8_t* train_labels; // Of shape (#Samples,) 
uint8_t* test_labels; // Of shape (#Samples,)

u8_matrix_t binarized_train; // Of shape (#Samples, 784 * bits_per_pixel)
u8_matrix_t binarized_test; // Of shape (#Samples, 784 * bits_per_pixel)

u8_matrix_t reordered_binarized_train; // Of shape (#Samples, 784 * bits_per_pixel)
u8_matrix_t reordered_binarized_test; // Of shape (#Samples, 784 * bits_per_pixel)

void reverse_bytes(uint32_t* element) {
    uint8_t tmp;
    uint8_t* ptr = (uint8_t*) element;

    tmp = ptr[0];
    ptr[0] = ptr[3];
    ptr[3] = tmp;

    tmp = ptr[1];
    ptr[1] = ptr[2];
    ptr[2] = tmp;
}

void read_mnist_file(char* file_path, size_t num_samples, size_t stride, size_t len_info, uint8_t* data, uint32_t* info, size_t offset) {
    FILE* fd = fopen(file_path, "r");

    if(fd == NULL) printf("Not able to read the file at path %s\n", file_path);
    
    size_t fread_res = fread(info, sizeof(uint32_t), len_info, fd);
    assert(fread_res > 0);
    assert(num_samples <= info[1]);
    for(size_t it = 0; it < len_info; ++it) reverse_bytes(info + it);
    
    if(offset > 0)
        fseek(fd, offset * stride, SEEK_CUR);
    fread_res = fread(data, sizeof(*data), num_samples * stride, fd);
    assert(fread_res > 0);

    fclose(fd);
}

void load_mnist_file(u8_matrix_t patterns, uint8_t* labels, char* image_path, char* label_path, size_t num_samples, size_t offset) {
    uint32_t* info_buffer = calloc(MNIST_LEN_INFO_IMAGE, sizeof(*info_buffer));

    read_mnist_file(image_path, num_samples, MNIST_IM_SIZE, MNIST_LEN_INFO_IMAGE, patterns.data, info_buffer, offset);
    assert(info_buffer[0] == 2051);
    printf("Image info: %d %d %d %d\n", info_buffer[0], info_buffer[1], info_buffer[2], info_buffer[3]);

    read_mnist_file(label_path, num_samples, 1, MNIST_LEN_INFO_LABEL, labels, info_buffer, offset);  
    assert(info_buffer[0] == 2049);

    free(info_buffer);
}

void load_mnist_train(u8_matrix_t patterns, uint8_t* labels, size_t num_samples) {
    load_mnist_file(patterns, labels, MNIST_TRAIN_IMAGE, MNIST_TRAIN_LABEL, num_samples, 0);
}

void load_mnist_test(u8_matrix_t patterns, uint8_t* labels, size_t num_samples) {
    load_mnist_file(patterns, labels, MNIST_TEST_IMAGE, MNIST_TEST_LABEL, num_samples, 0);
}

void load_infimnist(u8_matrix_t patterns, uint8_t* labels, size_t num_samples) {
    load_mnist_file(patterns, labels, INFIMNIST_PATTERNS, INFIMNIST_LABELS, num_samples, 0);
}

void load_infimnist_labels(uint8_t* labels, size_t num_samples) {
    uint32_t info_buffer[MNIST_LEN_INFO_LABEL];

    read_mnist_file(INFIMNIST_LABELS, num_samples, 1, MNIST_LEN_INFO_LABEL, labels, info_buffer, 0);  
    assert(info_buffer[0] == 2049);    
}

void load_mnist_train_offset(u8_matrix_t patterns, uint8_t* labels, size_t num_samples, size_t offset) {
    load_mnist_file(patterns, labels, MNIST_TRAIN_IMAGE, MNIST_TRAIN_LABEL, num_samples, offset);
}

void load_mnist_test_offset(u8_matrix_t patterns, uint8_t* labels, size_t num_samples, size_t offset) {
    load_mnist_file(patterns, labels, MNIST_TEST_IMAGE, MNIST_TEST_LABEL, num_samples, offset);
}

void load_infimnist_offset(u8_matrix_t patterns, uint8_t* labels, size_t num_samples, size_t offset) {
    load_mnist_file(patterns, labels, INFIMNIST_PATTERNS, INFIMNIST_LABELS, num_samples, offset);
}

void load_infimnist_labels_offset(uint8_t* labels, size_t num_samples, size_t offset) {
    uint32_t info_buffer[MNIST_LEN_INFO_LABEL];

    read_mnist_file(INFIMNIST_LABELS, num_samples, 1, MNIST_LEN_INFO_LABEL, labels, info_buffer, offset);  
    assert(info_buffer[0] == 2049);    
}

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0') 

void print_pixel(uint8_t value, int raw) {
    if(raw == 1) printf(BYTE_TO_BINARY_PATTERN" ", BYTE_TO_BINARY(value));
    else if(raw == 2) printf("%d ", value);
    else {
        char map[10]= " .,:;ox%#@";
        size_t index = (255 - value) * 10 / 256;
        printf("%c ", map[index]);
    }
}

void print_binarized_value(uint8_t value) {
    char map[2]= " @";
    printf("%c ", map[value - 1]);
}

void print_mnist_image_(u8_matrix_t images, uint8_t* labels, size_t index, int raw) {
    printf("Image %zu (Label %d)\n", index, labels[index]);
    for (size_t j = 0; j < MNIST_IM_SIZE; ++j) {
        print_pixel(*MATRIX(images, index, j), raw);
        if ((j+1) % 28 == 0) putchar('\n');
    }
    putchar('\n');
}

void print_binarized_image(u8_matrix_t m, uint8_t* labels, size_t index, size_t num_bits) {
    printf("Image %zu (Label %d) (Binarized)\n", index, labels[index]);
    for (size_t j = 0; j < MNIST_IM_SIZE; ++j) {
        char value = *MATRIX(m, index, j * num_bits);
        for(size_t b = 1; b < num_bits; ++b)
            value |= (*MATRIX(m, index, j * num_bits + b) << b);

        print_binarized_value(value);
        if ((j+1) % 28 == 0) putchar('\n');
    } 
    putchar('\n'); 
}

void print_binarized_image_raw(u8_matrix_t m, uint8_t* labels, size_t index, size_t num_bits) {
    if(labels != NULL)
        printf("Binarized image %zu (Label %d) (Binarized)\n", index, labels[index]);
    else
        printf("Binarized image %zu (Binarized)\n", index);
    for (size_t j = 0; j < MNIST_IM_SIZE * num_bits; ++j) {
        char value = *MATRIX(m, index, j);
        printf("%d ", value);
        if ((j+1) % 28 == 0) putchar('\n');
    }
}

void print_image(u8_matrix_t m, uint8_t* labels, size_t index) {
    print_mnist_image_(m, labels, index, 0);
}

void print_image_raw(u8_matrix_t m, uint8_t* labels, size_t index) {
    print_mnist_image_(m, labels, index, 1);
}

uint8_t thermometer_encode(uint8_t val, double mean, double std, size_t num_bits, double* skews, uint8_t* encodings) {
    size_t skew_index = 0;
    for(; skew_index < num_bits && val >= skews[skew_index] * std + mean; ++skew_index);

    // printf("val: %d, index: %d\n", val, skew_index);
        
    return encodings[skew_index];
}

void binarize_sample2(u8_matrix_t result, u8_matrix_t dataset, size_t sample_it, size_t num_bits, double* mean, double* variance, double* skews, uint8_t* encodings) { 
    for(size_t bit_it = 0; bit_it < num_bits; ++bit_it) {
        for(size_t offset_it = 0; offset_it < dataset.stride; ++offset_it) {
            char packed_encoding = thermometer_encode(*MATRIX(dataset, sample_it, offset_it), mean[offset_it], sqrt(variance[offset_it]), num_bits, skews, encodings);
            *MATRIX(result, sample_it, bit_it*dataset.stride + offset_it) = (packed_encoding >> bit_it) & 0x1;
        }
    }
}

void binarize_sample(u8_matrix_t result, u8_matrix_t dataset, size_t sample_it, size_t num_bits, double* mean, double* variance, double* skews, uint8_t* encodings) {
    for(size_t offset_it = 0; offset_it < dataset.stride; ++offset_it) {
        char packed_encoding = thermometer_encode(*MATRIX(dataset, sample_it, offset_it), mean[offset_it], sqrt(variance[offset_it]), num_bits, skews, encodings);
        // printf("packed "BYTE_TO_BINARY_PATTERN"\n", BYTE_TO_BINARY(packed_encoding));
        for(size_t bit_it = 0; bit_it < num_bits; ++bit_it) {
            // char x = (packed_encoding >> bit_it) & 0x1;
            // printf("    "BYTE_TO_BINARY_PATTERN"\n", BYTE_TO_BINARY(x));
            *MATRIX(result, sample_it, offset_it*num_bits + bit_it) = (packed_encoding >> bit_it) & 0x1;
        }
    }
}

void binarize_matrix_meanvar(u8_matrix_t result, u8_matrix_t dataset, double* mean, double* variance, size_t sample_size, size_t num_samples, size_t num_bits) {
    double skews[num_bits];
    for(size_t it = 0; it < num_bits; ++it) {
        skews[it] = gauss_inv((((double) (it + 1))) / (((double) num_bits + 1)));
        // printf("skew: %lf\n", skews[it]);
    }

    uint8_t encodings[9] = { 
        0b00000000, 0b00000001, 0b00000011, 
        0b00000111, 0b00001111, 0b00011111,
        0b00111111, 0b01111111, 0b11111111
    };

    for(size_t sample_it = 0; sample_it < num_samples; ++sample_it)
        binarize_sample2(result, dataset, sample_it, num_bits, mean, variance, skews, encodings);
}

void binarize_matrix(u8_matrix_t result, u8_matrix_t dataset, size_t sample_size, size_t num_samples, size_t num_bits) {
    double skews[num_bits];
    for(size_t it = 0; it < num_bits; ++it) {
        skews[it] = gauss_inv((((double) (it + 1))) / (((double) num_bits + 1)));
        // printf("skew: %lf\n", skews[it]);
    }

    uint8_t encodings[9] = {
        0b00000000,
        0b00000001,
        0b00000011,
        0b00000111,
        0b00001111,
        0b00011111,
        0b00111111,
        0b01111111,
        0b11111111
    };
    for(size_t it = 0; it < num_bits + 1; ++it) {
        // encodings[it] = (((uint8_t) 0xff) << it) & (((uint8_t) 0xff) >> (8 - num_bits));
        // printf("encoding: "BYTE_TO_BINARY_PATTERN"\n", BYTE_TO_BINARY(encodings[it]));
    }

    double mean[sample_size];
    double variance[sample_size];

    matrix_u8_mean(mean, dataset, sample_size, num_samples);
    matrix_u8_variance(variance, dataset, sample_size, num_samples, mean);

    // printf("Mean:");
    // for(size_t it = 0; it < sample_size; ++it) {
    //     if(it % 28 == 0)
    //         printf("\n");
    //     printf("%3.1f ", mean[it] / 255.0);
    // }

    // printf("\nVariance:");
    // for(size_t it = 0; it < sample_size; ++it) {
    //     if(it % 28 == 0)
    //         printf("\n");
    //     printf("%3.1f ", sqrt(variance[it]) / 255.0);
    // }

    for(size_t sample_it = 0; sample_it < num_samples; ++sample_it)
        binarize_sample2(result, dataset, sample_it, num_bits, mean, variance, skews, encodings);
}

void fill_input_random(uint8_t* input, size_t input_length) {
    for(size_t it = 0; it < input_length; ++it) {
        input[it] = rand() % 2;
    }
}

void reorder_dataset(u8_matrix_t result, u8_matrix_t dataset, uint16_t* order, size_t num_samples, size_t num_elements) {
    for(size_t it = 0; it < num_samples; ++it) {
        reorder_array(MATRIX_AXIS1(result, it), MATRIX_AXIS1(dataset, it), order, num_elements);
    }
}

