#include "data_manager.h"

#define READ_PTR(ptr, fp) FREAD_CHECK(ptr, sizeof(*ptr), 1, fp)
#define READ_FIELD(structure, name, fp) FREAD_CHECK(&(structure)->name, sizeof((structure)->name), 1, fp)
#define READ_BUFFER(structure, name, num, fp) FREAD_CHECK((structure)->name, sizeof(*(structure)->name), num, fp)

void read_model(const char* filename, model_t* model) {
    FILE* f = fopen(filename, "r");

    READ_FIELD(model, pad_zeros, f);
    READ_FIELD(model, num_inputs_total, f);
    READ_FIELD(model, bits_per_input, f);
    READ_FIELD(model, num_classes, f);
    READ_FIELD(model, filter_inputs, f);
    READ_FIELD(model, filter_entries, f);
    READ_FIELD(model, filter_hashes, f);
    READ_FIELD(model, bleach, f);

    model_init(model, model->num_inputs_total, model->num_classes, model->filter_inputs, model->filter_entries, model->filter_hashes, model->bits_per_input, model->bleach);

    READ_BUFFER(model, input_order, model->num_inputs_total, f);

    read_matrix(f, model->hash_parameters, model->filter_hashes * model->filter_inputs);
    read_tensor(f, model->filters, model->num_classes * model->num_filters * model->filter_entries);

    fclose(f);
}

void read_dataset(const char* filename, u8_matrix_t dataset, size_t* num_samples, size_t* sample_size) {
    FILE* f = fopen(filename, "r");

    READ_PTR(num_samples, f);
    READ_PTR(sample_size, f);

    read_matrix_u8(f, dataset, *num_samples * *sample_size);

    fclose(f);
}

void read_dataset_partial(const char* filename, u8_matrix_t dataset, size_t num_samples_to_fetch, size_t* num_samples_total, size_t* sample_size) {
    FILE* f = fopen(filename, "r");

    READ_PTR(num_samples_total, f);
    READ_PTR(sample_size, f);

    assert(num_samples_to_fetch <= *num_samples_total);

    read_matrix_u8(f, dataset, num_samples_to_fetch * *sample_size);

    fclose(f);
}

void read_matrix(FILE* f, u16_matrix_t matrix, size_t size) {
    READ_FIELD(&matrix, stride, f);

    READ_BUFFER(&matrix, data, size, f);
}

void read_matrix_u8(FILE* f, u8_matrix_t matrix, size_t size) {
    READ_FIELD(&matrix, stride, f);

    READ_BUFFER(&matrix, data, size, f);
}

void read_tensor(FILE* f, u16_tensor3d_t tensor, size_t size) {
    READ_FIELD(&tensor, stride1, f);
    READ_FIELD(&tensor, stride2, f);

    READ_BUFFER(&tensor, data, size, f);
}

#define SAVE_VAR(var, fp) FWRITE_CHECK(&var, sizeof(var), 1, fp)
#define SAVE_FIELD(structure, name, fp) FWRITE_CHECK(&(structure)->name, sizeof((structure)->name), 1, fp)
#define SAVE_BUFFER(structure, name, num, fp) FWRITE_CHECK((structure)->name, sizeof(*(structure)->name), num, fp)

void write_model(const char* filename, model_t* model) {
    FILE* f = fopen(filename, "w");

    SAVE_FIELD(model, pad_zeros, f);
    SAVE_FIELD(model, num_inputs_total, f);
    SAVE_FIELD(model, bits_per_input, f);
    SAVE_FIELD(model, num_classes, f);
    SAVE_FIELD(model, filter_inputs, f);
    SAVE_FIELD(model, filter_entries, f);
    SAVE_FIELD(model, filter_hashes, f);
    SAVE_FIELD(model, bleach, f);

    SAVE_BUFFER(model, input_order, model->num_inputs_total, f);

    write_matrix(f, model->hash_parameters, model->filter_hashes * model->filter_inputs);
    write_tensor(f, model->filters, model->num_classes * model->num_filters * model->filter_entries);

    fclose(f);
}

void write_dataset(const char* filename, u8_matrix_t dataset, size_t num_samples, size_t sample_size) {
    FILE* f = fopen(filename, "w");

    SAVE_VAR(num_samples, f);
    SAVE_VAR(sample_size, f);

    write_matrix_u8(f, dataset, num_samples * sample_size);

    fclose(f);
}

void write_matrix(FILE* f, u16_matrix_t matrix, size_t size) {
    SAVE_FIELD(&matrix, stride, f);

    SAVE_BUFFER(&matrix, data, size, f);
}

void write_matrix_u8(FILE* f, u8_matrix_t matrix, size_t size) {
    SAVE_FIELD(&matrix, stride, f);

    SAVE_BUFFER(&matrix, data, size, f);
}

void write_tensor(FILE* f, u16_tensor3d_t tensor, size_t size) {
    SAVE_FIELD(&tensor, stride1, f);
    SAVE_FIELD(&tensor, stride2, f);

    SAVE_BUFFER(&tensor, data, size, f);
}
