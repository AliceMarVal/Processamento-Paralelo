#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>    
#include <math.h>   
#include <sys/stat.h> 
#include <string.h> 

/*
    Compilação:
        gcc -O3 -march=native -fopenmp mult_par_blocked.c -o mult_par_blocked -lm
    
    Execução:
        ./mult_par_blocked
*/

#define BLOCK_SIZE 32

const double EPSILON = 1e-12;


double calculate_mean(const double* times, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += times[i];
    }
    return sum / n;
}

double calculate_stddev(const double* times, int n, double mean) {
    double sum_sq_diff = 0.0;
    for (int i = 0; i < n; i++) {
        sum_sq_diff += (times[i] - mean) * (times[i] - mean);
    }
    return sqrt(sum_sq_diff / n);
}

double calculate_max_relative_difference(const double* restrict c_seq, const double* restrict c_par, int N) {
    double max_diff = 0.0;
    long long num_elements = (long long)N * N;

    for (long long i = 0; i < num_elements; i++) {
        double numerator = fabs(c_seq[i] - c_par[i]);
        double denominator = fabs(c_seq[i]) + EPSILON;
        double relative_diff = numerator / denominator;
        
        if (relative_diff > max_diff) {
            max_diff = relative_diff;
        }
    }
    return max_diff;
}



double* allocate_matrix(int N) {
    double* mat = (double*)malloc((long long)N * N * sizeof(double));
    if (mat == NULL) {
        fprintf(stderr, "Falha na alocação de memória!\n");
        exit(EXIT_FAILURE);
    }
    return mat;
}

void initialize_matrix_zero(double* mat, int N) {
    for (long long i = 0; i < (long long)N * N; i++) {
        mat[i] = 0.0;
    }
}

void initialize_matrix_rand(double* mat, int N) {
    for (long long i = 0; i < (long long)N * N; i++) {
        mat[i] = (double)rand() / RAND_MAX;
    }
}


void dgemm_sequential_reference(const double* restrict a, const double* restrict b, double* restrict c, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = a[i*N+k];
            for (int j = 0; j < N; j++) {
                c[i*N+j] += r * b[k*N+j];
            }
        }
    }
}

void dgemm_parallel_blocked(const double* restrict a, const double* restrict b, double* restrict c, int N) {
    #pragma omp parallel for
    for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
        for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
            for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                
                int i_max = (i0 + BLOCK_SIZE > N) ? N : i0 + BLOCK_SIZE;
                for (int i = i0; i < i_max; i++) {
                    
                    int k_max = (k0 + BLOCK_SIZE > N) ? N : k0 + BLOCK_SIZE;
                    for (int k = k0; k < k_max; k++) {
                        
                        const double r = a[i * N + k];
                        
                        int j_max = (j0 + BLOCK_SIZE > N) ? N : j0 + BLOCK_SIZE;
                        for (int j = j0; j < j_max; j++) {
                            c[i * N + j] += r * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
}


int main() {
    srand(123456);

    const int N_sizes[] = {512, 1024, 2048, 4096};
    const int num_sizes = sizeof(N_sizes) / sizeof(N_sizes[0]);
    const int thread_counts[] = {2, 4, 8};
    const int num_thread_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    const int num_runs = 5;
    const char* base_output_dir = "resultados_par_blocked"; 

    mkdir(base_output_dir, 0777);

    for (int i_t = 0; i_t < num_thread_tests; i_t++) {
        int num_threads = thread_counts[i_t];
        omp_set_num_threads(num_threads); 

        char thread_dir_path[256];
        snprintf(thread_dir_path, 256, "%s/%d", base_output_dir, num_threads);
        mkdir(thread_dir_path, 0777);
        
        printf("\nIniciando benchmark paralelo com %d THREADS...\n", num_threads);

        for (int i_n = 0; i_n < num_sizes; i_n++) {
            int N = N_sizes[i_n];
            printf("  Testando N = %d...\n", N);

            // Alocar matrizes
            double* A = allocate_matrix(N);
            double* B = allocate_matrix(N);
            double* C_par = allocate_matrix(N);
            double* C_seq = allocate_matrix(N); 

            initialize_matrix_rand(A, N);
            initialize_matrix_rand(B, N);

            double runtimes[num_runs];
            double max_diff = 0.0;

            for (int run = 0; run < num_runs; run++) {
                initialize_matrix_zero(C_par, N);

                double start_time = omp_get_wtime();
                dgemm_parallel_blocked(A, B, C_par, N);
                double end_time = omp_get_wtime();

                runtimes[run] = end_time - start_time;
            }

            double mean = calculate_mean(runtimes, num_runs);
            double stddev = calculate_stddev(runtimes, num_runs, mean);
            printf("    Média: %f s, Desvio Padrão: %f s\n", mean, stddev);

            printf("    Calculando diferença numérica...\n");
            initialize_matrix_zero(C_seq, N);
            dgemm_sequential_reference(A, B, C_seq, N); 
            
            max_diff = calculate_max_relative_difference(C_seq, C_par, N);
            printf("    Diferença Relativa Máxima: %.15e\n", max_diff);

            char filepath[256];
            snprintf(filepath, 256, "%s/N_%d.txt", thread_dir_path, N);
            
            FILE *f_out = fopen(filepath, "w");
            if (f_out == NULL) {
                fprintf(stderr, "Erro ao abrir o arquivo de saída: %s\n", filepath);
                continue;
            }

            fprintf(f_out, "%f\n", mean);    
            fprintf(f_out, "%f\n", stddev);  
            fprintf(f_out, "%.15e\n", max_diff); 
            fclose(f_out);

            free(A);
            free(B);
            free(C_par);
            free(C_seq);
        }
    }

    return 0;
}