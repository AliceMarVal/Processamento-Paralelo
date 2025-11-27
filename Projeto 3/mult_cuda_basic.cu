#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <omp.h> 

/*
    Compilação:
        nvcc -Xcompiler -fopenmp  -O3 mult_cuda_basic.cu -o mult_cuda_basic -lm

    Execução:
        ./mult_cuda_basic
*/

#define BLOCK_SIZE 32
const double EPSILON = 1e-12;


double calculate_mean(const double* times, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += times[i];
    return sum / n;
}

double calculate_stddev(const double* times, int n, double mean) {
    double sum_sq_diff = 0.0;
    for (int i = 0; i < n; i++) sum_sq_diff += (times[i] - mean) * (times[i] - mean);
    return sqrt(sum_sq_diff / n);
}

double calculate_max_relative_difference(const double* c_ref, const double* c_test, int N) {
    double max_diff = 0.0;
    long long num_elements = (long long)N * N;
    for (long long i = 0; i < num_elements; i++) {
        double numerator = fabs(c_ref[i] - c_test[i]);
        double denominator = fabs(c_ref[i]) + EPSILON;
        double relative_diff = numerator / denominator;
        if (relative_diff > max_diff) max_diff = relative_diff;
    }
    return max_diff;
}

double* allocate_matrix(int N) {
    double* mat = (double*)malloc((long long)N * N * sizeof(double));
    if (mat == NULL) { fprintf(stderr, "Erro de alocação Host\n"); exit(1); }
    return mat;
}

void initialize_matrix_zero(double* mat, int N) {
    for (long long i = 0; i < (long long)N * N; i++) mat[i] = 0.0;
}

void initialize_matrix_rand(double* mat, int N) {
    for (long long i = 0; i < (long long)N * N; i++) mat[i] = (double)rand() / RAND_MAX;
}

void dgemm_sequential_reference(const double* a, const double* b, double* c, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = a[i*N+k];
            for (int j = 0; j < N; j++) {
                c[i*N+j] += r * b[k*N+j];
            }
        }
    }
}

__global__ void k_dgemm_basic(const double* A, const double* B, double* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum; 
    }
}

void dgemmCUDA(const double* h_A, const double* h_B, double* h_C, int N) {
    size_t size = (size_t)N * N * sizeof(double);
    double *d_A, *d_B, *d_C;
    

    // Alocar memória para as matrizes no diveice
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    
    // Copiar as matrizes do host para os respectivos espços de memória no divice
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    // Não tem motivo copiar C de Host -> Device

    // Definir a grade que se deseja utilizar
    dim3 dimBlock(32, 32);
    dim3 dimGrid(N/32, N/32);

    // Chamada ao kernel que implementa a GEMM
    k_dgemm_basic<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    

    // copiar a matriz resultante do device para o espaço de meória do host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    

    // Liberar a memória no divice
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main() {
    srand(123456);
    const int N_sizes[] = {512, 1024, 2048, 4096};
    const int num_sizes = 4;
    const int num_runs = 5;
    const char* output_dir = "resultados_cuda_basic";

    mkdir(output_dir, 0777);

    for (int i_n = 0; i_n < num_sizes; i_n++) {
        int N = N_sizes[i_n];
        printf("\nIniciando CUDA BASIC para N = %d...\n", N);

        double* A = allocate_matrix(N);
        double* B = allocate_matrix(N);
        double* C_cuda = allocate_matrix(N);
        double* C_ref = allocate_matrix(N);

        initialize_matrix_rand(A, N);
        initialize_matrix_rand(B, N);

        double runtimes[num_runs];

        // Benchmark
        for (int run = 0; run < num_runs; run++) {
            initialize_matrix_zero(C_cuda, N); 

            double start = omp_get_wtime();
            dgemmCUDA(A, B, C_cuda, N);
            double end = omp_get_wtime();
            
            runtimes[run] = end - start;
            printf("  Run %d/%d: %f s\n", run + 1, num_runs, runtimes[run]);
        }

        double mean = calculate_mean(runtimes, num_runs);
        double stddev = calculate_stddev(runtimes, num_runs, mean);
        printf("  Média: %f s, Desvio Padrão: %f s\n", mean, stddev);
 
        initialize_matrix_zero(C_ref, N);
        dgemm_sequential_reference(A, B, C_ref, N);
        
        double diff = calculate_max_relative_difference(C_ref, C_cuda, N);
        printf("  Diferença Relativa Máxima: %.15e\n", diff);

        char filepath[256];
        sprintf(filepath, "%s/N_%d.txt", output_dir, N);
        FILE *f = fopen(filepath, "w");
        if (f) {
            fprintf(f, "%f\n%f\n%.15e\n", mean, stddev, diff);
            fclose(f);
        } else {
            fprintf(stderr, "Erro ao abrir arquivo %s\n", filepath);
        }

        free(A); free(B); free(C_cuda); free(C_ref);
    }
    return 0;
}