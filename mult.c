#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> 

/*
Referências usadas: 
        https://www.tutorialspoint.com/cprogramming/c_restrict_keyword.htm
        https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
        https://docs.google.com/presentation/d/1kpXBZZD9nxtl74MS5OD-PJbmNbTjaCburZL2gR_RzCo/edit?slide=id.p#slide=id.p
        https://docs.google.com/presentation/d/1wUHIvn6NPCyMyTVhJyeA_pY0vruJ4CmAVX23Xww1ftQ/edit?slide=id.p#slide=id.p
        https://docs.google.com/presentation/d/1gEQYj2_2Gx9tKHo__5uj8An1m717rryCH_CJqCA-99k/edit?slide=id.p#slide=id.p
        https://www.openmp.org/resources/refguides

Compilação:
    gcc -O3 -march=native -fopenmp mult.c -o mult

Execução:
    OMP_NUM_THREADS=8 ./mult 1000
*/

// --- Funções ---

double* allocate_matrix(int N) {
    double* mat = (double*)malloc(N * N * sizeof(double));
    return mat;
}

void initialize_matrix_zero(double* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = 0.0;
    }
}

void initialize_matrix_rand(double* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (double)rand() / RAND_MAX;
    }
}

void dgemm_sequential(const double* restrict a, const double* restrict b, double* restrict c, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = a[i*N+k];
            for (int j = 0; j < N; j++) {
                c[i*N+j] += r * b[k*N+j];
            }
        }
    }
}

void dgemm_parallel(const double* restrict a, const double* restrict b, double* restrict c, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = a[i*N+k];
            for (int j = 0; j < N; j++) {
                c[i*N+j] += r * b[k*N+j];
            }
        }
    }
}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "O tamanho da matriz precisa ser passado\n");
        return 1;
    }
    int N = atoi(argv[1]);
    if (N <= 0) {
        fprintf(stderr, "O tamanho deve ser positivo.\n");
        return 1;
    }

    printf("Benchmark para matrizes de tamanho %d x %d\n", N, N);

    // Alocação das matrizes
    double* A = allocate_matrix(N);
    double* B = allocate_matrix(N);
    double* C_seq = allocate_matrix(N);
    double* C_par = allocate_matrix(N);


    // Inicialização
    srand(123456);
    initialize_matrix_rand(A, N);
    initialize_matrix_rand(B, N);
    
    double time_seq, time_par;

    // --- Benchmark Sequencial ---
    printf("\n[1] Executando versão sequencial...\n");
    initialize_matrix_zero(C_seq, N); 
    double start_seq = omp_get_wtime();
    dgemm_sequential(A, B, C_seq, N);
    double end_seq = omp_get_wtime();
    time_seq = end_seq - start_seq;
    printf("Tempo sequencial: %f segundos\n", time_seq);

    // --- Benchmark Paralelo ---
    printf("\n[2] Executando versão paralela...\n");
    initialize_matrix_zero(C_par, N); 
    double start_par = omp_get_wtime();
    dgemm_parallel(A, B, C_par, N);
    double end_par = omp_get_wtime();
    time_par = end_par - start_par;
    printf("Tempo paralelo: %f segundos\n", time_par);
    
    // --- Speedup ---
    printf("\n[3] Resultado:\n");
    double speedup = time_seq / time_par;
    printf("Speedup: %.2fx\n", speedup);
    
    free(A);
    free(B);
    free(C_seq);
    free(C_par);

    return 0;
}