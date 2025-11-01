#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>    
#include <math.h>   
#include <sys/stat.h> 
#include <string.h> 
#include <mpi.h>    

/*
    Compilação:
        mpicc -fopenmp -O3 -march=native mult_mpi.c -o mult_mpi -lm

    Execução:

        mpirun -np 2 ./mult_mpi
        mpirun -np 4 ./mult_mpi
        mpirun -np 8 ./mult_mpi
*/

#define BLOCK_SIZE 32

const double EPSILON = 1e-12; // 


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


double* allocate_matrix(int N, int M) {
    double* mat = (double*)malloc((long long)N * M * sizeof(double));
    if (mat == NULL) {
        fprintf(stderr, "Falha na alocação de memória!\n");
        MPI_Abort(MPI_COMM_WORLD, 1); // Aborta todos os processos
    }
    return mat;
}

void initialize_matrix_zero(double* mat, int N, int M) {
    for (long long i = 0; i < (long long)N * M; i++) {
        mat[i] = 0.0;
    }
}

void initialize_matrix_rand(double* mat, int N, int M) {
    for (long long i = 0; i < (long long)N * M; i++) {
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

void dgemm_local_blocked(const double* restrict a_local, const double* restrict b, double* restrict c_local, int local_rows, int N) {

    for (int i0 = 0; i0 < local_rows; i0 += BLOCK_SIZE) {
        for (int k0 = 0; k0 < N; k0 += BLOCK_SIZE) {
            for (int j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                
                int i_max = (i0 + BLOCK_SIZE > local_rows) ? local_rows : i0 + BLOCK_SIZE;
                for (int i = i0; i < i_max; i++) {
                    
                    int k_max = (k0 + BLOCK_SIZE > N) ? N : k0 + BLOCK_SIZE;
                    for (int k = k0; k < k_max; k++) {
                        
                        const double r = a_local[i * N + k]; 
                        
                        int j_max = (j0 + BLOCK_SIZE > N) ? N : j0 + BLOCK_SIZE;
                        for (int j = j0; j < j_max; j++) {
                            c_local[i * N + j] += r * b[k * N + j]; 
                        }
                    }
                }
            }
        }
    }
}


int main(int argc, char *argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    const int N_sizes[] = {512, 1024, 2048, 4096}; // 
    const int num_sizes = sizeof(N_sizes) / sizeof(N_sizes[0]);
    const int num_runs = 5;
    const char* base_output_dir = "resultados_mpi";
    char output_dir[256];

    if (rank == 0) {
        printf("Iniciando benchmark MPI com %d Processos...\n", size);
        mkdir(base_output_dir, 0777);
        snprintf(output_dir, 256, "%s/%d", base_output_dir, size);
        mkdir(output_dir, 0777);
    }

    for (int i_n = 0; i_n < num_sizes; i_n++) {
        int N = N_sizes[i_n];

        if (N % size != 0) {
            if (rank == 0) {
                fprintf(stderr, "  AVISO: N=%d não é divisível por P=%d. Pulando este teste.\n", N, size);
            }
            continue;
        }

        int local_rows = N / size;
        long long local_elements = (long long)local_rows * N;
        long long total_elements = (long long)N * N;

        double* A = NULL;
        double* C_par = NULL; 
        double* C_seq = NULL; 
        
        if (rank == 0) {
            A = allocate_matrix(N, N);
            C_par = allocate_matrix(N, N);
            C_seq = allocate_matrix(N, N);
        }
        
        double* B = allocate_matrix(N, N);
        double* A_local = allocate_matrix(local_rows, N);
        double* C_local = allocate_matrix(local_rows, N);

        if (rank == 0) {
            srand(123456); 
            initialize_matrix_rand(A, N, N);
            initialize_matrix_rand(B, N, N);
            
            initialize_matrix_zero(C_seq, N, N);
            dgemm_sequential(A, B, C_seq, N);
        }

        double runtimes[num_runs];
        
        if (rank == 0) {
            printf("  Testando N = %d (P = %d)...\n", N, size);
        }

        for (int run = 0; run < num_runs; run++) {
            initialize_matrix_zero(C_local, local_rows, N);
            
            MPI_Barrier(MPI_COMM_WORLD);
            double start_time = omp_get_wtime();

            MPI_Scatter(A, local_elements, MPI_DOUBLE,
                        A_local, local_elements, MPI_DOUBLE,
                        0, MPI_COMM_WORLD); 
            
            MPI_Bcast(B, total_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

            dgemm_local_blocked(A_local, B, C_local, local_rows, N);

            MPI_Gather(C_local, local_elements, MPI_DOUBLE,
                       C_par, local_elements, MPI_DOUBLE,
                       0, MPI_COMM_WORLD); 

            MPI_Barrier(MPI_COMM_WORLD);
            double end_time = omp_get_wtime();

            if (rank == 0) {
                runtimes[run] = end_time - start_time;
            }
        }

        if (rank == 0) {
            double mean = calculate_mean(runtimes, num_runs);
            double stddev = calculate_stddev(runtimes, num_runs, mean);
            printf("    Média: %f s, Desvio Padrão: %f s\n", mean, stddev);

            double max_diff = calculate_max_relative_difference(C_seq, C_par, N);
            printf("    Diferença Relativa Máxima: %.15e\n", max_diff);

            char filepath[256];
            snprintf(filepath, 256, "%s/N_%d.txt", output_dir, N);
            
            FILE *f_out = fopen(filepath, "w");
            if (f_out == NULL) {
                fprintf(stderr, "Erro ao abrir o arquivo de saída: %s\n", filepath);
            } else {
                fprintf(f_out, "%f\n", mean);    
                fprintf(f_out, "%f\n", stddev);  
                fprintf(f_out, "%.15e\n", max_diff); 
                fclose(f_out);
            }
        }

        if (rank == 0) {
            free(A);
            free(C_par);
            free(C_seq);
        }
        free(B);
        free(A_local);
        free(C_local);
    }

    MPI_Finalize(); 

    if (rank == 0) {
        printf("\nBenchmark MPI com %d processos concluído.\n", size);
    }

    return 0;
}