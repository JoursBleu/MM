#include <immintrin.h>
#include <sys/time.h>
#include <stdio.h>

#include <mkl.h>

#define MM 4000
#define KK 4000
#define NN 4000

#define BLOCK_M 64
#define BLOCK_K 64
#define BLOCK_N 64

#define SIMD_SIZE 16

#define SIMD_BLOCK_M BLOCK_M/SIMD_SIZE
#define SIMD_BLOCK_K BLOCK_K/SIMD_SIZE
#define SIMD_BLOCK_N BLOCK_N/SIMD_SIZE

// L1 (32K) should be enough for a 8K(16*512)

void aligned_MM(float *A, float *B, float *C, int M, int K, int N) {
  int I, J, L;
  int i, j, l;

  for(I=0; I<M; I++)
  for(J=0; J<K; J+=SIMD_SIZE)
  for(L=0; L<M; L+=SIMD_SIZE) {
    __m512 a;
    __m512 b, c;
    float *A_ptr = A + I*K + J;
    float *B_ptr = B + J*N + K;
    float *C_ptr = C + I*N + L;

    c = _mm512_load_ps(C_ptr);
    for(j=0; j<SIMD_SIZE; j++) {
      a = _mm512_set1_ps(A_ptr[j]);
      b = _mm512_load_ps(B_ptr + j*N);
      c += _mm512_mul_ps (a, b);
    }
    _mm512_store_ps(C_ptr, c);
  }
}

void Block_C_SIMD(const float *A, const float *B, float *C, const int m, const int n, const int M, const int K, const int N) {
  int i, j, l;
  int align_K = BLOCK_K*(K/BLOCK_K);

  __m512 b[SIMD_BLOCK_N];
  __m512 c[BLOCK_M*SIMD_BLOCK_N] = {0.};

  for(l=0; l<K; l++) {
    for(j=0; j<SIMD_BLOCK_M; j++)
      b[j] = _mm512_load_ps(&B[l*N+j*SIMD_SIZE]);

    for(i=0; i<m; i++) {
      __m512 a = _mm512_set1_ps(A[i*K+l]);
      for(j=0; j<SIMD_BLOCK_M; j++) {
        c[i*SIMD_BLOCK_N+j] += a*b[j];
      }
    }
  }

  for(i=0; i<m; i++)
  for(j=0; j<SIMD_BLOCK_M; j++) {
    __m512 cc = _mm512_load_ps(&C[i*M+j*SIMD_SIZE]);
    cc += c[i*SIMD_BLOCK_N+j];
    _mm512_store_ps(&C[i*M+j*SIMD_SIZE],cc);
  }
}

void MatMul(float *A, float *B, float *C, int M, int K, int N) {
  int I, J, L;
  
  for(I=0; I<M; I++)
  for(L=0; L<N; L++) {
    float c = 0.;
    for(J=0; J<K; J++)
      c += A[I*K + J] * B[J*N + K];
    C[I*N + L] += c;
  }
}

void Block_AB(float *A, float *B, float *C, const int m, const int n, const int k, int M, int K, int N) {
  int i, j, l;
  float a[BLOCK_M*BLOCK_K], b[BLOCK_N*BLOCK_K];
  for(i=0; i<k; i++)
  for(l=0; l<m; l++)
    a[i] = A[l*K+i];
  for(l=0; l<k; l++)
  for(i=0; i<n; i++)
    b[i] = B[l*N+i];
  
}

void Block_C(float *A, float *B, float *C, const int m, const int n, int M, int K, int N) {
  int i, j, l;
  int align_K = BLOCK_K*(K/BLOCK_K);
  float a[BLOCK_M], b[BLOCK_N];
  float c[BLOCK_M*BLOCK_N] = {0.};

  for(l=0; l<K; l++) {
    for(i=0; i<m; i++) a[i] = A[i*K+l];
    for(i=0; i<n; i++) b[i] = B[l*N+i];

    for(i=0; i<m; i++)
    for(j=0; j<n; j++) {
      c[i*BLOCK_N+j] += a[i]*b[j];
    }
  }

  for(i=0; i<m; i++)
  for(j=0; j<n; j++)
    C[i*M+j] += c[i*BLOCK_N+j];
}

void MatMul_block(float *A, float *B, float *C, int M, int K, int N) {
  int I, J, L;
  int align_M = BLOCK_M*(M/BLOCK_M);
  int align_N = BLOCK_N*(N/BLOCK_N);

  for(I=0; I<align_M; I+=BLOCK_M)
  for(L=0; L<align_N; L+=BLOCK_N)
    Block_C(&A[I*K], &B[L], &C[I*N+L], BLOCK_M, BLOCK_N, M, K, N);

  for(L=0; L<align_N; L+=BLOCK_N)
    Block_C(&A[align_M*K], &B[L], &C[align_M*N+L], M-align_M, BLOCK_N, M, K, N);
  for(I=0; I<align_M; I+=BLOCK_M)
    Block_C(&A[I*K], &B[align_N], &C[I*N+align_N], BLOCK_M, N-align_N, M, K, N);
  Block_C(&A[align_M*K], &B[align_N], &C[align_M*N+align_N], M-align_M, N-align_N, M, K, N);
}

void MatMul_block_ins(float *A, float *B, float *C, int M, int K, int N) {
  int I, J, L;
  int align_M = BLOCK_M*(M/BLOCK_M);
  int align_N = BLOCK_N*(N/BLOCK_N);

  for(I=0; I<align_M; I+=BLOCK_M)
  for(L=0; L<align_N; L+=BLOCK_N)
    Block_C_SIMD(&A[I*K], &B[L], &C[I*N+L], BLOCK_M, BLOCK_N, M, K, N);

  for(L=0; L<align_N; L+=BLOCK_N)
    Block_C(&A[align_M*K], &B[L], &C[align_M*N+L], M-align_M, BLOCK_N, M, K, N);
  for(I=0; I<align_M; I+=BLOCK_M)
    Block_C(&A[I*K], &B[align_N], &C[I*N+align_N], BLOCK_M, N-align_N, M, K, N);
  Block_C(&A[align_M*K], &B[align_N], &C[align_M*N+align_N], M-align_M, N-align_N, M, K, N);
}

void MatMul_ins(float *A, float *B, float *C, int M, int K, int N) {
  int I, J, L;
  int align_K = SIMD_SIZE*(K/SIMD_SIZE);
  int align_N = SIMD_SIZE*(N/SIMD_SIZE);
  aligned_MM(A, B, C, M, align_K, align_N);
}

int main() {
  int i, j, l;
  float *A, *B, *C;

  struct timeval begin, end;
  int timeuse;

  A = (float*)_mm_malloc(sizeof(float)*MM*KK, 64);
  B = (float*)_mm_malloc(sizeof(float)*KK*NN, 64);
  C = (float*)_mm_malloc(sizeof(float)*MM*NN, 64);

  for (i=0; i<MM*KK; i++) A[i] = 1.;
  for (i=0; i<KK*NN; i++) B[i] = 2.;
  for (i=0; i<MM*NN; i++) C[i] = 0.;

  gettimeofday( &begin, NULL );
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, MM, NN, KK, 1., A, KK, B, NN, 1., C, NN);
  gettimeofday( &end, NULL );
  timeuse = 1000000 * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec;
  printf("mkl time: %d us\n", timeuse);
/*
  gettimeofday( &begin, NULL );
  MatMul(A, B, C, MM, KK, NN);
  gettimeofday( &end, NULL );
  timeuse = 1000000 * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec;
  printf("org time: %d us\n", timeuse);
*/
  gettimeofday( &begin, NULL );
  MatMul_block(A, B, C, MM, KK, NN);
  gettimeofday( &end, NULL );
  timeuse = 1000000 * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec;
  printf("opt time: %d us\n", timeuse);
  gettimeofday( &begin, NULL );
  MatMul_block_ins(A, B, C, MM, KK, NN);
  gettimeofday( &end, NULL );
  timeuse = 1000000 * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec;
  printf("opt time: %d us\n", timeuse);
/*
  gettimeofday( &begin, NULL );
  MatMul_ins(A, B, C, MM, KK, NN);
  gettimeofday( &end, NULL );
  timeuse = 1000000 * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec;
  printf("ins time: %d us\n", timeuse);
*/
  _mm_free(A);
  _mm_free(B);
  _mm_free(C);

  return 0;
}
