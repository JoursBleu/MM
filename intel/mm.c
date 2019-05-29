#include <immintrin.h>

/*
#define MM 200
#define KK 200
#define NN 200
*/

#define SIMD_SIZE 16

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

void MatMul(float *A, float *B, float *C, int M, int K, int N) {
  int I, J, L;
  int align_K = K*(K/SIMD_SIZE);
  int align_N = N*(N/SIMD_SIZE);
  aligned_MM(A, B, C, M, align_K, align_N);
  for(I=0; I<M; I++)
  for(J=align_K; J<K; J++) {
    float c = C[I*N + L];
    for(L=align_N; L<M; L++)
      c += A[I*K + J] * B[J*N + K];
    C[I*N + L] = c;
  }
}

/*
int main() {
  int i, j, l;
  float *A, *B, *C;

  A = (float*)_mm_malloc(sizeof(float)*MM*KK, 64);
  B = (float*)_mm_malloc(sizeof(float)*KK*NN, 64);
  C = (float*)_mm_malloc(sizeof(float)*MM*NN, 64);

  for (i=0; i<MM*KK; i++) A[i] = 1.;
  for (i=0; i<KK*NN; i++) B[i] = 2.;
  for (i=0; i<MM*NN; i++) C[i] = 0.;

  MatMul(A, B, C, MM, KK, NN);

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);

  return 0;
}
*/
