#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>

#include <cmath>
#include "CudaTimer.h"


cublasHandle_t handle;


void GPU_mul(double *A, double *B, double *C,
    int lda, int ldb, int ldc,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    double alpha, double beta) {
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, XB, YA, XA, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

void GPU_add(double *A, double *B, double *C,
    int lda, int ldb, int ldc,
    int XA, int YA,
    double alpha, double beta) {
  cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, XA, YA, &alpha, A, lda, &beta, B, ldb, C, ldc);
}

void verifyByCUBLAS(double *d_A, double *d_B, double *d_C, int M, int N, int K) {
  double one = 1.0;
  double zero = 0.0;
#if CMAJOR
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, d_A, M, d_B, K, &zero, d_C, M);
#else
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, d_B, N, d_A, K, &zero, d_C, N);
#endif
}

/*
  lda, ldb, ldc is the width in actual memory.
  XA, XB, XC is the width for computation.
  A = XA x YA
  B = XB x YB
  C = XC x YC
*/
void strassen(double *A, double *B, double *C,
    int lda, int ldb, int ldc,
    int XA, int XB, int XC,
    int YA, int YB, int YC,
    int depth) {

  int XA2 = XA / 2;
  int XB2 = XB / 2;
  int XC2 = XC / 2;
  
  int YA2 = YA / 2;
  int YB2 = YB / 2;
  int YC2 = YC / 2;

  double *W_1, *W_2;
  int lw1 = (XA2 > XC2 ? XA2 : XC2);
  int lw2 = XB2;
  cudaMalloc((void **)&W_1, lw1 * YA2 * sizeof(double));
  cudaMalloc((void **)&W_2, lw2 * YB2 * sizeof(double));

  int dXA = XA2;
  int dYA = YA2 * lda;
  int dXB = XB2;
  
  int dYB = YB2 * ldb;
  int dXC = XC2;
  int dYC = YC2 * ldc;

  double *A11, *A12, *A21, *A22;
  double *B11, *B12, *B21, *B22;
  double *C11, *C12, *C21, *C22;
  
  A11 = A;
  A12 = A + dXA;
  A21 = A + dYA;
  A22 = A + dXA + dYA;
  
  B11 = B;
  B12 = B + dXB;
  B21 = B + dYB;
  B22 = B + dXB + dYB;
  
  C11 = C;
  C12 = C + dXC;
  C21 = C + dYC;
  C22 = C + dXC + dYC;

  /* cutoff criteria */
  bool stop = false;
  
#if 0
  int cutoff = 2048;
  float mm = cutoff / XB2;
  float nn = cutoff / YA2;
  float kk = cutoff / XA2;
  if ((mm + nn + kk) >= 3) {
      stop = true;
  }
#endif

  if (depth <= 1 || stop) {
    GPU_add(A11, A21, W_1, lda, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A11 - A21
    GPU_add(B22, B12, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - B12
    GPU_mul(W_1, W_2, C21, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C21 = W_1 * W_2
    GPU_add(A21, A22, W_1, lda, lda, lw1, XA2, YA2, 1.0,  1.0); // W_1 = A21 + A22
    GPU_add(B12, B11, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B12 - B11
    GPU_mul(W_1, W_2, C22, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C22 = W_1 * W_2
    GPU_add(W_1, A11, W_1, lw1, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = W_1- A11
    GPU_add(B22, W_2, W_2, ldb, lw2, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - W_2
    GPU_mul(W_1, W_2, C11, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C11 = W_1 * W_2
    GPU_add(A12, W_1, W_1, lda, lw1, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A12 - W_1
    GPU_mul(W_1, B22, C12, lw1, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C12 = W_1 * B22
    GPU_add(C22, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C22 + C12
    GPU_mul(A11, B11, W_1, lda, ldb, lw1, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // W_1= A11 * B11
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1 + C11
    GPU_add(C11, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C11 + C12
    GPU_add(C11, C21, C11, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = C11 + C21
    GPU_add(W_2, B21, W_2, lw2, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = W_2- B21
    GPU_mul(A22, W_2, C21, lda, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C21 = A22 * W_2
    GPU_add(C11, C21, C21, ldc, ldc, ldc, XC2, YC2, 1.0, -1.0); // C11 = C11 - C21
    GPU_add(C11, C22, C22, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C22 = C11 + C22
    GPU_mul(A12, B21, C11, lda, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C11 = A12 * B21
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1+ C11
  }
  else {
    GPU_add(A11, A21, W_1, lda, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A11 - A21
    GPU_add(B22, B12, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - B12
    strassen(W_1, W_2, C21, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(A21, A22, W_1, lda, lda, lw1, XA2, YA2, 1.0,  1.0); // W_1 = A21 + A22
    GPU_add(B12, B11, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B12 - B11
    strassen(W_1, W_2, C22, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(W_1, A11, W_1, lw1, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = W_1- A11
    GPU_add(B22, W_2, W_2, ldb, lw2, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - W_2
    strassen(W_1, W_2, C11, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(A12, W_1, W_1, lda, lw1, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A12 - W_1
    strassen(W_1, B22, C12, lw1, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(C22, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C22 + C12
    strassen(A11, B11, W_1, lda, ldb, lw1, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1 + C11
    GPU_add(C11, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C12 = C11 + C12
    GPU_add(C11, C21, C11, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = C11 + C21
    GPU_add(W_2, B21, W_2, lw2, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = W_2- B21
    strassen(A22, W_2, C21, lda, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(C11, C21, C21, ldc, ldc, ldc, XC2, YC2, 1.0, -1.0); // C11 = C11 - C21
    GPU_add(C11, C22, C22, ldc, ldc, ldc, XC2, YC2, 1.0,  1.0); // C22 = C11 + C22
    strassen(A12, B21, C11, lda, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, depth - 1);
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0,  1.0); // C11 = W_1+ C11
  }
  cudaFree(W_1);
  cudaFree(W_2);

  /* dynamic peeling fix-up */
  int pxa = XA % 2;
  int pya = YA % 2;
  int pxb = XB % 2;
  int pyb = YB % 2;
  int pxc = XC % 2;
  int pyc = YC % 2;
  
  int nxa = XA - pxa;
  int nya = YA - pya;
  int nxb = XB - pxb;
  int nyb = YB - pyb;
  int nxc = XC - pxc;
  int nyc = YC - pyc;

  double *a12, *a21;
  double *b12, *b21;
  double *c12, *c21;
  int dxa = nxa;
  int dya = nya * lda;
  int dxb = nxb;
  int dyb = nyb * ldb;
  int dxc = nxc;
  int dyc = nyc * ldc;
  
  a12 = A + dxa;
  a21 = A + dya;
  // a22 = A + dxa + dya;
  b12 = B + dxb;
  b21 = B + dyb;
  // b22 = B + dxb + dyb;
  c12 = C + dxc;
  c21 = C + dyc;
  // c22 = C + dxc + dyc;

  /* 
    A11 = nxa x nya
    a12 = pxa x nya
    a21 = nxa x pya
    a22 = pxa x pya
   */
  GPU_mul(a21, B11, c21, lda, ldb, ldc, nxa,  XB,  XC, pya, nyb, pyc, 1.0, 0.0);
  GPU_mul(A11, b12, c12, lda, ldb, ldc, nxa, pxb, pxc,  YA, nyb,  YC, 1.0, 0.0);
  GPU_mul(a12, b21, C11, lda, ldb, ldc, pxa,  XB,  XC,  YA, pyb,  YC, 1.0, 1.0);
}


int main (int argc, char **argv) {

  if(argc != 7) {
    printf("Usage: strassen <M> <N> <K> <iter> <check> <depth>\n");
    return -1;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int iter = atoi(argv[4]);
  int check = atoi(argv[5]);
  int depth = atoi(argv[6]);

  int sizeA = M * K;
  int sizeB = K * N;
  int sizeC = M * N;
  int memSizeA = sizeA * sizeof(double);
  int memSizeB = sizeB * sizeof(double);
  int memSizeC = sizeC * sizeof(double);

  double *h_A = (double *)malloc(memSizeA);
  double *h_B = (double *)malloc(memSizeB);
  double *h_C = (double *)malloc(memSizeC);
  double *v_C = (double *)malloc(memSizeC);
  
  for (int i = 0; i < sizeA; ++i) {
    h_A[i] = i % 3;
  }
  for (int i = 0; i < sizeB; ++i) {
    h_B[i] = i % 3;
  }
  for (int i = 0; i < sizeC; ++i) {
    h_C[i] = 0.0f;
    v_C[i] = 0.0f;
  }

  double *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, memSizeA);
  cudaMalloc((void**)&d_B, memSizeB);
  cudaMalloc((void**)&d_C, memSizeC);
  cudaMemcpy(d_A, h_A, memSizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, memSizeB, cudaMemcpyHostToDevice);

  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    fflush(NULL);
    return EXIT_FAILURE;
  }

  CudaTimer ct;
  ct.start();
  for (int i = 0; i < iter; ++i) {
    strassen(d_A, d_B, d_C, K, N, N, K, N, N, M, K, M, depth);
  }
  ct.stop();

  double strassenTime = ct.value() / iter;
  cudaMemcpy(h_C, d_C, memSizeC, cudaMemcpyDeviceToHost);

#if 1
  ct.start();
  for (int i = 0; i < iter; ++i) {
    verifyByCUBLAS (d_A, d_B, d_C, M, N, K);
  }
  ct.stop();
  
  double classicTime = ct.value() / iter;
  cudaMemcpy (v_C, d_C, memSizeC, cudaMemcpyDeviceToHost);

  double speedup = classicTime / strassenTime;
  printf ("%d %d %d %.2f %.2f %.2f\n", M, N, K, strassenTime, classicTime, speedup);
#endif

  if (check) {
    double absErr = 0.0;
    for(int i = 0; i < sizeC; ++i) {
      absErr += abs(h_C[i] - v_C[i]);
    }
    if (absErr > 1) {
      printf("CHECK: Absolute error: %lf\n", absErr);
    }
  }

  free(h_A);
  free(h_B);
  free(h_C);
  free(v_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS shutdown error\n");
    fflush(NULL);
    return EXIT_FAILURE;
  }
}
