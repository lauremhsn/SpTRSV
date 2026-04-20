#include "common.h"
#include <cuda/atomic>

__global__ void kernel1(unsigned int n, unsigned int k,
                        unsigned int* cscColPtrs, unsigned int* cscRowIdxs, float* cscVals,
                        unsigned int* csrRowPtrs, unsigned int* csrColIdxs, float* csrVals,
                        float* B, float* X,
                        float* partialSum, int* inDegree, int* nextRow) {

    unsigned int b = threadIdx.x;
    __shared__ int sharedRow;

    while (true) {
        if (b == 0) {
            cuda::atomic_ref<int, cuda::thread_scope_device> nxt(*nextRow);
            sharedRow = nxt.fetch_add(1, cuda::memory_order_relaxed);
        }
        __syncthreads();

        int i = sharedRow;
        if (i >= (int)n) return;

        if (b == 0) {
            cuda::atomic_ref<int, cuda::thread_scope_device> deg(inDegree[i]);
            while (deg.load(cuda::memory_order_acquire) != 0);
        }
        __syncthreads();

        if (b < k) {
            float diag = 1.0f;
            for (unsigned int p = csrRowPtrs[i]; p < csrRowPtrs[i + 1]; ++p) {
                if (csrColIdxs[p] == (unsigned int)i) {
                    diag = (csrVals[p] != 0.0f) ? csrVals[p] : 1.0f;
                    break;
                }
            }

            X[i * k + b] = (B[i * k + b] - partialSum[i * k + b]) / diag;
        }
        __syncthreads();

        if (b < k) {
            for (unsigned int p = cscColPtrs[i]; p < cscColPtrs[i + 1]; ++p) {
                unsigned int r = cscRowIdxs[p];
                if (r > (unsigned int)i) {
                    atomicAdd(&partialSum[r * k + b], cscVals[p] * X[i * k + b]);
                }
            }
        }
        __syncthreads();

        if (b == 0) {
            __threadfence();
            for (unsigned int p = cscColPtrs[i]; p < cscColPtrs[i + 1]; ++p) {
                unsigned int r = cscRowIdxs[p];
                if (r > (unsigned int)i) {
                    cuda::atomic_ref<int, cuda::thread_scope_device> deg(inDegree[r]);
                    deg.fetch_sub(1, cuda::memory_order_release);
                }
            }
        }
        __syncthreads();
    }
}

void sptrsv_gpu1(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                  CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols) {

    unsigned int n = L_r_host->numRows;
    unsigned int k = numCols;

    CSRMatrix csrPtr;
    cudaMemcpy(&csrPtr, L_r, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    CSCMatrix cscPtr;
    cudaMemcpy(&cscPtr, L_c, sizeof(CSCMatrix), cudaMemcpyDeviceToHost);
    DenseMatrix bPtr, xPtr;
    cudaMemcpy(&bPtr, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&xPtr, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    int* inDegree_h = (int*)calloc(n, sizeof(int));
    for (unsigned int r = 0; r < n; ++r) {
        for (unsigned int p = L_r_host->rowPtrs[r]; p < L_r_host->rowPtrs[r + 1]; ++p) {
            if (L_r_host->colIdxs[p] < r) {
                inDegree_h[r]++;
            }
        }
    }

    int* inDegree_d;
    float* partialSum_d;
    int* nextRow_d;
    cudaMalloc((void**)&inDegree_d, n * sizeof(int));
    cudaMalloc((void**)&partialSum_d, (size_t)n * k * sizeof(float));
    cudaMalloc((void**)&nextRow_d, sizeof(int));

    cudaMemcpy(inDegree_d, inDegree_h, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(partialSum_d, 0, (size_t)n * k * sizeof(float));
    cudaMemset(nextRow_d, 0, sizeof(int));

    free(inDegree_h);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    dim3 block(k);
    dim3 grid(numSMs);

    kernel1<<<grid, block>>>(n, k,
        cscPtr.colPtrs, cscPtr.rowIdxs, cscPtr.values,
        csrPtr.rowPtrs, csrPtr.colIdxs, csrPtr.values,
        bPtr.values, xPtr.values,
        partialSum_d, inDegree_d, nextRow_d);

    cudaFree(inDegree_d);
    cudaFree(partialSum_d);
    cudaFree(nextRow_d);
}
