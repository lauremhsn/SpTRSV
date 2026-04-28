#include "common.h"
#include <cuda/atomic>

__global__ void kernel2(unsigned int n, unsigned int k, const unsigned int* __restrict__ rowPtrs, const unsigned int* __restrict__ colIdxs, const float* __restrict__ vals, const unsigned int* __restrict__ cscColPtrs, const unsigned int* __restrict__ cscRowIdxs, const unsigned int* __restrict__ cscDownStart, const float* __restrict__ diagArr, const float* __restrict__ B, float* X, int* inDegree, int* nextRow){
    unsigned int b = threadIdx.x;

    __shared__ int sharedRow;

    while (true){
        if (b == 0){
            cuda::atomic_ref<int, cuda::thread_scope_device> nxt(*nextRow);
            sharedRow = nxt.fetch_add(1, cuda::memory_order_relaxed);
        }
        __syncthreads();

        int i = sharedRow;
        if (i>=(int)n){
            return;
        }
        if (b == 0){
            cuda::atomic_ref<int, cuda::thread_scope_device> deg(inDegree[i]);
            while (deg.load(cuda::memory_order_acquire) != 0)
            #if __CUDA_ARCH__ >= 700
                __nanosleep(8);
            #else
                ;
            #endif
        }
        __syncthreads();

        if (b<k){
            float sum = B[i*k + b];
            float d = __ldg(&diagArr[i]);

            for (unsigned int p = rowPtrs[i]; p<rowPtrs[i + 1]; ++p) {
                unsigned int j = __ldg(&colIdxs[p]);
                float val = __ldg(&vals[p]);
                if (j<i) sum -= val*X[j*k + b];
            }
            X[i*k + b] = sum/d;
        }
        __syncthreads();

        if (b == 0) __threadfence();
        __syncthreads();

        unsigned int pStart = cscDownStart[i];
        unsigned int pEnd = cscColPtrs[i + 1];

        for (unsigned int p = pStart + b; p<pEnd; p += blockDim.x) {
            unsigned int r = cscRowIdxs[p];
            cuda::atomic_ref<int, cuda::thread_scope_device> deg(inDegree[r]);
            deg.fetch_sub(1, cuda::memory_order_release);
        }
        __syncthreads();
    }
}

void sptrsv_gpu2(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X, CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols){

    unsigned int n = L_r_host->numRows;
    unsigned int k = numCols;

    //extract device pointers
    CSRMatrix csrPtr;
    cudaMemcpy(&csrPtr, L_r, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    CSCMatrix cscPtr;
    cudaMemcpy(&cscPtr, L_c, sizeof(CSCMatrix), cudaMemcpyDeviceToHost);
    DenseMatrix bPtr, xPtr;
    cudaMemcpy(&bPtr, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&xPtr, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    int* inDegree_h = (int*)calloc(n, sizeof(int));
    float* diag_h = (float*)malloc(n*sizeof(float));
    unsigned int* cscDownStart_h = (unsigned int*)malloc(n*sizeof(unsigned int));

    for (unsigned int r = 0; r < n; r++){
        diag_h[r] = 1.0f;
        for (unsigned int p = L_r_host->rowPtrs[r]; p<L_r_host->rowPtrs[r+1]; p++){
            unsigned int c = L_r_host->colIdxs[p];
            if (c<r) inDegree_h[r]++;
            else if (c == r) {
                float v = L_r_host->values[p];
                diag_h[r] = (v != 0.0f) ? v : 1.0f;
            }
        }
    }

    for (unsigned int c = 0; c < n; c++) {
        unsigned int ds = L_c_host->colPtrs[c + 1];
        for (unsigned int p = L_c_host->colPtrs[c]; p < L_c_host->colPtrs[c+1]; p++)
            if (L_c_host->rowIdxs[p] > c) { ds = p; break; }
        cscDownStart_h[c] = ds;
    }

    int* inDegree_d;
    int* nextRow_d;
    float* diag_d;
    unsigned int* cscDownStart_d;

    cudaMalloc((void**)&inDegree_d, n*sizeof(int));
    cudaMalloc((void**)&nextRow_d, sizeof(int));
    cudaMalloc((void**)&diag_d, n*sizeof(float));
    cudaMalloc((void**)&cscDownStart_d, n*sizeof(unsigned int));

    cudaMemcpy(inDegree_d, inDegree_h, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(diag_d, diag_h, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cscDownStart_d, cscDownStart_h, n*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(nextRow_d, 0, sizeof(int));

    free(inDegree_h);
    free(diag_h);
    free(cscDownStart_h);

    int numSMs = 0;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    int blocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel2, (int)k, 0);
    if (blocksPerSM < 1) blocksPerSM = 1;

    dim3 block(k);
    dim3 grid(numSMs * blocksPerSM);

    kernel2<<<grid, block>>>(n, k, threads, csrPtr.rowPtrs, csrPtr.colIdxs, csrPtr.values, cscPtr.colPtrs, cscPtr.rowIdxs, cscDownStart_d, diag_d, bPtr.values, xPtr.values, inDegree_d, nextRow_d);

    cudaFree(inDegree_d);
    cudaFree(nextRow_d);
    cudaFree(diag_d);
    cudaFree(cscDownStart_d);
}
