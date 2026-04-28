#include "common.h"
#include <cuda/atomic>

#define TILE 4

__global__ void kernel2(unsigned int  n, unsigned int  k, unsigned int  threads, unsigned int* rowPtrs, unsigned int* colIdxs, float* vals, float* B, float* X, int* ready, int* nextRow){
    unsigned int tid = threadIdx.x;
    unsigned int bBase = tid*TILE;

    __shared__ int sharedRow;

    while (true){
        if (tid == 0){
            cuda::atomic_ref<int, cuda::thread_scope_device> nxt(*nextRow);
            sharedRow = nxt.fetch_add(1, cuda::memory_order_relaxed);
        }
        __syncthreads();

        int i = sharedRow;
        if (i>=(int)n){
            return;
        }

        if (bBase<k){
            float sum[TILE];
            #pragma unroll
            for (int t = 0; t<TILE; ++t)
                sum[t] = ((bBase + t)<k) ? B[i*k + bBase + t] : 0.0f;

            float diag = 1.0f;

            for (unsigned int p = rowPtrs[i]; p < rowPtrs[i + 1]; p++){
                unsigned int j = colIdxs[p];
                float val = vals[p];

                if (j<i){
                    //spin until dependency j is done
                    cuda::atomic_ref<int, cuda::thread_scope_device> rref(ready[j]);
                    while (rref.load(cuda::memory_order_acquire) == 0)
                    #if __CUDA_ARCH__ >= 700 //nanosleep was not introduced for some older GPUs, hence the checking of compute compatibility 700.
                        __nanosleep(32);
                    #else
                        ;
                    #endif
                    //load TILE values from X[j], reuse val across all TILE
                    #pragma unroll
                    for (int t = 0; t<TILE; ++t){
                        if (bBase + t<k){
                            sum[t] -= val*X[j*k + bBase + t];
                        }
                    }
                } 
                else if (j == i){
                    diag = (val != 0.0f) ? val : 1.0f;
                }
            }

            //write TILE results
            #pragma unroll
            for (int t = 0; t<TILE; t++){
                if (bBase + t<k){
                    X[i*k + bBase + t] = sum[t]/diag;
                }
            }
        }

        __syncthreads();

        if (tid == 0){
            __threadfence();
            cuda::atomic_ref<int, cuda::thread_scope_device> rdy(ready[i]);
            rdy.store(1, cuda::memory_order_release);
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
    DenseMatrix bPtr, xPtr;
    cudaMemcpy(&bPtr, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&xPtr, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    int* ready_d;
    int* nextRow_d;
    cudaMalloc((void**)&ready_d, n*sizeof(int));
    cudaMalloc((void**)&nextRow_d, sizeof(int));
    cudaMemset(ready_d, 0, n*sizeof(int));
    cudaMemset(nextRow_d, 0, sizeof(int));

    //k/TILE threads per block (each handles TILE columns)
    unsigned int threads = k/TILE;

    int numSMs, maxThreadsPerSM;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

    int blocksPerSM = maxThreadsPerSM/(int)threads;
    if (blocksPerSM<1){
        blocksPerSM = 1;
    }
    int numBlocks = numSMs*blocksPerSM;

    dim3 block(threads);
    dim3 grid(numBlocks);

    kernel2<<<grid, block>>>(n, k, threads, csrPtr.rowPtrs, csrPtr.colIdxs, csrPtr.values, bPtr.values, xPtr.values, ready_d, nextRow_d);

    cudaFree(ready_d);
    cudaFree(nextRow_d);
}
