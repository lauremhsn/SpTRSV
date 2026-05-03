#include "common.h"
#include <cooperative_groups.h>
#include <vector>
#include <algorithm>

namespace cg = cooperative_groups;

__global__ void kernel3_levelset(
    unsigned int n, unsigned int k,
    const unsigned int* __restrict__ rowPtrs,
    const unsigned int* __restrict__ colIdxs,
    const float*        __restrict__ vals,
    const float*        __restrict__ diagArr,
    const float*        __restrict__ B,
    float*                            X,
    const unsigned int* __restrict__ levelRows,
    const unsigned int* __restrict__ levelOffsets,
    unsigned int numLevels)
{
    cg::grid_group grid = cg::this_grid();
    unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x  * blockDim.x;
    for (unsigned int lev = 0; lev < numLevels; ++lev) {
        unsigned int lvStart = levelOffsets[lev];
        unsigned int lvSize  = levelOffsets[lev + 1] - lvStart;
        unsigned int total   = lvSize * k;
        for (unsigned int w = tid; w < total; w += stride) {
            unsigned int rIdx = w / k;
            unsigned int b    = w % k;
            unsigned int row  = levelRows[lvStart + rIdx];
            float sum = B[row * k + b];
            unsigned int rs = rowPtrs[row], re = rowPtrs[row + 1];
            #pragma unroll 4
            for (unsigned int p = rs; p < re; ++p) {
                unsigned int j = __ldg(&colIdxs[p]);
                if (j < row) sum -= __ldg(&vals[p]) * X[j * k + b];
            }
            X[row * k + b] = sum / __ldg(&diagArr[row]);
        }
        grid.sync();
    }
}


__global__ void kernel3_large(
    unsigned int n, unsigned int k,
    const unsigned int* __restrict__ sRowIdx,
    const float*        __restrict__ sDiag,
    const unsigned int* __restrict__ sOffsets,
    const unsigned int* __restrict__ sColIdx,
    const float*        __restrict__ sVal,
    const float*        __restrict__ B,
    float*                            X)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int idx = 0; idx < n; ++idx) {
        unsigned int i  = __ldg(&sRowIdx[idx]);
        float diag      = __ldg(&sDiag[idx]);
        unsigned int rs = __ldg(&sOffsets[idx]);
        unsigned int re = __ldg(&sOffsets[idx + 1]);

        float sum = B[i * k + b];

        #pragma unroll 4
        for (unsigned int p = rs; p < re; ++p) {
            unsigned int j = __ldg(&sColIdx[p]);
            float val      = __ldg(&sVal[p]);
            sum -= val * X[j * k + b];
        }

        X[i * k + b] = sum / diag;
        __syncthreads();
    }
}

void sptrsv_gpu3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X, CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;
    unsigned int k = numCols;

    std::vector<unsigned int> levels(n, 0);
    std::vector<float>        diag(n, 1.0f);
    unsigned int maxLevel = 0;
    for (unsigned int r = 0; r < n; ++r) {
        unsigned int lev = 0;
        for (unsigned int p = L_r_host->rowPtrs[r]; p < L_r_host->rowPtrs[r+1]; ++p) {
            unsigned int c = L_r_host->colIdxs[p];
            if (c < r) { if (levels[c]+1u > lev) lev = levels[c]+1u; }
            else if (c == r) { float v = L_r_host->values[p]; diag[r] = v!=0.f?v:1.f; }
        }
        levels[r] = lev;
        if (lev > maxLevel) maxLevel = lev;
    }
    unsigned int numLevels = maxLevel + 1;
    double avgRowsPerLevel = (double)n / numLevels;

    std::vector<unsigned int> levelOffsets(numLevels+1, 0);
    for (unsigned int r = 0; r < n; ++r) levelOffsets[levels[r]+1]++;
    for (unsigned int l = 1; l <= numLevels; ++l) levelOffsets[l] += levelOffsets[l-1];
    std::vector<unsigned int> levelRows(n);
    { std::vector<unsigned int> cur = levelOffsets;
    for (unsigned int r = 0; r < n; ++r) levelRows[cur[levels[r]]++] = r; }

    CSRMatrix csrPtr; cudaMemcpy(&csrPtr, L_r, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    DenseMatrix bPtr, xPtr;
    cudaMemcpy(&bPtr, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&xPtr, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    if (avgRowsPerLevel >= 50.0) {
        CSCMatrix cscPtr; cudaMemcpy(&cscPtr, L_c, sizeof(CSCMatrix), cudaMemcpyDeviceToHost);
        unsigned int *levelRows_d, *levelOffsets_d; float *diag_d;
        cudaMalloc(&levelRows_d,    n*sizeof(unsigned int));
        cudaMalloc(&levelOffsets_d, (numLevels+1)*sizeof(unsigned int));
        cudaMalloc(&diag_d,         n*sizeof(float));
        cudaMemcpy(levelRows_d,    levelRows.data(),    n*sizeof(unsigned int),            cudaMemcpyHostToDevice);
        cudaMemcpy(levelOffsets_d, levelOffsets.data(), (numLevels+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(diag_d,         diag.data(),         n*sizeof(float),                    cudaMemcpyHostToDevice);

        int threadsPerBlock = 128, blocksPerSM;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel3_levelset, threadsPerBlock, 0);
        if (blocksPerSM < 1) blocksPerSM = 1;
        unsigned int maxLevelSize = 0;
        for (unsigned int l = 0; l < numLevels; ++l) {
            unsigned int sz = levelOffsets[l+1] - levelOffsets[l];
            if (sz > maxLevelSize) maxLevelSize = sz;
        }
        int maxNeeded = ((int)(maxLevelSize*k) + threadsPerBlock - 1) / threadsPerBlock;
        int gridSize  = std::min(numSMs*blocksPerSM, std::max(numSMs, maxNeeded));

        void* args[] = {&n,&k,&csrPtr.rowPtrs,&csrPtr.colIdxs,&csrPtr.values,
                        &diag_d,&bPtr.values,&xPtr.values,
                        &levelRows_d,&levelOffsets_d,&numLevels};
        cudaLaunchCooperativeKernel((void*)kernel3_levelset, gridSize, threadsPerBlock, args);
        cudaFree(levelRows_d); cudaFree(levelOffsets_d); cudaFree(diag_d);

    } else {
        std::vector<unsigned int> sRowIdx_h  = levelRows;
        std::vector<float>        sDiag_h(n);
        std::vector<unsigned int> sOffsets_h(n+1, 0);
        std::vector<unsigned int> sColIdx_h;
        std::vector<float>        sVal_h;
        sColIdx_h.reserve(n * 4);
        sVal_h.reserve(n * 4);

        for (unsigned int idx = 0; idx < n; ++idx) {
            unsigned int i  = levelRows[idx];
            sDiag_h[idx]    = diag[i];
            sOffsets_h[idx] = (unsigned int)sColIdx_h.size();
            for (unsigned int p = L_r_host->rowPtrs[i]; p < L_r_host->rowPtrs[i+1]; ++p) {
                unsigned int c = L_r_host->colIdxs[p];
                if (c < i) {
                    sColIdx_h.push_back(c);
                    sVal_h.push_back(L_r_host->values[p]);
                }
            }
        }
        sOffsets_h[n] = (unsigned int)sColIdx_h.size();
        unsigned int totalLower = (unsigned int)sColIdx_h.size();

        unsigned int *sRowIdx_d, *sOffsets_d, *sColIdx_d;
        float        *sDiag_d, *sVal_d;
        cudaMalloc(&sRowIdx_d,  n*sizeof(unsigned int));
        cudaMalloc(&sDiag_d,    n*sizeof(float));
        cudaMalloc(&sOffsets_d, (n+1)*sizeof(unsigned int));
        cudaMalloc(&sColIdx_d,  totalLower*sizeof(unsigned int));
        cudaMalloc(&sVal_d,     totalLower*sizeof(float));

        cudaMemcpy(sRowIdx_d,  sRowIdx_h.data(),  n*sizeof(unsigned int),          cudaMemcpyHostToDevice);
        cudaMemcpy(sDiag_d,    sDiag_h.data(),    n*sizeof(float),                  cudaMemcpyHostToDevice);
        cudaMemcpy(sOffsets_d, sOffsets_h.data(), (n+1)*sizeof(unsigned int),       cudaMemcpyHostToDevice);
        cudaMemcpy(sColIdx_d,  sColIdx_h.data(),  totalLower*sizeof(unsigned int),  cudaMemcpyHostToDevice);
        cudaMemcpy(sVal_d,     sVal_h.data(),     totalLower*sizeof(float),          cudaMemcpyHostToDevice);


        int colWidth  = 64;
        int numBlocks = std::max(1, (int)k / colWidth);

        kernel3_large<<<numBlocks, colWidth>>>(
            n, k,
            sRowIdx_d, sDiag_d, sOffsets_d, sColIdx_d, sVal_d,
            bPtr.values, xPtr.values);

        cudaFree(sRowIdx_d); cudaFree(sDiag_d); cudaFree(sOffsets_d);
        cudaFree(sColIdx_d); cudaFree(sVal_d);
    }
}