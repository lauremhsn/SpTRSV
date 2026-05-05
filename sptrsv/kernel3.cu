#include "common.h"
#include <cooperative_groups.h>
#include <vector>
#include <algorithm>
#include <cstring>

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
            unsigned int row  = __ldg(&levelRows[lvStart + rIdx]);

            float sum = __ldg(&B[(long)row * k + b]);
            unsigned int rs = rowPtrs[row], re = rowPtrs[row + 1];

            for (unsigned int p = rs; p < re; ++p) {
                unsigned int j = __ldg(&colIdxs[p]);
                if (j < row) {
                    float val = __ldg(&vals[p]);
                    float xj  = __ldg(&X[(long)j * k + b]);
                    float prod = __fmul_rn(val, xj);
                    sum = __fsub_rn(sum, prod);
                }
            }
            X[(long)row * k + b] = sum / __ldg(&diagArr[row]);
        }
        grid.sync();
    }
}

__global__ void init_X_from_B(unsigned int n, unsigned int k, const float* __restrict__ B, float* X)
{
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * k) return;
    X[t] = B[t];
}

__global__ void kernel3_chain(
    unsigned int n, unsigned int k,
    const float*        __restrict__ sRcpDiag,
    const unsigned int* __restrict__ sOffsets,
    const unsigned int* __restrict__ sColTimesK,
    const float*        __restrict__ sVal,
    float*                            X)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= k) return;

    float        rcp_curr    = __ldg(&sRcpDiag[0]);
    unsigned int rs_curr     = __ldg(&sOffsets[0]);
    unsigned int re_curr     = __ldg(&sOffsets[1]);
    unsigned int rowOff_curr = 0;

    for (unsigned int idx = 0; idx < n; ++idx) {
        float        rcp_next    = 0;
        unsigned int rs_next     = 0;
        unsigned int re_next     = 0;
        if (idx + 1 < n) {
            rcp_next    = __ldg(&sRcpDiag[idx + 1]);
            rs_next     = __ldg(&sOffsets[idx + 1]);
            re_next     = __ldg(&sOffsets[idx + 2]);
        }

        float sum = X[rowOff_curr + b];

        unsigned int p = rs_curr;
        unsigned int re_padded = (re_curr + 1u) & ~1u;

        for (; p < re_padded; p += 2) {
            uint2  cols = *reinterpret_cast<const uint2*>(&sColTimesK[p]);
            float2 vals = *reinterpret_cast<const float2*>(&sVal[p]);
            sum -= vals.x * X[cols.x + b];
            sum -= vals.y * X[cols.y + b];
        }

        X[rowOff_curr + b] = sum * rcp_curr;
        __threadfence_block();

        rowOff_curr += k;
        rcp_curr    = rcp_next;
        rs_curr     = rs_next;
        re_curr     = re_next;
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
        unsigned int *levelRows_d, *levelOffsets_d; float *diag_d;
        cudaMalloc(&levelRows_d,    n*sizeof(unsigned int));
        cudaMalloc(&levelOffsets_d, (numLevels+1)*sizeof(unsigned int));
        cudaMalloc(&diag_d,         n*sizeof(float));
        cudaMemcpy(levelRows_d,    levelRows.data(),    n*sizeof(unsigned int),             cudaMemcpyHostToDevice);
        cudaMemcpy(levelOffsets_d, levelOffsets.data(), (numLevels+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(diag_d,         diag.data(),         n*sizeof(float),                    cudaMemcpyHostToDevice);

        int threadsPerBlock = 256, blocksPerSM;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel3_levelset, threadsPerBlock, 0);
        if (blocksPerSM < 1) blocksPerSM = 1;
        unsigned int maxLevelSize = 0;
        for (unsigned int l = 0; l < numLevels; ++l) {
            unsigned int sz = levelOffsets[l+1] - levelOffsets[l];
            if (sz > maxLevelSize) maxLevelSize = sz;
        }
        int maxNeeded = ((long)maxLevelSize*k + threadsPerBlock - 1) / threadsPerBlock;
        int gridSize  = std::min(numSMs*blocksPerSM, std::max(numSMs, maxNeeded));

        void* args[] = {&n,&k,&csrPtr.rowPtrs,&csrPtr.colIdxs,&csrPtr.values,
                        &diag_d,&bPtr.values,&xPtr.values,
                        &levelRows_d,&levelOffsets_d,&numLevels};
        cudaLaunchCooperativeKernel((void*)kernel3_levelset, gridSize, threadsPerBlock, args);
        cudaFree(levelRows_d); cudaFree(levelOffsets_d); cudaFree(diag_d);

    } else {
        std::vector<float>        sRcpDiag_h(n);
        std::vector<unsigned int> sOffsets_h(n+1, 0);
        std::vector<unsigned int> sColTimesK_h;
        std::vector<float>        sVal_h;
        sColTimesK_h.reserve(n * 5 + 16);
        sVal_h.reserve(n * 5 + 16);

        for (unsigned int idx = 0; idx < n; ++idx) {
            unsigned int i = idx;
            sRcpDiag_h[idx] = 1.0f / diag[i];

            if (sColTimesK_h.size() & 1u) {
                sColTimesK_h.push_back(0);
                sVal_h.push_back(0.0f);
            }
            sOffsets_h[idx] = (unsigned int)sColTimesK_h.size();

            for (unsigned int p = L_r_host->rowPtrs[i]; p < L_r_host->rowPtrs[i+1]; ++p) {
                unsigned int c = L_r_host->colIdxs[p];
                if (c < i) {
                    sColTimesK_h.push_back(c * k);
                    sVal_h.push_back(L_r_host->values[p]);
                }
            }
        }
        sOffsets_h[n] = (unsigned int)sColTimesK_h.size();
        sColTimesK_h.push_back(0);
        sColTimesK_h.push_back(0);
        sVal_h.push_back(0.0f);
        sVal_h.push_back(0.0f);
        unsigned int totalLower = (unsigned int)sColTimesK_h.size();

        unsigned int *sOffsets_d, *sColTimesK_d;
        float        *sRcpDiag_d, *sVal_d;
        cudaMalloc(&sRcpDiag_d,   n*sizeof(float));
        cudaMalloc(&sOffsets_d,   (n+1)*sizeof(unsigned int));
        cudaMalloc(&sColTimesK_d, totalLower*sizeof(unsigned int));
        cudaMalloc(&sVal_d,       totalLower*sizeof(float));

        cudaMemcpy(sRcpDiag_d,   sRcpDiag_h.data(),   n*sizeof(float),                cudaMemcpyHostToDevice);
        cudaMemcpy(sOffsets_d,   sOffsets_h.data(),   (n+1)*sizeof(unsigned int),     cudaMemcpyHostToDevice);
        cudaMemcpy(sColTimesK_d, sColTimesK_h.data(), totalLower*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(sVal_d,       sVal_h.data(),       totalLower*sizeof(float),        cudaMemcpyHostToDevice);

        int initThreads = 256;
        int initBlocks  = ((long)n*k + initThreads - 1) / initThreads;
        init_X_from_B<<<initBlocks, initThreads>>>(n, k, bPtr.values, xPtr.values);

        int colWidth  = 32;
        int numBlocks = ((int)k + colWidth - 1) / colWidth;

        kernel3_chain<<<numBlocks, colWidth>>>(
            n, k,
            sRcpDiag_d, sOffsets_d, sColTimesK_d, sVal_d,
            xPtr.values);

        cudaFree(sRcpDiag_d); cudaFree(sOffsets_d);
        cudaFree(sColTimesK_d); cudaFree(sVal_d);
    }
}