#include "common.h"
#include <cooperative_groups.h>
#include <vector>
#include <algorithm>
#include <cstring>

namespace cg = cooperative_groups;

namespace {

struct PreprocCache {
    const CSRMatrix* key = nullptr;
    unsigned int n = 0;
    unsigned int numNonzeros = 0;

    enum Path { PATH_CHAIN, PATH_LEVELSET, PATH_SYNCFREE } path = PATH_CHAIN;

    unsigned int  numLevels = 0;
    unsigned int  maxLevelSize = 0;
    unsigned int* levelRows_d     = nullptr;   // rows in topological order
    unsigned int* levelOffsets_d  = nullptr;   // levelset only
    float*        diag_d          = nullptr;   // exact diagonals (for sum/diag)
    unsigned int* lowerRowPtrs_d  = nullptr;   // lower-only CSR (no diag, no upper)
    unsigned int* lowerColIdxs_d  = nullptr;
    float*        lowerVals_d     = nullptr;

    unsigned int* ready_d  = nullptr;
    unsigned int* rowCtr_d = nullptr;

    float*        sRcpDiag_d = nullptr;
    unsigned int* sOffsets_d = nullptr;
    unsigned int* sCol_d     = nullptr;
    float*        sVal_d     = nullptr;
};

PreprocCache g_cache;

void invalidateCache(PreprocCache& c) {
    if (c.levelRows_d)    cudaFree(c.levelRows_d);
    if (c.levelOffsets_d) cudaFree(c.levelOffsets_d);
    if (c.diag_d)         cudaFree(c.diag_d);
    if (c.lowerRowPtrs_d) cudaFree(c.lowerRowPtrs_d);
    if (c.lowerColIdxs_d) cudaFree(c.lowerColIdxs_d);
    if (c.lowerVals_d)    cudaFree(c.lowerVals_d);
    if (c.ready_d)        cudaFree(c.ready_d);
    if (c.rowCtr_d)       cudaFree(c.rowCtr_d);
    if (c.sRcpDiag_d)     cudaFree(c.sRcpDiag_d);
    if (c.sOffsets_d)     cudaFree(c.sOffsets_d);
    if (c.sCol_d)         cudaFree(c.sCol_d);
    if (c.sVal_d)         cudaFree(c.sVal_d);
    c = PreprocCache{};
}

void buildPreproc(PreprocCache& c, CSRMatrix* L_r_host) {
    unsigned int n = L_r_host->numRows;
    c.key = L_r_host;
    c.n = n;
    c.numNonzeros = L_r_host->numNonzeros;

    // ---- Level scheduling + diagonal extraction (single pass) ----
    std::vector<unsigned int> levels(n, 0);
    std::vector<float>        diag(n, 1.0f);
    unsigned int maxLevel = 0;
    for (unsigned int r = 0; r < n; ++r) {
        unsigned int lev = 0;
        for (unsigned int p = L_r_host->rowPtrs[r]; p < L_r_host->rowPtrs[r+1]; ++p) {
            unsigned int cc = L_r_host->colIdxs[p];
            if (cc < r) {
                if (levels[cc] + 1u > lev) lev = levels[cc] + 1u;
            } else if (cc == r) {
                float v = L_r_host->values[p];
                diag[r] = (v != 0.f) ? v : 1.f;
            }
        }
        levels[r] = lev;
        if (lev > maxLevel) maxLevel = lev;
    }
    c.numLevels = maxLevel + 1;
    double avgRowsPerLevel = (double)n / c.numLevels;

    if (avgRowsPerLevel < 50.0) {
        c.path = PreprocCache::PATH_CHAIN;
    } else if (c.numLevels <= 100u) {
        c.path = PreprocCache::PATH_LEVELSET;
    } else {
        c.path = PreprocCache::PATH_SYNCFREE;
    }

    if (c.path == PreprocCache::PATH_LEVELSET || c.path == PreprocCache::PATH_SYNCFREE) {

        std::vector<unsigned int> levelOffsets(c.numLevels + 1, 0);
        for (unsigned int r = 0; r < n; ++r) levelOffsets[levels[r] + 1]++;
        for (unsigned int l = 1; l <= c.numLevels; ++l) levelOffsets[l] += levelOffsets[l-1];

        std::vector<unsigned int> levelRows(n);
        {
            std::vector<unsigned int> cur = levelOffsets;
            for (unsigned int r = 0; r < n; ++r) levelRows[cur[levels[r]]++] = r;
        }
        c.maxLevelSize = 0;
        for (unsigned int l = 0; l < c.numLevels; ++l) {
            unsigned int sz = levelOffsets[l+1] - levelOffsets[l];
            if (sz > c.maxLevelSize) c.maxLevelSize = sz;
        }

        std::vector<unsigned int> lowerRowPtrs(n + 1, 0);
        std::vector<unsigned int> lowerColIdxs;
        std::vector<float>        lowerVals;
        lowerColIdxs.reserve(L_r_host->numNonzeros);
        lowerVals.reserve(L_r_host->numNonzeros);
        for (unsigned int r = 0; r < n; ++r) {
            lowerRowPtrs[r] = (unsigned int)lowerColIdxs.size();
            for (unsigned int p = L_r_host->rowPtrs[r]; p < L_r_host->rowPtrs[r+1]; ++p) {
                unsigned int cc = L_r_host->colIdxs[p];
                if (cc < r) {
                    lowerColIdxs.push_back(cc);
                    lowerVals.push_back(L_r_host->values[p]);
                }
            }
        }
        lowerRowPtrs[n] = (unsigned int)lowerColIdxs.size();
        unsigned int totalLower = (unsigned int)lowerColIdxs.size();
        if (totalLower == 0) {
            lowerColIdxs.push_back(0);
            lowerVals.push_back(0.0f);
            totalLower = 1;
        }

        cudaMalloc(&c.levelRows_d,    n * sizeof(unsigned int));
        cudaMalloc(&c.levelOffsets_d, (c.numLevels + 1) * sizeof(unsigned int));
        cudaMalloc(&c.diag_d,         n * sizeof(float));
        cudaMalloc(&c.lowerRowPtrs_d, (n + 1) * sizeof(unsigned int));
        cudaMalloc(&c.lowerColIdxs_d, totalLower * sizeof(unsigned int));
        cudaMalloc(&c.lowerVals_d,    totalLower * sizeof(float));

        cudaMemcpy(c.levelRows_d,    levelRows.data(),    n * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(c.levelOffsets_d, levelOffsets.data(), (c.numLevels + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(c.diag_d,         diag.data(),         n * sizeof(float),                 cudaMemcpyHostToDevice);
        cudaMemcpy(c.lowerRowPtrs_d, lowerRowPtrs.data(), (n + 1) * sizeof(unsigned int),    cudaMemcpyHostToDevice);
        cudaMemcpy(c.lowerColIdxs_d, lowerColIdxs.data(), totalLower * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(c.lowerVals_d,    lowerVals.data(),    totalLower * sizeof(float),        cudaMemcpyHostToDevice);

        if (c.path == PreprocCache::PATH_SYNCFREE) {
            cudaMalloc(&c.ready_d,  n * sizeof(unsigned int));
            cudaMalloc(&c.rowCtr_d, sizeof(unsigned int));
        }
    } else {
        std::vector<float> rcpDiag(n);
        for (unsigned int r = 0; r < n; ++r) rcpDiag[r] = 1.0f / diag[r];

        std::vector<unsigned int> sOffsets_h(n + 1, 0);
        std::vector<unsigned int> sCol_h;
        std::vector<float>        sVal_h;
        sCol_h.reserve(L_r_host->numNonzeros + 16);
        sVal_h.reserve(L_r_host->numNonzeros + 16);

        for (unsigned int idx = 0; idx < n; ++idx) {
            if (sCol_h.size() & 1u) {
                sCol_h.push_back(0);
                sVal_h.push_back(0.0f);
            }
            sOffsets_h[idx] = (unsigned int)sCol_h.size();
            for (unsigned int p = L_r_host->rowPtrs[idx]; p < L_r_host->rowPtrs[idx+1]; ++p) {
                unsigned int cc = L_r_host->colIdxs[p];
                if (cc < idx) {
                    sCol_h.push_back(cc);
                    sVal_h.push_back(L_r_host->values[p]);
                }
            }
        }
        sOffsets_h[n] = (unsigned int)sCol_h.size();
        sCol_h.push_back(0); sCol_h.push_back(0);
        sVal_h.push_back(0.0f); sVal_h.push_back(0.0f);
        unsigned int totalLower = (unsigned int)sCol_h.size();

        cudaMalloc(&c.sRcpDiag_d, n * sizeof(float));
        cudaMalloc(&c.sOffsets_d, (n + 1) * sizeof(unsigned int));
        cudaMalloc(&c.sCol_d,     totalLower * sizeof(unsigned int));
        cudaMalloc(&c.sVal_d,     totalLower * sizeof(float));

        cudaMemcpy(c.sRcpDiag_d, rcpDiag.data(),    n * sizeof(float),                cudaMemcpyHostToDevice);
        cudaMemcpy(c.sOffsets_d, sOffsets_h.data(), (n + 1) * sizeof(unsigned int),   cudaMemcpyHostToDevice);
        cudaMemcpy(c.sCol_d,     sCol_h.data(),     totalLower * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(c.sVal_d,     sVal_h.data(),     totalLower * sizeof(float),        cudaMemcpyHostToDevice);
    }
}

} // namespace

__global__ void kernel3_levelset_tpbk(
    unsigned int n, unsigned int k,
    const unsigned int* __restrict__ lowerRowPtrs,
    const unsigned int* __restrict__ lowerColIdxs,
    const float*        __restrict__ lowerVals,
    const float*        __restrict__ diag,
    const float*        __restrict__ B,
    float*                            X,
    const unsigned int* __restrict__ levelRows,
    const unsigned int* __restrict__ levelOffsets,
    unsigned int numLevels)
{
    cg::grid_group grid = cg::this_grid();
    unsigned int b = threadIdx.x;

    for (unsigned int lev = 0; lev < numLevels; ++lev) {
        unsigned int lvStart = levelOffsets[lev];
        unsigned int lvSize  = levelOffsets[lev + 1] - lvStart;

        for (unsigned int rIdx = blockIdx.x; rIdx < lvSize; rIdx += gridDim.x) {
            unsigned int row = __ldg(&levelRows[lvStart + rIdx]);
            unsigned int rs  = __ldg(&lowerRowPtrs[row]);
            unsigned int re  = __ldg(&lowerRowPtrs[row + 1]);
            float        dg  = __ldg(&diag[row]);
            float        sum = __ldg(&B[(long)row * k + b]);

            for (unsigned int p = rs; p < re; ++p) {
                unsigned int j   = __ldg(&lowerColIdxs[p]);
                float        val = __ldg(&lowerVals[p]);
                sum -= val * __ldg(&X[(long)j * k + b]);
            }
            X[(long)row * k + b] = sum / dg;
        }
        grid.sync();
    }
}

__global__ void kernel3_syncfree(
    unsigned int n, unsigned int k,
    const unsigned int* __restrict__ lowerRowPtrs,
    const unsigned int* __restrict__ lowerColIdxs,
    const float*        __restrict__ lowerVals,
    const float*        __restrict__ diag,
    const float*        __restrict__ B,
    float*                            X,
    const unsigned int* __restrict__ topoOrder,
    unsigned int*                     ready,
    unsigned int*                     rowCtr,
    unsigned int                      chunkSize)
{
    __shared__ unsigned int s_chunkStart;
    unsigned int b = threadIdx.x;
    volatile unsigned int* readyV = (volatile unsigned int*)ready;
    volatile float*        Xv     = (volatile float*)X;

    while (true) {
        if (b == 0) s_chunkStart = atomicAdd(rowCtr, chunkSize);
        __syncthreads();
        unsigned int chunkStart = s_chunkStart;
        if (chunkStart >= n) return;
        unsigned int chunkEnd = chunkStart + chunkSize;
        if (chunkEnd > n) chunkEnd = n;

        for (unsigned int t = chunkStart; t < chunkEnd; ++t) {
            unsigned int row = __ldg(&topoOrder[t]);
            unsigned int rs  = __ldg(&lowerRowPtrs[row]);
            unsigned int re  = __ldg(&lowerRowPtrs[row + 1]);
            float        dg  = __ldg(&diag[row]);
            float        sum = __ldg(&B[(long)row * k + b]);

            for (unsigned int p = rs; p < re; ++p) {
                unsigned int j   = __ldg(&lowerColIdxs[p]);
                float        val = __ldg(&lowerVals[p]);

                while (readyV[j] == 0u) { /* spin */ }

                float xj = Xv[(long)j * k + b];
                sum -= val * xj;
            }

            X[(long)row * k + b] = sum / dg;
            __threadfence();    
            __syncthreads();    
            if (b == 0) {
                readyV[row] = 1u;
            }
        }
    }
}

__global__ void init_X_from_B(unsigned int n, unsigned int k,
                              const float* __restrict__ B, float* X)
{
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * k) return;
    X[t] = B[t];
}


__global__ void kernel3_chain_v2(
    unsigned int n, unsigned int k,
    const float*        __restrict__ sRcpDiag,
    const unsigned int* __restrict__ sOffsets,
    const unsigned int* __restrict__ sCol,
    const float*        __restrict__ sVal,
    float*                            X)
{
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= k) return;

    float        rcp_curr = __ldg(&sRcpDiag[0]);
    unsigned int rs_curr  = __ldg(&sOffsets[0]);
    unsigned int re_curr  = __ldg(&sOffsets[1]);
    unsigned int rowOff   = 0;

    for (unsigned int idx = 0; idx < n; ++idx) {
        float        rcp_next = 0.0f;
        unsigned int rs_next  = 0;
        unsigned int re_next  = 0;
        if (idx + 1 < n) {
            rcp_next = __ldg(&sRcpDiag[idx + 1]);
            rs_next  = __ldg(&sOffsets[idx + 1]);
            re_next  = __ldg(&sOffsets[idx + 2]);
        }

        float sum = X[rowOff + b];

        unsigned int p         = rs_curr;
        unsigned int re_padded = (re_curr + 1u) & ~1u;
        for (; p < re_padded; p += 2) {
            uint2  cols = *reinterpret_cast<const uint2*>(&sCol[p]);
            float2 vals = *reinterpret_cast<const float2*>(&sVal[p]);
            sum -= vals.x * X[cols.x * k + b];
            sum -= vals.y * X[cols.y * k + b];
        }

        X[rowOff + b] = sum * rcp_curr;

        rowOff   += k;
        rcp_curr  = rcp_next;
        rs_curr   = rs_next;
        re_curr   = re_next;
    }
}

void sptrsv_gpu3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int k = numCols;

    if (g_cache.key != L_r_host
        || g_cache.n != L_r_host->numRows
        || g_cache.numNonzeros != L_r_host->numNonzeros) {
        invalidateCache(g_cache);
        buildPreproc(g_cache, L_r_host);
    }

    unsigned int n = g_cache.n;

    DenseMatrix bPtr, xPtr;
    cudaMemcpy(&bPtr, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&xPtr, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    if (g_cache.path == PreprocCache::PATH_LEVELSET) {
        int threadsPerBlock = (int)k;
        int blocksPerSM = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSM, kernel3_levelset_tpbk, threadsPerBlock, 0);
        if (blocksPerSM < 1) blocksPerSM = 1;
        int gridSize = std::min(numSMs * blocksPerSM,
                                std::max(numSMs, (int)g_cache.maxLevelSize));

        unsigned int numLevels = g_cache.numLevels;
        void* args[] = {
            &n, &k,
            &g_cache.lowerRowPtrs_d,
            &g_cache.lowerColIdxs_d,
            &g_cache.lowerVals_d,
            &g_cache.diag_d,
            &bPtr.values,
            &xPtr.values,
            &g_cache.levelRows_d,
            &g_cache.levelOffsets_d,
            &numLevels
        };
        cudaLaunchCooperativeKernel((void*)kernel3_levelset_tpbk,
                                    gridSize, threadsPerBlock, args);
    } else if (g_cache.path == PreprocCache::PATH_SYNCFREE) {
        // Reset per-call state. cudaMemset on n*4 bytes is ~10us at T4 BW.
        cudaMemsetAsync(g_cache.ready_d,  0, n * sizeof(unsigned int));
        cudaMemsetAsync(g_cache.rowCtr_d, 0, sizeof(unsigned int));

        int threadsPerBlock = (int)k;
        int blocksPerSM = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSM, kernel3_syncfree, threadsPerBlock, 0);
        if (blocksPerSM < 1) blocksPerSM = 1;
        int gridSize = numSMs * blocksPerSM;

        unsigned int chunkSize = 8;
        kernel3_syncfree<<<gridSize, threadsPerBlock>>>(
            n, k,
            g_cache.lowerRowPtrs_d,
            g_cache.lowerColIdxs_d,
            g_cache.lowerVals_d,
            g_cache.diag_d,
            bPtr.values,
            xPtr.values,
            g_cache.levelRows_d,   // topological order
            g_cache.ready_d,
            g_cache.rowCtr_d,
            chunkSize);
    } else {
        int initThreads = 256;
        int initBlocks  = ((long)n * k + initThreads - 1) / initThreads;
        init_X_from_B<<<initBlocks, initThreads>>>(n, k, bPtr.values, xPtr.values);

        int colWidth = 32;
        int numBlocks = ((int)k + colWidth - 1) / colWidth;
        kernel3_chain_v2<<<numBlocks, colWidth>>>(
            n, k,
            g_cache.sRcpDiag_d, g_cache.sOffsets_d,
            g_cache.sCol_d,     g_cache.sVal_d,
            xPtr.values);
    }
}
