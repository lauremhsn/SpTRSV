#include "common.h"
#include <vector>
#include <algorithm>

__global__ void kernel3_levelset(
    unsigned int  lvSize,
    unsigned int  lvStart,
    unsigned int  k,
    const unsigned int* __restrict__ levelRows,
    const unsigned int* __restrict__ rowPtrs,
    const unsigned int* __restrict__ colIdxs,
    const float*        __restrict__ vals,
    const float*        __restrict__ diagArr,
    const float*        __restrict__ B,
    float*        X
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= lvSize * k) return;
    unsigned int row = levelRows[lvStart + tid / k];
    unsigned int b   = tid % k;

    float sum = B[row * k + b];
    float d   = __ldg(&diagArr[row]);

    for (unsigned int p = rowPtrs[row]; p < rowPtrs[row+1]; p++) {
        unsigned int j = __ldg(&colIdxs[p]);
        if (j < row) sum -= __ldg(&vals[p]) * X[j * k + b];
    }
    X[row * k + b] = sum / d;
}

__global__ void kernel3_seq(
    unsigned int n, unsigned int k,
    const unsigned int* __restrict__ rowOrder,
    const float*        __restrict__ diagArr,
    const unsigned int* __restrict__ offsets,
    const unsigned int* __restrict__ colIdx, 
    const float*        __restrict__ vals,
    const float*        __restrict__ B,
    float*        X
) {
    unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= k) return;

    for (unsigned int idx = 0; idx < n; idx++) {
        unsigned int i  = __ldg(&rowOrder[idx]);
        float d         = __ldg(&diagArr[idx]);
        unsigned int rs = __ldg(&offsets[idx]);
        unsigned int re = __ldg(&offsets[idx+1]);

        float sum = B[i * k + b];
        #pragma unroll 4
        for (unsigned int p = rs; p < re; p++)
            sum -= __ldg(&vals[p]) * X[__ldg(&colIdx[p]) * k + b];

        X[i * k + b] = sum / d;
        __syncthreads();
    }
}

void sptrsv_gpu3(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X,
                 CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols)
{
    unsigned int n = L_r_host->numRows;
    unsigned int k = numCols;

    CSRMatrix csrPtr; cudaMemcpy(&csrPtr, L_r, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    DenseMatrix bPtr, xPtr;
    cudaMemcpy(&bPtr, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&xPtr, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    std::vector<unsigned int> levels(n, 0);
    std::vector<float>        diagVec(n, 1.0f);
    unsigned int maxLevel = 0;

    for (unsigned int r = 0; r < n; r++) {
        unsigned int lev = 0;
        for (unsigned int p = L_r_host->rowPtrs[r]; p < L_r_host->rowPtrs[r+1]; p++) {
            unsigned int c = L_r_host->colIdxs[p];
            if (c < r) { if (levels[c]+1u > lev) lev = levels[c]+1u; }
            else if (c == r) { float v = L_r_host->values[p]; diagVec[r] = v?v:1.f; }
        }
        levels[r] = lev;
        if (lev > maxLevel) maxLevel = lev;
    }

    unsigned int numLevels = maxLevel + 1;
    double avgRows = (double)n / numLevels;

    // build level offsets and row order
    std::vector<unsigned int> levelOff(numLevels+1, 0);
    for (unsigned int r = 0; r < n; r++) levelOff[levels[r]+1]++;
    for (unsigned int l = 1; l <= numLevels; l++) levelOff[l] += levelOff[l-1];

    std::vector<unsigned int> levelRows(n);
    { std::vector<unsigned int> cur = levelOff;
      for (unsigned int r = 0; r < n; r++) levelRows[cur[levels[r]]++] = r; }

    float*        diag_d;
    unsigned int* levelRows_d;
    cudaMalloc(&diag_d,     n * sizeof(float));
    cudaMalloc(&levelRows_d, n * sizeof(unsigned int));
    cudaMemcpy(diag_d,      diagVec.data(),   n*sizeof(float),        cudaMemcpyHostToDevice);
    cudaMemcpy(levelRows_d, levelRows.data(), n*sizeof(unsigned int), cudaMemcpyHostToDevice);

    const int THREADS = 256;

    if (avgRows >= 50.0) {
        for (unsigned int l = 0; l < numLevels; l++) {
            unsigned int sz     = levelOff[l+1] - levelOff[l];
            unsigned int lStart = levelOff[l];
            if (sz == 0) continue;

            unsigned int total  = sz * k;
            unsigned int blocks = (total + THREADS - 1) / THREADS;

            kernel3_levelset<<<blocks, THREADS>>>(
                sz, lStart, k, levelRows_d,
                csrPtr.rowPtrs, csrPtr.colIdxs, csrPtr.values,
                diag_d, bPtr.values, xPtr.values
            );
            // default stream serializes — no sync needed
        }

    } else {
        // repack into compact arrays in topological row order
        std::vector<float>        sDiag(n);
        std::vector<unsigned int> sOffsets(n+1, 0);
        std::vector<unsigned int> sColIdx;
        std::vector<float>        sVals;
        sColIdx.reserve(L_r_host->rowPtrs[n]);
        sVals.reserve(L_r_host->rowPtrs[n]);

        for (unsigned int idx = 0; idx < n; idx++) {
            unsigned int i = levelRows[idx];
            sDiag[idx]     = diagVec[i];
            sOffsets[idx]  = (unsigned int)sColIdx.size();
            for (unsigned int p = L_r_host->rowPtrs[i]; p < L_r_host->rowPtrs[i+1]; p++) {
                unsigned int c = L_r_host->colIdxs[p];
                if (c < i) { sColIdx.push_back(c); sVals.push_back(L_r_host->values[p]); }
            }
        }
        sOffsets[n] = (unsigned int)sColIdx.size();

        unsigned int *sRow_d, *sOff_d, *sCol_d; float *sDiag_d, *sVal_d;
        cudaMalloc(&sRow_d,  n*sizeof(unsigned int));
        cudaMalloc(&sDiag_d, n*sizeof(float));
        cudaMalloc(&sOff_d,  (n+1)*sizeof(unsigned int));
        cudaMalloc(&sCol_d,  sColIdx.size()*sizeof(unsigned int));
        cudaMalloc(&sVal_d,  sVals.size()*sizeof(float));

        cudaMemcpy(sRow_d,  levelRows.data(), n*sizeof(unsigned int),         cudaMemcpyHostToDevice);
        cudaMemcpy(sDiag_d, sDiag.data(),     n*sizeof(float),                 cudaMemcpyHostToDevice);
        cudaMemcpy(sOff_d,  sOffsets.data(),  (n+1)*sizeof(unsigned int),      cudaMemcpyHostToDevice);
        cudaMemcpy(sCol_d,  sColIdx.data(),   sColIdx.size()*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(sVal_d,  sVals.data(),     sVals.size()*sizeof(float),       cudaMemcpyHostToDevice);

        int colWidth  = 64;
        int numBlocks = std::max(1, (int)k / colWidth);

        kernel3_seq<<<numBlocks, colWidth>>>(
            n, k, sRow_d, sDiag_d, sOff_d, sCol_d, sVal_d,
            bPtr.values, xPtr.values);

        cudaFree(sRow_d); cudaFree(sDiag_d); cudaFree(sOff_d);
        cudaFree(sCol_d); cudaFree(sVal_d);
    }

    cudaFree(diag_d);
    cudaFree(levelRows_d);
}
