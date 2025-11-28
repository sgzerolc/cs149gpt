Env problem: `module_ref.so: undefined symbol`

env: ubuntu 22.04, amd arch
python packages:
Python 3.10
torch==2.1.2
setuptools==80.0.0

Warm-up:
A 4-D tensor utilizes a row-major memory layout, where the inntermost dimension is stored
contiguously in memory. This structure is selected to maximize spatial locality and improve CPU
cache efficiency.

A row-major layout levarages hardware by optimizing for cache lines and SIMD vectorization. CPUs
don't read single bytes from RAM. Instead, they read contiguous chunks of data called cache lines.
If iterating over the outer dimension, every read would likely require fetching a new cache line
(64 bytes) to get one integer (4 bytes), wasting 60 bytes. It results in cache misses and massive
bandwidth waste.

CPUs use SIMD instructions to process multiple numbers at once. SIMD is designed to load contiguous
chunks of memory into vector registers. If data is scattered, those values will be fetched through
costly `gather` instruction.

Blocked matrix multiplication

Blocked matrix multiplication is based on matrix block multiplication. For example, given two
matices A and B to be multiplied to produce matrix C, they can be decomposed into sub-matrices and
still generate the same result.

Assume:
A, shape (i, j); B, shape (j, k); C, shape (i, k);
i = L + r1; j = L + r2; k = L + r3 where r1 < L, r2 < L, r3 < L.

We can compute C by blocks of size L x L:
```
          C_ik = A_ij * B_jk
C_(L+r1)(L+r3) = A_(L+r1)(L+r2) * B_(L+r2)(L+r3)
               = [A_LL   A_Lr2] *  [B_LL   B_Lr3]
                  A_r1L  A_r1r2     B_r2L  B_r2r3

CLL = A_LL * B_LL + A_Lr2 * B_r2L
CLr3 = A_LL * B_Lr3 + A_Lr2 * B_r2r3
Cr1L = A_r1L * B_LL + A_r1r2 * B_r2L
Cr1r3 = A_r1L * B_Lr3 + A_r1r2 * B_r2r3

```

Debugging:
Use small dataset.
python3 gpt149.py part2 -N 8
NR=4
N * d = 8 * 32

