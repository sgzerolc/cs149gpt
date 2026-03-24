Env problem: `module_ref.so: undefined symbol`

env: ubuntu 22.04, amd arch
python packages:
Python 3.10
torch==2.1.2
setuptools==80.0.0
numpy=1.26.4

## Excercises

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

Mistakes:

1. 4-quadrant approach: split the matrix into 4 quadrants (LL, Lr, rL, rr) with one remainder
   per dimension, requiring 8 separate tile calls. Simpler alternative: loop over tiles with
   min() to handle remainders naturally.

2. Reusing transposed helpers for non-transposed multiply: vecABBlocked calls vecAB_tTile, which
   was written for Q×Kᵀ. The access pattern for P×V (non-transposed) is different — indexing
   may be wrong.

3. B_y is uninitialized in vecABBlocked — declared but never assigned, then passed to
   vecAB_tUpdate on the last line.

Debugging:
Use small dataset.
`python3 gpt149.py part2 -N 8`
NR=4
N * d = 8 * 32

## Design Ideas

### Attention mechanism

Input: Q, K, V — all N×d matrices.

Step 1: S = Q × Kᵀ (N×N) — score all word pairs
Step 2: P = softmax(S) row by row — normalize to probabilities
Step 3: O = P × V (N×d) — weighted sum of value vectors

### Part 2: Blocked Matmul + Unfused Softmax

Problem: Naive matmul has terrible cache behavior — values get evicted before reuse.

Idea: Process TILE×TILE blocks. The output block stays in cache while accumulating partial
products through the shared dimension. min() handles remainder tiles at edges.

Three separate passes: compute full S (N×N) -> softmax all rows -> compute full O. Each pass
reads/writes the entire N×N matrix.

Memory: O(N²) for S matrix. Softmax does NOT need blocking — it's just a row-wise operation.

### Part 3: Fused Attention + OpenMP

Problem: Part 2 materializes the N×N matrix and makes three separate passes over it.

Idea: Once you have one row of S, you can softmax it and multiply by V immediately. You never
need that row again. Fuse all three steps per row and reuse a single N-length scratch buffer.

Parallelism: Each row is fully independent. Give each thread its own scratch buffer, then
collapse(3) over batch × head × row. B × H × N total independent iterations.

Memory: O(N × T) where T = number of threads. Drops from ~N² to ~N.

### OpenMP

Basics: `#pragma omp parallel for` above a loop distributes iterations across CPU threads
automatically. `collapse(n)` merges n perfectly nested loops into one big iteration space.

```cpp
#pragma omp parallel for collapse(2)
for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++)
        // B*H iterations distributed across threads
```

The one rule: every iteration must be independent. If iteration 5 reads what iteration 3
wrote, you have a race condition.

Why fusion enables parallelism: In Parts 1 & 2, the three passes (matmul -> softmax -> matmul)
are separate — parallelizing one pass still requires synchronizing before the next. In Part 3,
each row is a self-contained unit (compute S row -> softmax -> multiply by V -> write O[i]).
No row reads another row's result, so all rows can run in parallel.

Scratch buffer sharing: The N-length scratch buffer can't be shared between threads — that's
a race condition. Solution: preallocate one buffer per thread. Each thread gets its own copy
based on its thread ID. Variables declared inside the parallelized loop are automatically
private to each thread.

### Part 4: Flash Attention

Problem: Part 3 still computes a full row of S (length N) before softmaxing.

Idea: Break the row into Bc-sized chunks. Process one K/V block at a time, maintaining a
running denominator and running output that get corrected as each new block arrives.

The correction: After seeing a new block, the old output was divided by l_old but should be
divided by (l_old + l_new). Scale old O by l_old/(l_old + l_new), add new block's contribution
normalized by the same total.

Memory: O(Br × Bc) for S_block + O(Br) for denominators. Overall O(N).

Tradeoff: More computation (rescaling at every block), so actually slower than Part 3. The win
is memory footprint, which matters at large N on real hardware with limited SRAM.

Architecture implication — why cache-sized blocks:
- SRAM (L1 cache): ~1-4 ns latency, ~32-64 KB capacity
- DRAM (main memory): ~50-100 ns latency, ~8-64 GB capacity
- That's a ~50-100x latency gap per access.
- Blocking doesn't reduce total computation — it reduces trips to DRAM. Load a tile into
  cache, reuse it fully, then move on.
- Flash Attention sizes Br and Bc so that the working set (Q block Br×d, K block Bc×d,
  S_block Br×Bc, V block Bc×d) all fit in SRAM simultaneously. Block sizes are derived
  from M (cache size in floats, e.g. M=131072).
- Intuition: closer to CPU = faster but smaller. You want your working set to fit in the
  fastest level possible.

## Essence of Inner Loops

The output element determines loop order:
- i, j (outer) select which output element
- k (inner) accumulates into that element

### Part 2

Q × Kᵀ -> S: six nested loops. Outer three step by TILE, inner three within a tile.
```
for ti in [0, N, TILE):          // tile over rows of S
  for tj in [0, N, TILE):        // tile over cols of S
    for tk in [0, d, TILE):      // tile over shared dim
      for i in [ti, min(ti+TILE, N)):
        for j in [tj, min(tj+TILE, N)):
          for k in [tk, min(tk+TILE, d)):
            S[i][j] += Q[i][k] * K[j][k]    // K[j][k] because Kᵀ[k][j] = K[j][k]
```

Softmax: two passes per row.
```
for i in [0, N):
  sum = 0
  for j in [0, N):
    sum += exp(S[i][j])
  for j in [0, N):
    P[i][j] = exp(S[i][j]) / sum
```

P × V -> O: same tiling, different dimensions.
```
for ti in [0, N, TILE):
  for tj in [0, d, TILE):        // cols of O are d, not N
    for tk in [0, N, TILE):      // shared dim is N
      for i in [ti, min(ti+TILE, N)):
        for j in [tj, min(tj+TILE, d)):
          for k in [tk, min(tk+TILE, N)):
            O[i][j] += P[i][k] * V[k][j]
```

### Part 3

Everything per row i, with thread-local scratch buffer (length N):
```
#pragma omp parallel for collapse(3)
for b, h, i:
  // Row of Q × Kᵀ -> scratch
  for j in [0, N):
    scratch[j] = 0
    for k in [0, d):
      scratch[j] += Q[i][k] * K[j][k]

  // Softmax in place
  sum = 0
  for j in [0, N):
    scratch[j] = exp(scratch[j])
    sum += scratch[j]
  for j in [0, N):
    scratch[j] /= sum

  // scratch × V -> O[i]
  for j in [0, d):
    O[i][j] = 0
    for k in [0, N):
      O[i][j] += scratch[k] * V[k][j]
```

### Part 4

For a fixed Q block (rows qi..qi+Br), sweep K/V blocks (cols kj..kj+Bc):
```
for qi in [0, N, Br):
  l[0..Br] = 0                        // running denominators
  O[qi..qi+Br] = 0                    // running output

  for kj in [0, N, Bc):
    // S_block (Br × Bc) = Q_block × K_blockᵀ
    for i in [0, Br):
      for j in [0, Bc):
        S_block[i][j] = 0
        for k in [0, d):
          S_block[i][j] += Q[qi+i][k] * K[kj+j][k]

    // Exp + local row sums
    for i in [0, Br):
      l_new[i] = 0
      for j in [0, Bc):
        S_block[i][j] = exp(S_block[i][j])
        l_new[i] += S_block[i][j]

    // Correct old O and add new contribution
    for i in [0, Br):
      scale = l[i] / (l[i] + l_new[i])
      for j in [0, d):
        O[qi+i][j] *= scale
        for k in [0, Bc):
          O[qi+i][j] += S_block[i][k] * V[kj+k][j] / (l[i] + l_new[i])

    // Update running denominators
    for i in [0, Br):
      l[i] += l_new[i]
```
