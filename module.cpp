#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>
#include <cstdio>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

// overwrite the value at (x, y) with val
inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

// overwrite the value at (x, y, z, b) with val instead of adding val to the original value
inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b,
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 *
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * embedded dimensionality would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */

    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < N; i++) {
                // Watch for dimensionality
                for (int j = 0; j < N; j++) {
                    /*
                     * Matrix multiplication: S = Q * K_t
                     * K_t [i, j] = K [j, i]
                     * For shapes being Q (i, k), K_t (k, j),
                     * QK_t [i, j] = sum of Q[i, k] * K [j, k] for k = 0 to d-1
                     */
                    float val = 0.0;
                    for (int k = 0; k < d; k++) {
                        float Q_ik = fourDimRead(Q, b, h, i, k, H, N, d);
                        float K_jk = fourDimRead(K, b, h, j, k, H, N, d);
                        val += Q_ik * K_jk;
                    }
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }
        }
    }

    /*
     * softmax w/o numerical stability, meaning no subtraction of max value
     * A vector: f(x) = [exp(x_0) ..  exp(x_n)]
     * l(x) = sum of f(x_i) for i = 0 to n
     * P_i = softmax(S_i) = f(x) / l(x)
     *
     * P[i, j] = exp(x_j)/sum(exp(x_0) + ... + exp(x_(n-1))) for i = 0 to N-1
     * P = P_i for i = 0 to n-1
     */
    for (int i = 0; i < N; i++) {
        float S_i = 0.0;
        for (int j = 0; j < N; j++) {
            float S_ij = std::exp(twoDimRead(QK_t, i, j, N));
            S_i += S_ij;
            twoDimWrite(QK_t, i, j, N, S_ij);
        }

        for (int j = 0; j < N; j++) {
            float P_ij = twoDimRead(QK_t, i, j, N) / S_i;
            twoDimWrite(QK_t, i, j, N, P_ij);
        }
    }

    /*
     * O = P * V, P (i, k), V (k, j)
     * O[i, j] = sum of P[i, k] * V[k, j] for k = 0 to N-1
     */
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < d; j++) {
                    float val = 0.0;
                    for (int k = 0; k < N; k++) {
                        float P_ik = twoDimRead(QK_t, i, k, N);
                        float V_kj = fourDimRead(V, b, h, k, j, H, N, d);
                        val += P_ik * V_kj;
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, val);
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

// Calculate AB = A * B = C
// 2D tensor * 4D tensor = 4D tensor
// A_y != B_x
// A (x, z), B (z, y), C (x, y).
void vecAB(std::vector<float> &CVec, std::vector<float> &AVec, std::vector<float> &BVec,
        int A_x, int A_y, int B_x, int B_y, int x, int y, int z,
        int B, int H, int N, int d) {
    int Ax, Ay, Bx, By;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < x; i++) {
                Ax = A_x + i;
                for (int j = 0; j < y; j++) {
                    Bx = B_x + j;
                    float val = 0.0f;
                    // Don't think there would be much performance boost to read one tensor
                    // at a time. So just read tensors together.
                    for (int k = 0; k < z; k++) {
                        int yi = A_y + k;
                        int xj = B_x + k;
                        float A_ik = twoDimRead(AVec, Ax, yi, N);
                        float B_kj = fourDimRead(BVec, b, h, xj, Bx, H, N, d);
                        val += A_ik * B_kj;
                    }
                    fourDimWrite(CVec, b, h, Ax, Bx, H, N, d, val);
                }
            }
        }
    }
}

// two 4D tensors
// B_t is the transpose of B
// A (x, z), B (z, y), C (x, y).
// B_t (y, z)
void vecAB_tUpdate(std::vector<float> &CVec, std::vector<float> &AVec, std::vector<float> &BVec,
        int A_x, int A_y, int B_x, int B_y, int x, int y, int z,
        int B, int H, int N, int d) {
    int Ax, Ay, Bx, By;
    float old;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < x; i++) {
                Ax = A_x + i;
                for (int j = 0; j < y; j++) {
                    Bx = B_x + j;
                    float val = 0.0;
                    // Don't think there would be much performance boost to read one tensor
                    // at a time. So just read tensors together.
                    for (int k = 0; k < z; k++) {
                        int yi = A_y + k;
                        int xj = B_y + k;
                        float A_ik = fourDimRead(AVec, b, h, Ax, yi, H, N, d);
                        float B_jk = fourDimRead(BVec, b, h, Bx, xj, H, N, d);
                        val += A_ik * B_jk;
                    }
                    // Adding old values to vals is not allowed
                    old = twoDimRead(CVec, Ax, Bx, N);
                    val += old;
                    if (h == 0 && b == 0) {
                        //printf("bar C(%d, %d) = val %f\n", Ax, Bx, val);
                        twoDimWrite(CVec, Ax, Bx, N, val);
                    }
                }
            }
        }
    }
}

void vecAB_tTile(std::vector<float> &CVec, std::vector<float> &AVec, std::vector<float> &BVec,
        int Ax, int Ay, int Bx, int By, int x, int y, int z,
        int B, int H, int N, int d, int NR) {
    int A_x, A_y, B_x, B_y;
    int L_i = x / NR;
    int L_j = z / NR;
    int L_k = y / NR;

    //printf("foo Shapes x %d y%d z %d\n", x, y, z);
    //printf("foo Main: A(%d + %d, %d + %d) * B(%d + %d, %d + %d)\n",
    //        Ax, x, Ay, z, Bx, y, By, z);

    if (x > NR) {
        x = NR;
    }

    if (y > NR) {
        y = NR;
    }

    if (z > NR) {
        z = NR;
    }

    for (int i = 0; i < L_i; i++) {
        A_x = Ax + NR * i;
        for (int k = 0; k < L_k; k++) {
            B_x = Bx + NR * k;
            for (int j = 0; j < L_j; j++) {
                A_y = Ay + NR * j;
                B_y = By + NR * j;
                // Iterate over z points
                //printf("foo A(%d, %d); B(%d, %d)\n",
                //        A_x, A_y, B_x, B_y);
                //printf("foo A(%d + %d, %d + %d) * B(%d + %d, %d + %d)\n",
                //        A_x, x, A_y, z, B_x, y, B_y, z);
                vecAB_tUpdate(CVec, AVec, BVec, A_x, A_y, B_x, B_y, x, y, z, B, H, N, d);
            }
        }
    }
}

void vecAB_tBlocked(std::vector<float> &CVec, std::vector<float> &AVec,
        std::vector<float> &BVec, int x, int y, int z,
        int B, int H, int N, int d) {
    int A_x, A_y, B_x, B_y;
    // Assume 4 byte float, 64 byte cache lines as well as N and d are very large
    // int L = 64;
    // number of floats in a cache line
    // int NR = L / sizeof(float);
    int NR = 32; // for simplicity, set NR to 16

    // Blocked matrix multiplication
    // Assume:
    // A, shape (x:i, z:j); B, shape (z:j, y:k); C, shape (x:i, y:k);
    // i = L + r1; j = L + r2; k = L + r3 where r1 < L, r2 < L, r3 < L.
    // Then:
    // C_ik = A_ij * B_jk
    // C_(L+r1)(L+r3) = A_(L+r1)(L+r2) * B_(L+r2)(L+r3)
    //                = [A_LL   A_Lr2] *  [B_LL   B_Lr3]
    //                   A_r1L  A_r1r2     B_r2L  B_r2r3
    int L_i = x / NR;
    int r1_i = x % NR;
    int L_j = z / NR;
    int r2_j = z % NR;
    int L_k = y / NR;
    int r3_k = y % NR;
    A_y = z - r2_j;
    B_x = y - r3_k;
    A_x = x - r1_i;
    //printf("Shapes N %d x d %d B %d H %d\n", N, d, B, H);
    //printf("z:A_y: %d, y:B_x: %d, x:A_x: %d\n", A_y, B_x, A_x);
    //printf("r1_i: %d, r2_j: %d, r3_k: %d\n", r1_i, r2_j, r3_k);

    // CLL = A_LL * B_LL + A_Lr2 * B_r2L
    vecAB_tTile(CVec, AVec, BVec, 0, 0, 0, 0, A_x, B_x, A_y, B, H, N, d, NR);
    vecAB_tTile(CVec, AVec, BVec, 0, A_y, 0, A_y, A_x, B_x, r2_j, B, H, N, d, NR);

    // CLr3 = A_LL * B_Lr3 + A_Lr2 * B_r2r3
    vecAB_tTile(CVec, AVec, BVec, 0, 0, B_x, 0, A_x, r3_k, A_y, B, H, N, d, NR);
    vecAB_tTile(CVec, AVec, BVec, 0, A_y, B_x, A_y, A_x, r3_k, r2_j, B, H, N, d, NR);

    // Cr1L = A_r1L * B_LL + A_r1r2 * B_r2L
    vecAB_tTile(CVec, AVec, BVec, A_x, 0, 0, 0, r1_i, B_x, A_y, B, H, N, d, NR);
    vecAB_tTile(CVec, AVec, BVec, A_x, A_y, 0, A_y, r1_i, B_x, r2_j, B, H, N, d, NR);

    // Cr1r3 = A_r1L * B_Lr3 + A_r1r2 * B_r2r3
    vecAB_tTile(CVec, AVec, BVec, A_x, 0, B_x, 0, r1_i, r3_k, A_y, B, H, N, d, NR);
    vecAB_tUpdate(CVec, AVec, BVec, A_x, A_y, B_x, B_y, r1_i, r3_k, r2_j, B, H, N, d);
}

// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    //vecAB_tUpdate(QK_t, Q, K, 0, 0, 0, 0, N, N, d, B, H, N, d);
    vecAB_tBlocked(QK_t, Q, K, N, N, d, B, H, N, d);

    for (int i = 0; i < N; i++) {
        float S_i = 0.0;
        for (int j = 0; j < N; j++) {
            float S_ij = std::exp(twoDimRead(QK_t, i, j, N));
            S_i += S_ij;
            twoDimWrite(QK_t, i, j, N, S_ij);
        }

        for (int j = 0; j < N; j++) {
            float P_ij = twoDimRead(QK_t, i, j, N) / S_i;
            twoDimWrite(QK_t, i, j, N, P_ij);
        }
    }

    vecAB(O, QK_t, V, 0, 0, 0, 0, N, d, N, B, H, N, d);

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    for (int b = 0; b < B; b++){

        //loop over heads
        for (int h = 0; h < H; h++){
            for (int i = 0; i < N ; i++){

		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});
                std::vector<float> ORow = formatTensor(ORowTensor);
		//YOUR CODE HERE
            }
	}
    }


    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor,
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {

    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
