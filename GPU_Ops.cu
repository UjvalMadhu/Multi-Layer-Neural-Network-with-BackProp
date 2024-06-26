//============================================
//  CMPE 755 High Performance Architectures
//============================================
// Projet: Multi Layer Neural Network
// GPU Functions File
//--------------------------------------------
// Authors: Ujval Madhu, Evan Ruttenberg
//============================================
#include <math.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#ifndef __COMMON_H__
#include "commonGPU.h"
#endif __COMMON_H__

__global__ void kern1(float** X, float** Y, float** W, int sam, int u) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	Y[sam][u] += (X[sam][idx] * W[idx + 1][u]);
}

// ForwardCPU : Performs one formward pass of the input using sigmoid activation as default
// X     = SxinF Input Matrix
// Y     = SxUnits Output Matrix
// W     = (inF+1) x Units Weight Matrix
// S     = Number of Input Samples
// inF   = Number of Input Features/units
// Units = Number of Output Units

void ForwardGPU(float** X, float** Y, float** W, int S, int inF, int Units) {
	dim3 dimg(1);
	dim3 blockDim(inF);
	for (int sam = 0; sam < S; sam++) {
		for (int u = 0; u < Units; u++) {
			Y[sam][u] = 0.0;
				kern1<<<dimg, blockDim>>>(X, Y, W, sam, u);
			Y[sam][u] += W[0][u];
			//Y[sam][u] = sigmoid(Y[sam][u]);
		}
	}

}

__global__ void kern2(float** X, float** Y) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	Y[idy][idx] = 1.0 / (1.0 + exp(-X[idy][idx]));
}

// SigmoidAct(float** X, float ** Y, int S, int Units) : Applies Sigmoid Activation to the SxUnits input matrix Y
// X = Input matrix
// Y = Output Matrix
// S = number of Samples 
// U = Number of Units

void SigmoidAct(float** X, float** Y, int S, int Units) {
	dim3 dimg(1);
	dim3 dims(Units, S);
	kern2<<<dimg, dims>>>(X, Y);
}

// expSum(Y, Units): calculates the sum of natural exponential of all the Units of Y
float expSum(float* Y, int Units) {
	float y_esum = 0.0;
	for (int u = 0; u < Units; u++) {
		if (isnan(exp(Y[u]))) {
		//	int i = 1;
			printf("nan in y %f %d\n", Y[u], u);
			exit(1);
		}
		y_esum += exp(Y[u]);
	}
	return(y_esum);
}

__global__ void kern3(float** X, float** Y, int sam, float esum_y) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	//if (isnan(exp(X[sam][idx]))) {
		//int i = 1;
		//printf("nan in X %f %d %d\n",X[sam][idx], sam, idx);
	//}
	Y[sam][idx] = exp(X[sam][idx]) / esum_y;
}

// NormExp(float**  X, float** Y,int S, int U): returns the normalized exponential of the matrix X
// X is a a matrix of size SxU
// Y is the output Matrix of size SxU
// U is the number of units in X
// S = number of input samples

void NormExp(float** X, float** Y, int S, int U) {
	dim3 dim(U);
	dim3 dimg(1);
	float esum_y;
	for (int sam = 0; sam < S; sam++) {
		esum_y = expSum(X[sam], U);
		//if (isnan(esum_y)) {
		//	int i = 1;
			//printf("nan in esum %d\n", sam);
		//}

		//for (int u = 0; u < U; u++) {
			kern3<<<dimg, dim>>>(X, Y, sam, esum_y);
		//}
	}
}

// CatCrEnt(Y, Y_O, S, U): Calculates the Categorical cross entropy Cost 
// Y   = Actual Output Matrix (SxU)
// Y_O = Calculated Output Matrix (SxU)
// C   = Cost Matrix (Sx1)
// S   = Number of Samples
// U   = Number of Units

void CatCrEnt(float** Y, float** Y_O, float** C, int S, int Units) {

	for (int sam = 0; sam < S; sam++) {
		C[sam][0] = 0.0;
		float er = 0.0;
		for (int u = 0; u < Units; u++) {
			if (Y_O[sam][u] < 0.000001) { er = 10; }
			else { er = log(abs(Y_O[sam][u])); }
			C[sam][0] -= Y[sam][u] * er;

		}
	}

}

__global__ void kern4(float** delta, float** Y, float** Y_O, float** Z_O, float z_sum, int sam) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	delta[sam][0] = -((Y[sam][idx] / Y_O[sam][idx]) * (z_sum * exp(Z_O[sam][idx]) - pow(exp(Z_O[sam][idx]), 2))) / pow(z_sum, 2);
}

__global__ void kern5(float** delta, float** WO_updt, float** Y_1, int sam) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	WO_updt[idy+1][idx] += delta[sam][idy] * Y_1[sam][idy];
}

__global__ void kern6(float** delta, float** WO_updt, int sam) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	WO_updt[0][idx] += delta[sam][idx];
}

__global__ void kern7(float** WO_updt, int S) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	WO_updt[idy][idx] /= (float) S;
}

//void backProp_O(float** Y, float** Y_O, float** Z_O, float** Y_1, float** WO_updt, int S, int UL, int UL_m1)
// Calculates the derivative of the cost w.r.t the weights, i.e the weight update value averaged across the given number of samples
// Y       = Actual Output Matrix (S x UL)
// Y_O     = Calculated Output Matrix (S x UL)
// Z_O     = Output Matrix before activation (S x UL)
// Y_1     = Input to output layer (S x UL_m1)
// WO_Updt = Matrix for storing derivatives w.r.t each weight (UL_m1+1 x UL)
// S = Number of Samples, UL = Number of units in output layer, UL_m1 = Number of units in Input Layer

void backProp_O(float** Y, float** Y_O, float** Z_O, float** Y_1, float** delta, float** WO_updt, int S, int UL, int UL_m1) {
	// delta = (dC/dY_O)(dY_O/dZ_O) ;  (ULx1) matix
	dim3 dimg(1);
	
	// Accumulating Errors/weight_updates for each sample
	dim3 dimjk(UL_m1, UL);
	for(int sam = 0; sam < S; sam++){
		float z_sum = expSum(Z_O[sam], UL);
		dim3 dimi(UL);
			kern4<<<dimg, dimi>>>(delta, Y, Y_O, Z_O, z_sum, sam);

		// weight Updates
		kern5<<<dimg, dimjk>>>(delta, WO_updt, Y_1, sam);
		// Bias Updates
		// Bias_update = dC/dB = delta

		kern6<<<dimg, dimi>>>(delta, WO_updt, sam);
	}

	// Taking average of all the errors by dividing by the total number of Samples

	kern7<<<dimg, dimjk>>>(WO_updt, S);
}

__global__ void kern8(float** delta, float** delta2, float** Y_1, float** WO, int sam) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	float tmp = delta[sam][idy] * WO[idx + 1][idy];
	atomicAdd(&(delta2[sam][idx]), tmp);
	__syncthreads();
	if (idy == 0)
		delta2[sam][idx] *= (Y_1[sam][idx] * (1 - Y_1[sam][idx]));

}

__global__ void kern9(float** W1_updt, float** delta2, float** X, int sam) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	W1_updt[idx][idy] = delta2[sam][idy];
	if (idx == 0)
		W1_updt[idx][idy] *= X[sam][idx];
}

// Void backProp_H()
// Calculates the derivative of the cost w.r.t the weights for the Hidden layer
// Y    = Actual Output Matrix (S x UL);                Y_O     = Calculated Output Matrix (S x UL)
// Z_O     = Output Matrix before activation (S x UL) ; Y_1     = Input to output layer (S x UL_m1)

void backProp_H(float ** X, float ** Y, float** Y_O, float** Y_1, float** WO, float** delta, float** W1_updt, int S, int UL_m2, int UL_m1, int UL) {
	// delta = Sum{(dC/dY_O)(dY_O/dZ_O)(dZ_O/dY_1)}(dY_1/dZ_1) ;  (ULx1) matix
	float** delta2 = allocFloatMat(S, UL_m1);
	dim3 dimg(1);
	if (delta2 == NULL) {
		printf("Memory Allocation Error for backprop_H");
	}
	for (int sam = 0; sam < S; sam++) {
		//for (int i = 0; i < UL_m1; i++) {
			dim3 dimu(UL_m1, UL);
			dim3 dimu2(UL_m2, UL_m1);
			kern8<<<dimg, dimu>>>(delta, delta2, Y_1, WO, sam);
			kern9<<<dimg, dimu2>>>(W1_updt, delta2, X, sam);
		//}
	}
	dim3 dimjk(UL_m2+1, UL_m1);
	// Taking average of all the errors by dividing by the total number of Samples
	kern7<<<dimg, dimjk>>>(W1_updt, S);
	freeFloatMat(delta2, S);
}


// updateW(float ** W, float ** W_updt, int U1, int U2, int eta)
// W      = Weight Matrix to be updated (U1 x U2)
// W_updt = Weight Update Value Matrix (U1 x U2)
// eta    = Learning rate
void updateW(float** W, float** W_updt,int U1,int U2,int eta){
	for (int i = 0; i < U1; i++) {
		for (int j = 0; j < U2; j++) {
			W[i][j] = W[i][j] - (eta * W_updt[i][j]);
		}
	}
}