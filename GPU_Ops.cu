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
		y_esum += exp(Y[u]);
	}
	return(y_esum);
}

__global__ void kern3(float** X, float** Y, int sam, float esum_y) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
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
		for (int u = 0; u < Units; u++) {
			C[sam][0] -= (float) Y[sam][u] * log(abs(Y_O[sam][u]));
		}
	}

}

__global__ void kern4(float** delta, float** Y, float** Y_O, float** Z_O, float z_sum, int sam) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	delta[idx][0] = (Y[sam][idx] / Y_O[sam][idx]) * (z_sum * Z_O[sam][idx] - pow(Z_O[sam][idx], 2)) / z_sum;
}

__global__ void kern5(float** delta, float** WO_updt, float** Y_1, int sam) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	WO_updt[idy+1][idx] += delta[idy][0] * Y_1[sam][idy];
}

__global__ void kern6(float** delta, float** WO_updt) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	WO_updt[0][idx] += delta[idx][0];
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

void backProp_O(float** Y, float** Y_O, float** Z_O, float** Y_1, float** WO_updt, int S, int UL, int UL_m1) {
	// delta = (dC/dY_O)(dY_O/dZ_O) ;  (ULx1) matix
	dim3 dimg(1);
	float** delta = allocFloatMat(UL, 1);
	if (delta == NULL) {
		printf("Memory Allocation Error for backprop_O");
	}
	
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

		kern6<<<dimg, dimi>>>(delta, WO_updt);
	}

	// Taking average of all the errors by dividing by the total number of Samples

	kern7<<<dimg, dimjk>>>(WO_updt, S);
	freeFloatMat(delta, UL);
}

__global__ void kern8(float** X, float** delta, float** Y, float** Y_O, float** WO, float** W1_updt, float** Y_1, int UL, int k, int sam) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	delta[idx][0] = 0.0;
	for (int i = 0; i < UL; i++)
		delta[idx][0] += (Y[sam][i] / Y_O[sam][i]) * (Y_O[sam][i] * (1 - Y_O[sam][i])) * WO[idx + 1][i];
	delta[idx][0] *= Y_1[sam][idx] * (1 - Y_1[sam][idx]);
	if (k == 0)
		W1_updt[0][idx] += delta[idx][0];
	W1_updt[k+1][idx] += delta[idx][0] * X[sam][k];
}

// Void backProp_H()
// Calculates the derivative of the cost w.r.t the weights for the Hidden layer
// Y    = Actual Output Matrix (S x UL);                Y_O     = Calculated Output Matrix (S x UL)
// Z_O     = Output Matrix before activation (S x UL) ; Y_1     = Input to output layer (S x UL_m1)

void backProp_H(float ** X, float ** Y, float** Y_O, float** Y_1, float** WO, float** W1_updt, int S, int UL_m2, int UL_m1, int UL) {
	// delta = Sum{(dC/dY_O)(dY_O/dZ_O)(dZ_O/dY_1)}(dY_1/dZ_1) ;  (ULx1) matix
	float** delta = allocFloatMat(UL_m1, 1);
	dim3 dimg(1);
	if (delta == NULL) {
		printf("Memory Allocation Error for backprop_H");
	}
	for (int sam = 0; sam < S; sam++) {
		for (int i = 0; i < UL_m2; i++) {
			dim3 dimu(UL_m1);
			kern8<<<dimg, dimu>>>(X, delta, Y, Y_O, WO, W1_updt, Y_1, UL, i, sam);

		}
	}
	dim3 dimjk(UL_m2+1, UL_m1);
	// Taking average of all the errors by dividing by the total number of Samples
	kern7<<<dimg, dimjk>>>(W1_updt, S);
	freeFloatMat(delta, UL_m1);
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