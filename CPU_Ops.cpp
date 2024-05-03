//============================================
//  CMPE 755 High Performance Architectures
//============================================
// Projet: Multi Layer Neural Network
// CPU Functions File
//--------------------------------------------
// Authors: Ujval Madhu, Evan Ruttenberg
//============================================
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#ifndef __COMMON_H__
#include "common.h"
#endif __COMMON_H__

// ForwardCPU : Performs one formward pass of the input using sigmoid activation as default
// X     = SxinF Input Matrix
// Y     = SxUnits Output Matrix
// W     = (inF+1) x Units Weight Matrix
// S     = Number of Input Samples
// inF   = Number of Input Features/units
// Units = Number of Output Units

void ForwardCPU(float** X, float** Y, float** W, int S, int inF, int Units) {
	for (int sam = 0; sam < S; sam++) {
		for (int u = 0; u < Units; u++) {
			Y[sam][u] = 0.0;
			for (int f = 0; f < inF; f++) {
				Y[sam][u] = Y[sam][u] + X[sam][f] * W[f + 1][u];
			}
			Y[sam][u] += W[0][u];
			//Y[sam][u] = sigmoid(Y[sam][u]);
		}
	}

}

// SigmoidAct(float** X, float ** Y, int S, int Units) : Applies Sigmoid Activation to the SxUnits input matrix Y
// X = Input matrix
// Y = Output Matrix
// S = number of Samples 
// U = Number of Units

void SigmoidAct(float** X, float** Y, int S, int Units) {
	for (int sam = 0; sam < S; sam++) {
		for (int u = 0; u < Units; u++) {
			Y[sam][u] = sigmoid(X[sam][u]);
		}
	}
}

// expSum(Y, Units): calculates the sum of natural exponential of all the Units of Y
float expSum(float* Y, int Units) {
	float y_esum = 0.0;
	for (int u = 0; u < Units; u++) {
		y_esum += exp(Y[u]);
	}
	return(y_esum);
}


// NormExp(float**  X, float** Y,int S, int U): returns the normalized exponential of the matrix X
// X is a a matrix of size SxU
// Y is the output Matrix of size SxU
// U is the number of units in X
// S = number of input samples

void NormExp(float** X, float** Y, int S, int U) {

	float esum_y;
	for (int sam = 0; sam < S; sam++) {
		esum_y = expSum(X[sam], U);
		for (int u = 0; u < U; u++) {
			Y[sam][u] = exp(X[sam][u]) / esum_y;
		}
	}
}

// CatCrEnt(Y, Y_O, S, U): Calculates the Categorical cross entropy Cost 
// Y   = Actual Output Matrix (SxU)
// Y_O = Calculated Output Matrix (SxU)
// C   = Cost Matrix (Sx1)
// S   = Number of Samples
// U   = Number of Units

void CatCrEnt(uint8_t** Y, float** Y_O, float** C, int S, int Units) {
	
	for (int sam = 0; sam < S; sam++) {
		C[sam][0] = 0.0;
		for (int u = 0; u < Units; u++) {
			C[sam][0] -= Y[sam][u] * log(Y_O[sam][u]);
		}
	}

}

//void backProp_O(uint8_t** Y, float** Y_O, float** Z_O, float** Y_1, float** WO_updt, int S, int UL, int UL_m1)
// Calculates the derivative of the cost w.r.t the weights, i.e the weight update value averaged across the given number of samples
// Y       = Actual Output Matrix (S x UL)
// Y_O     = Calculated Output Matrix (S x UL)
// Z_O     = Output Matrix before activation (S x UL)
// Y_1     = Input to output layer (S x UL_m1)
// WO_Updt = Matrix for storing derivatives w.r.t each weight (UL_m1+1 x UL)
// S = Number of Samples, UL = Number of units in output layer, UL_m1 = Number of units in Input Layer

void backProp_O(uint8_t** Y, float** Y_O, float** Z_O, float** Y_1, float** WO_updt, int S, int UL, int UL_m1) {
	// delta = (dC/dY_O)(dY_O/dZ_O) ;  (ULx1) matix
	float** delta = allocFloatMat(UL, 1);
	if (delta == NULL) {
		printf("Memory Allocation Error for backprop_O");
	}
	
	// Accumulating Errors/weight_updates for each sample
	for(int sam = 0; sam < S; sam++){
		float z_sum = 0.0;
		z_sum = expSum(Z_O[sam], UL);
		for (int i = 0; i < UL; i++) {
			// dC/dY_O = Y/Y_O 
			// dY_O/dZ_O = (z_sum*Z_O - Z_O^2)/z_sum      for Normalized Exponential
			// delta = (dC/dY_O)(dY_O/dZ_O)
			delta[i][0] = (Y[sam][i] / Y_O[sam][i]) * (z_sum * Z_O[sam][i] - pow(Z_O[sam][i], 2)) / z_sum;
			if (sam == 0 & i == 1) {
				printf("\n Delta Values");
				printf("\n Delta[i] = %f", delta[i][0]);
				printf("\n Y[0][i] = %d", Y[0][i]);
				printf("\n Y_O[0][i] = %f", Y_O[0][i]);
				printf("\n z_sum = %f", z_sum);
				printf("\n Z_O[0][i] = %f", Z_O[0][i]);
				printf("\n pow(Z_O[0][i], 2)) = %f", pow(Z_O[0][i], 2));
			}
		}
		// weight Updates
		// WO_updt = dC/dW = delta*Y_1  for each weight
		for (int j = 0; j < UL_m1; j++) {
			for (int k = 0; k < UL; k++) {
				WO_updt[j+1][k] += delta[k][0] * Y_1[sam][j];
			}
		}
		// Bias Updates
		// Bias_update = dC/dB = delta
		for (int k = 0; k < UL; k++) {
			WO_updt[0][k] += delta[k][0];
		}
	}

	// Taking average of all the errors by dividing by the total number of Samples
	for (int j = 0; j < UL_m1 + 1; j++) {
		for (int k = 0; k < UL; k++) {
			WO_updt[j][k] = WO_updt[j][k]/S;
		}
	}

}

// Void backProp_H()
// Calculates the derivative of the cost w.r.t the weights for the Hidden layer
// Y    = Actual Output Matrix (S x UL);                Y_O     = Calculated Output Matrix (S x UL)
// Z_O     = Output Matrix before activation (S x UL) ; Y_1     = Input to output layer (S x UL_m1) 

void backProp_H(float ** X, uint8_t ** Y, float** Y_O, float** Y_1, float** WO, float** W1_updt, int S, int UL_m2, int UL_m1, int UL) {
	// delta = Sum{(dC/dY_O)(dY_O/dZ_O)(dZ_O/dY_1)}(dY_1/dZ_1) ;  (ULx1) matix
	float** delta = allocFloatMat(UL_m1, 1);
	if (delta == NULL) {
		printf("Memory Allocation Error for backprop_O");
	}
	for (int sam = 0; sam < S; sam++) {
		for (int i = 0; i < UL_m2; i++) {
			for (int j = 0; j < UL_m1; j++) {
				delta[j][0] = 0.0;
				for (int u = 0; u < UL; u++) {
					delta[j][0] += (Y[sam][u] / Y_O[sam][u]) * (Y_O[sam][u] * (1 - Y_O[sam][u])) * WO[j + 1][u];
				}
				delta[j][0] = delta[j][0] * Y_1[sam][j] * (1 - Y_1[sam][j]);
				if (i == 0) {
					W1_updt[0][j] += delta[j][0];
				}
				W1_updt[i+1][j] += delta[j][0] * X[sam][i];
			}
		}
	}

	// Taking average of all the errors by dividing by the total number of Samples
	for (int j = 0; j < UL_m2 + 1; j++) {
		for (int k = 0; k < UL_m1; k++) {
			W1_updt[j][k] = W1_updt[j][k] / S;
		}
	}
}


// updateW(float ** W, float ** W_updt, int U1, int U2, int eta)
// W      = Weight Matrix to be updated (U1 x U2)
// W_updt = Weight Update Value Matrix (U1 x U2)
// eta    = Learning rate
void updateW(float** W, float** W_updt,int U1,int U2,int eta){
	for (int i = 0; i < U1; i++) {
		for (int j = 0; j < U1; j++) {
			W[i][j] = W[i][j] - (eta * W_updt[i][j]);
		}
	}
}
