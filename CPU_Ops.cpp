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

// Sigmoid: Returns the sigmoid of the input value
// x = float input
float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
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

void CatCrEnt(float** Y, float** Y_O, float** C, int S, int Units) {
	
	for (int sam = 0; sam < S; sam++) {
		C[sam][0] = 0.0;
		float er = 0.0;
		for (int u = 0; u < Units; u++) {
			if (Y_O[sam][u] < 0.000001) { er = 10; }
			else { er = log(Y_O[sam][u]); }
			C[sam][0] -= Y[sam][u] * er;
			
		}
	}

}

//void backProp_O(float** Y, float** Y_O, float** Z_O, float** Y_1, float** WO_updt, int S, int UL, int UL_m1)
// Calculates the derivative of the cost w.r.t the weights, i.e the weight update value averaged across the given number of samples
// Y       = Actual Output Matrix (S x UL)
// Y_O     = Calculated Output Matrix (S x UL)
// Z_O     = Output Matrix before activation (S x UL)
// Y_1     = Input to output layer (S x UL_m1)
// WO_Updt = Matrix for storing derivatives w.r.t each weight (UL_m1+1 x UL)
// S = Number of Samples, UL = Number of units in output layer, UL_m1 = Number of units in Input Layer

void backProp_O(float** Y, float** Y_O, float** Z_O, float** Y_1, float** delta1, float** WO_updt, int S, int UL, int UL_m1) {
	// delta = (dC/dY_O)(dY_O/dZ_O) ;  (SxUL) matix
	
	// Accumulating Errors/weight_updates for each sample
	for(int sam = 0; sam < S; sam++){
		float z_sum = 0.0;
		z_sum = expSum(Z_O[sam], UL);
		for (int i = 0; i < UL; i++) {
			// dC/dY_O = Y/Y_O 
			// dY_O/dZ_O = (z_sum*Z_O - Z_O^2)/z_sum      for Normalized Exponential
			// delta = (dC/dY_O)(dY_O/dZ_O)
			delta1[sam][i] = -((Y[sam][i] / Y_O[sam][i]) * (z_sum * exp(Z_O[sam][i]) - pow(exp(Z_O[sam][i]), 2))) / pow(z_sum, 2);
			//if (sam == 0 & (i == 4 || i == 5 || i == 6)) {
			//	printf("\n\n ******************\n\n");
			//	printf("\n Delta Values at sam = %d, i = %d", sam,i);
			//	printf("\n Delta[i] = %f", delta1[sam][i]);
			//	printf("\n Y[0][i] = %f", Y[sam][i]);
			//	printf("\n Y_O[0][i] = %f", Y_O[sam][i]);
			//	printf("\n z_sum = %f", z_sum);
			//	printf("\n pow(z_sum, 2) = %f", pow(z_sum, 2));
			//	printf("\n Z_O[0][i] = %f", Z_O[sam][i]);
			//	printf("\n pow(Z_O[0][i], 2)) = %f", pow(Y_O[sam][i], 2));
			//	printf("\n\n ******************\n\n");
			//}
		}
		// weight Updates
		// WO_updt = dC/dW = (dC/dY_O)(dY_O/dZ_O)(dZ_O/dW) = delta*(dZ_O/dW) = delta*Y_1  for each weight
		
		for (int j = 0; j < UL_m1; j++) {
			for (int k = 0; k < UL; k++) {
				//if ((j == 0 || j == 1) & (k == 0 || k == 1 || k == 2)) {
				//	printf("\n\n ******************\n\n");
				//	printf("\n WO_Updt Values at sam = %d", sam);
				//	printf("\n Before Update: WO_updt[%d+1][%d] = %f",j,k, WO_updt[j + 1][k]);
				//	printf("\n  delta1[%d][%d] = %f",sam, k,delta1[sam][k]);
				//	printf("\n  Y_1[%d][%d] = %f", sam, j, Y_1[sam][j]);
				//}
				WO_updt[j+1][k] += delta1[sam][k] * Y_1[sam][j];

				//if ((j == 0 || j == 1) & (k == 0 || k == 1 || k == 2)) {
				//	printf("\n After Update: WO_updt[%d+1][%d] = %f", j, k, WO_updt[j + 1][k]);
				//}
			}
		}
		// Bias Updates
		// Bias_update = dC/dB = delta
		for (int k = 0; k < UL; k++) {
			//if (k == 0 || k == 1 || k == 2) {
			//	printf("\n\n");
			//	printf("\n Before Update: WO_updt[0][%d] = %f", k, WO_updt[0][k]);
			//}
			WO_updt[0][k] += delta1[sam][k];
			/*if (k == 0 || k == 1 || k == 2) {
				printf("\n After Update: WO_updt[0][%d] = %f", k, WO_updt[0][k]);
			}*/
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

void backProp_H(float ** X, float ** Y, float** Y_O, float** Y_1, float** WO, float** delta1, float** delta2, float** W1_updt, int S, int UL_m2, int UL_m1, int UL) {
	

	//for (int sam = 0; sam < S; sam++) {
	//	for (int i = 0; i < UL_m2; i++) {
	//		for (int j = 0; j < UL_m1; j++) {
	//			delta2[sam][j] = 0.0;
	//			W1_updt[i][j]  = 0.0;
	//			if(i == (UL_m2-1)){ W1_updt[i][j] = 0.0; }
	//			for (int u = 0; u < UL; u++) {
	//				delta2[sam][j] += delta1[sam][u] * WO[j + 1][u];
	//			}
	//			if (sam == 0 & (j == 0 || j == 1)) {
	//				printf("\n delta2_m1[%d][%d] = %f",sam,j, delta2[sam][j]);
	//				printf("\n Y_1[%d][%d] = %f", sam, j, Y_1[sam][j]);
	//				printf("\n 1 - Y_1[%d][%d] = %f", sam, j, 1 - Y_1[sam][j]);
	//				printf("\n delta2_m2[%d][%d] = %f", sam, j, Y_1[sam][j] * (1 - Y_1[sam][j]));
	//			}
	//			delta2[sam][j] = delta2[sam][j] * Y_1[sam][j] * (1 - Y_1[sam][j]);
	//			if (i == 0) {
	//				W1_updt[0][j] += delta2[sam][j];
	//			}
	//			W1_updt[i+1][j] += delta2[sam][j] * X[sam][i];
	//		}
	//	}
	//}
	// delta = Sum{(dC/dY_O)(dY_O/dZ_O)(dZ_O/dY_1)}(dY_1/dZ_1) ;  (SxUL_m1) matix
	for(int sam = 0; sam < S; sam++){
		// delta2 Calculation
		for (int u1 = 0; u1 < UL_m1; u1++) {
			for(int u = 0; u < UL; u++){
				delta2[sam][u1] += delta1[sam][u] * WO[u1 + 1][u];
			}
			delta2[sam][u1] *= (Y_1[sam][u1] * (1 - Y_1[sam][u1]));
		}
		 //Weight_update calculation
		for (int f = 0; f < UL_m2; f++) {
			for (int u1 = 0; u1 < UL_m1; u1++) {
				if (f == 0) {
					W1_updt[f][u1] = delta2[sam][u1];
				}
				else {
					W1_updt[f][u1] = delta2[sam][u1] * X[sam][f];
				}
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
void updateW(float** W, float** W_updt,int U1,int U2,float eta){
	for (int i = 0; i < U1; i++) {
		for (int j = 0; j < U2; j++) {
			W[i][j] = W[i][j] - (eta * W_updt[i][j]);
			/*if (i == 1 & j == 0) {
				printf("\n  Weight Values");
				printf("\n W[1][0] = %f", W[i][j]);
				printf("\n W_updt[1][0] = %f", W_updt[i][j]);
				printf("\n eta = %f", eta);
				printf("\n eta*W_updt[1][0] = %f", eta * W_updt[i][j]);
				printf("\n W[1][0] - (eta * W_updt[1][0]) = %f", W[i][j] - (eta * W_updt[i][j]));
			}*/
		}

	}

}

// Print Statements
// ForwardCPU print Statements
//if ((sam == 0) & (u == 1)) {
//	if (f == 0) { printf("\n\n**At S =0, u = 1"); }
//	printf("\n %d: X[0][f] * W[f + 1][u] = %f", f, X[sam][f] * W[f + 1][u]);
//	printf("\n %d: Z[0][u] = %f", f, Y[sam][u]);
//	printf("\n %d: X[0][f] = %f", f, X[sam][f]);
//	printf("\n %d: W[0][f+1] = %f", f, W[f + 1][u]);
//}
//if ((sam == 1) & (u == 1)) {
//	if (f == 0) { printf("\n\n**At S = 1 u = 1"); }
//	printf("\n %d: X[0][f] * W[f + 1][u] = %f", f, X[sam][f] * W[f + 1][u]);
//	printf("\n %d: Z[0][u] = %f", f, Y[sam][u]);
//	printf("\n %d: X[0][f] = %f", f, X[sam][f]);
//	printf("\n %d: W[0][f+1] = %f", f, W[f + 1][u]);
//}