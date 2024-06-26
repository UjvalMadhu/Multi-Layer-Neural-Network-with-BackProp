//============================================
//  CMPE 755 High Performance Architectures
//============================================
// Projet: Multi Layer Neural Network
// Main File
//--------------------------------------------
// Authors: Ujval Madhu, Evan Ruttenberg
//============================================
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <cuda.h>

#ifndef __COMMON_H__
#include "commonGPU.h"
#endif

#define CHECK_DERR(crud) derr = crud; if (derr != 0) {fprintf(stderr, "Line %d: %s\n", __LINE__, cudaGetErrorString(derr)); exit(EXIT_FAILURE);}
cudaError_t derr;
// Constant Declaration
const int F  = 500;     //Number of Input Features
const int S  = 60000/2;   //Number of Samples
const int U1 = 256;     //Number of Units of Layer 1
const int U2 = 256;     //Number of Units of Layer 2
const int UL = 10;
const int E = 30; //Epochs
const float eta    = 0.001;   //Learning Rate

//========== M A I N   F U N C T I O N================

int main(int argc, char* argv[]) {

	CHECK_DERR(cudaSetDeviceFlags(cudaDeviceMapHost))
	// ======== Feature Extraction ============ //
	// Training Dataset
	FILE* train_set = fopen("../mnist_train.csv", "r");
	if (train_set== NULL) {
		fprintf(stderr, "Missing training data\n");
		exit(1);
	}


	// y: Output Vector Memory Allocation
	float* y_num = (float*)malloc(S * sizeof(float)); // y = 60000x1 Matrix
	if (y_num == NULL) {
		fprintf(stderr, "Missing training data\n");
		exit(1);
	}


	// x: Input Vector Memory Allocation
	float** x = allocFloatMat(S, F, true);                    // x = 60000x784 Matrix
	if (x == NULL) {
		exit(1);
	}


	// Populating the input and Output Matrices
	uint8_t tmp ;
	for (int i = 0; i < S; i++) {
		fscanf(train_set, "%hhu", &tmp);
		y_num[i] = (float) tmp;
		for (int j = 0; j < F; j++) {
			fscanf(train_set, ",%hhu", &tmp);
			/*if (i == 0) {
				printf("\n x[0][%u ] = %u \n", j, tmp);
			}*/
			x[i][j] = (float)(tmp/255.0);    // Normalizaing x
			//if (isnan(x[i][j])) {
			//	printf("nan x %d %d %hhu\n", i, j, tmp);
			//}
		}
	}
	/* //Print Input and Output Values
	for (int i = 0; i < 784; i++) {
		printf("\n x[0][% d] = %f \n", i, x[0][i]);
	}
	for (int i = 0; i < 10; i++) {
		printf("\n y_num[% d] = %u \n", i, y_num[i]);
	}*/

	
	//======= CATEGORICAL ENCODING OF OUTPUT =======//

	float** y = allocFloatMat(S, 10, true);
	if (y == NULL) {
		printf("Memory Allocation Error for Y");
		exit(1);
	}

	for (int i = 0; i < S; i++) {
		for (int j = 0; j < 10; j++) {
			y[i][j] = (j == y_num[i]) ? 1 : 0;
		}
	}

	/*for (int i = 0; i < 10; i++) {
		printf("\n y[% d] = ", i);
		for (int j = 0; j < 10; j++) {
			printf("%u ,", y[i][j]);
		}
		printf("\n", i);
	}*/

	//======================= WEIGHT MATRICES ==========================//

	// w1: Layer 1 Weight Vector Memory Allocation
	float** w1 = WeightGen(F, U1);
	if (w1 == NULL) {
		printf("\n Layer 1 Weight Matrix Allocation Error \n");
		exit(1);
	}

	// w1_updt: Layer 1 Weight Updation Matrix Memory Allocation
	float** w1_updt = WeightGen(F, U1);
	if (w1_updt == NULL) {
		printf("\n Layer 1 Weight Matrix Allocation Error \n");
		exit(1);
	}

	// wO_updt: Output Layer Weight Updation Matrix Memory Allocation
	float** wO_updt = WeightGen(U1, 10);
	if (wO_updt == NULL) {
		printf("\n Layer 1 Weight Matrix Allocation Error \n");
		exit(1);
	}

	// wO: Output Layer Weight Vector Memory Allocation
	float** wO = WeightGen(U1, 10);
	if (wO == NULL) {
		printf("\n Layer 1 Weight Matrix Allocation Error \n");
		exit(1);
	}

	float** z_O = allocFloatMat(S/E, 10);
	if (z_O == NULL) {
		printf("\n Output Layer Z Units Matrix Allocation Error \n");
		exit(1);
	}
	// y_O: Output layer units, 10 in number
	float** y_O = allocFloatMat(S/E, 10);
	if (y_O == NULL) {
		printf("\n Output Layer Y Units Matrix Allocation Error \n");
		exit(1);
	}
	float** delta1 = allocFloatMat(S/E, UL);
	if (delta1 == NULL) {
		printf("Memory Allocation Error for delta1");
		exit(1);
	}

	// In this Implementation we will be using Batch Gradient Descent Optimization
	// 
	// 
	//=============================== Layer One ===================================//

	// z_1 layer one units
	float** z_1 = allocFloatMat(S/E, U1);
	if (z_1 == NULL) {
		printf("\n Layer 1 Z Units Matrix Allocation Error \n");
		exit(1);
	}

	// y_1 layer one units
	float** y_1 = allocFloatMat(S/E, U1);
	if (y_1 == NULL) {
		printf("\n Layer 1 Y Units Matrix Allocation Error \n");
		exit(1);
	}
	float** C = allocFloatMat(S/E,1, true);
	if (C == NULL) {
		printf("Memory Allocation Error for Cost C");
		exit(1);
	}
	float** xbd = allocFloatMat(S, F);
	for (int i = 0; i < S; i++)
	CHECK_DERR(cudaMemcpy(xbd[i], x[i], F * sizeof(float), cudaMemcpyHostToDevice))
	float **yd;
	CHECK_DERR(cudaMalloc((void ***) &yd, S * sizeof(float*)))
	for (int i = 0; i < S; i++) {
		CHECK_DERR(cudaMalloc((void **) &(yd[i]), 10 * sizeof(float)))
		CHECK_DERR(cudaMemcpy(yd[i], y[i], 10 * sizeof(float), cudaMemcpyHostToDevice))
	}
	clock_t timenow = clock();
	for (int e = 0; e < E; e++) {
		ForwardGPU(&(xbd[(S/E)*e]), z_1, w1, S/E, F, U1);       // Forward Pass with Sigmoid activation

		SigmoidAct(z_1, y_1, S/E, U1);

		//============================ Output Layer =====================================//

		// z_O: Output layer units, 10 in number


		// The output layer does not use the sigmoid activation function but
		// uses the Normalized Exponential Activation so that the total sum of
		// all Output values of the network for a given sample = 1 and individual
		// outtput units will have different probabilites
		// Our goal would be to have the correct unit have the highest probability of 1.

		ForwardGPU(y_1, z_O, wO, S/E, U1, UL);            // Forward Pass
		NormExp(z_O, y_O, S/E, UL);                       // Normalized Exponential Activation

		//============================== Backpropagation =================================//
		// In this implementation, we use categorical cross entropy cost
		// The categorical cross entropy cost  = sum over all classes {y*ln(y_O)}


		CatCrEnt(&(y[(S/E)*e]), y_O, C, S/E, 10);                        // Categorical Cross Entropy Cost for all samples

		// Since we are using BGD optimization we will be accumulating all the errors from all the
		// training samples and averaging them in the w_updt matices in the backProp step.


		// 1. Backpropagation at the Output Layer
		backProp_O(&(yd[(S/E)*e]), y_O, z_O, y_1, delta1, wO_updt, S/E, 10, U1);

		// 2. Backpropagation at the first Layer
		backProp_H(&(xbd[(S/E)*e]), &(yd[(S/E)*e]), y_O, y_1, wO, delta1, w1_updt, S/E, U2, U1, 10);

		updateW(w1, w1_updt, F, U1, eta);

		// 2. w2: U1 x 10

		updateW(wO, wO_updt, U1, UL, eta);
	}
	timenow = clock() - timenow;
	double gpu_time = (1000.0 * timenow) / (double) CLOCKS_PER_SEC;
	printf("GPU EPOCHS took %lf ms\n", gpu_time);
	printf("got\n");
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < UL; j++){
			printf("  %f  ", y_O[i][j]);
		}
		printf("\n");
	}
	printf("expected\n");
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < UL; j++){
			printf("  %f  ", y[i][j]);
		}
		printf("\n");
	}
	freeFloatMat(yd, S);
	freeFloatMat(xbd, S);
	freeFloatMat(x, S, true);
	freeFloatMat(y, S, true);
	free(y_num);
	freeFloatMat(C, S/E, true);
	freeFloatMat(w1, F+1);
	freeFloatMat(wO, U1+1);
	freeFloatMat(w1_updt, F+1);
	freeFloatMat(wO_updt, U1+1);
	freeFloatMat(y_1, S/E);
	freeFloatMat(y_O, S/E);
	freeFloatMat(z_1, S/E);
	freeFloatMat(z_O, S/E);
	freeFloatMat(delta1, S/E);
	return 0;
}
//==================================================
//     F U N C T I O N      D E F I N I T I O N S
// =================================================

// allocFloatMat: Allocates memory for a Matrix
// i = Number of Rows
// j = Number of Columns
// Returns a Double Pointer Matrix

float** allocFloatMat(int i, int j, bool reg) {
	float** mat;
	if (reg)
		mat = (float**)malloc(i * sizeof(float*));
	else
		CHECK_DERR(cudaMalloc((void ***) &mat, i * sizeof(float*)))
	if (mat == NULL) {
		return mat;
	}
	for (int ii = 0; ii < i; ii++) {
		if (reg)
			mat[ii] = (float*)malloc(j * sizeof(float));
		else
			CHECK_DERR(cudaMalloc((void **) &(mat[ii]), j * sizeof(float)))
		if (mat[ii] == NULL) {
			return (float**)mat[ii];
		}
	}
	return mat;
}

// freeFloatMat: Frees the memory allocated for a Matrix
// i = Number of Rows
// mat = Double pointer of matrix

void freeFloatMat(float** mat, int i, bool reg) {
	for (int ii = 0; ii < i; ii++) {
		if (reg)
			free(mat[ii]);
		else
			CHECK_DERR(cudaFree(mat[ii]))
	}
	if (reg)
		free(mat);
	else
		CHECK_DERR(cudaFree(mat))
}

// mac: Returns the  Multiply Accumulate Result of Two Vectors, for input and weight MAC
// x = input vector array 1
// w = weight vector array without bias term

float mac(float* x, float* w, int Feat) {
	float accum = 0.0;
	for (int i = 0; i < Feat; i++) {
		accum += x[i] * w[i];
	}
	return accum;
}

// Sigmoid: Returns the sigmoid of the input value
// x = float input

float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

// forward: Returns the y = w[0] + sum(x*w)
// x = input vector
// w = weight vector with bias
// F = Number of features of X

float forward(float* x, float* w, int F) {
	float sum = mac(x, &(w[1]), F);
	sum += w[0];
	return sum; //sigmoid(sum);
}

// WeightGen(U , F): Creates a weight matrix + the bias term for each unit, and 
// assigns random values to the elements
// U = number of units in current layer
// F = Number of features of input layer

float** WeightGen(int F, int U) {
	float** w = allocFloatMat(F + 1, U);              // w = (1(bias) + 784) x U  Matrix
	if (w == NULL) {
		return w;
	}

	//srand(time(NULL));                              // Random with Seed
	//printf("Weight Matrix");
	for (int i = 0; i < F + 1; i++) {
		for (int j = 0; j < U; j++) {
			w[i][j] = (float)rand() / (float)RAND_MAX;
			//printf("\n w[%d][%d] = %f \n", i,j, w[i][j]);
			w[i][j] *= pow(-1, rand());
			//printf("\n w[%d][%d] = %f \n", i, j, w[i][j]);
		}
		if (w[i] == NULL) {
			return w;
		}
	}
	return w;
}

