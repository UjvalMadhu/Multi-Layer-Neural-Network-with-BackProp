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

// Constant Declaration
const int F  = 784;     //Number of Input Features
const int S  = 60000;   //Number of Samples
const int U1 = 128;     //Number of Units of Layer 1
const int U2 = 256;     //Number of Units of Layer 2 

//========== M A I N   F U N C T I O N================

int main(int argc, char* argv[]) {
	cudaSetDeviceFlags(cudaDeviceMapHost);
	// ======== Feature Extraction ============ //

	// Training Dataset
	FILE* train_set = fopen("mnist_train.csv", "r");
	if (train_set== NULL) {
		fprintf(stderr, "Missing training data\n");
		exit(1);
	}


	// y: Output Vector Memory Allocation
	uint8_t* y_num = (uint8_t*)malloc(S * sizeof(uint8_t)); // y = 60000x1 Matrix
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
		fscanf(train_set, "%hhu", &(y_num[i]));   
		for (int j = 0; j < F; j++) {
			fscanf(train_set, ",%hhu", &tmp);
			/*if (i == 0) {
				printf("\n x[0][%u ] = %u \n", j, tmp);
			}*/
			x[i][j] = (float)(tmp/255.0);    // Normalizaing x 
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

	uint8_t** y = (uint8_t **)malloc(S * sizeof(uint8_t *));
	if (y == NULL) {
		printf("Memory Allocation Error for Y");
		exit(1);
	}
	for (int ii = 0; ii < S; ii++) {
		y[ii] = (uint8_t*)malloc(10 * sizeof(uint8_t));
		if (y[ii] == NULL) {
			printf("Memory Allocation Error for Y[%u]",ii);
			exit(1);
		}
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

	// In this Implementation we will be using Batch Gradient Descent Optimization
	// 
	// 
	//=============================== Layer One ===================================//

	// z_1 layer one units
	float** z_1 = allocFloatMat(S, U1);
	if (z_1 == NULL) {
		printf("\n Layer 1 Z Units Matrix Allocation Error \n");
		exit(1);
	}

	// y_1 layer one units
	float** y_1 = allocFloatMat(S, U1);
	if (y_1 == NULL) {
		printf("\n Layer 1 Y Units Matrix Allocation Error \n");
		exit(1);
	}
	float** xd = allocFloatMat(S, F);
	for (int i = 0; i < S; i++)
		cudaMemcpy(xd[i], x[i], F * sizeof(float), cudaMemcpyHostToDevice);
	ForwardGPU(xd, z_1, w1, S, F, U1);       // Forward Pass with Sigmoid activation

	SigmoidAct(z_1, y_1, S, U1);

	//============================ Output Layer =====================================//

	// z_O: Output layer units, 10 in number
	float** z_O = allocFloatMat(S, 10);
	if (z_O == NULL) {
		printf("\n Output Layer Z Units Matrix Allocation Error \n");
		exit(1);
	}
	// y_O: Output layer units, 10 in number
	float** y_O = allocFloatMat(S, 10);
	if (y_O == NULL) {
		printf("\n Output Layer Y Units Matrix Allocation Error \n");
		exit(1);
	}

	// The output layer does not use the sigmoid activation function but
	// uses the Normalized Exponential Activation so that the total sum of
	// all Output values of the network for a given sample = 1 and individual
	// outtput units will have different probabilites
	// Our goal would be to have the correct unit have the highest probability of 1.

	ForwardGPU(y_1, z_O, wO, S, U1, 10);			// Forward Pass
	NormExp(z_O, y_O, S, 10);                       // Normalized Exponential Activation

	//============================== Backpropagation =================================//
	// In this implementation, we use categorical cross entropy cost
	// The categorical cross entropy cost  = sum over all classes {y*ln(y_O)}
	
	float** C = allocFloatMat(S,1, true);
	if (C == NULL) {
		printf("Memory Allocation Error for Cost C");
		exit(1);
	}
	CatCrEnt(y, y_O, C, S, 10);						// Categorical Cross Entropy Cost for all samples

	// Since we are using BGD optimization we will be accumulating all the errors from all the 
	// training samples and averaging them in the w_updt matices in the backProp step.

	uint8_t** yd;
	cudaMalloc((void***) &yd, S * sizeof(uint8_t*));
	for (int i = 0; i < S; i++) {
		cudaMalloc((void **) &(yd[i]), 10 * sizeof(uint8_t));
		cudaMemcpy(yd[i], y[i], 10 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	}
	// 1. Backpropagation at the Output Layer
	backProp_O(yd, y_O, z_O, y_1, wO_updt, S, 10, U1);

	// 2. Backpropagation at the first Layer
	backProp_H(xd, yd, y_O, y_1, wO, w1_updt, S, U2, U1, 10);
	for (int i = 0; i < S; i++)
		cudaFree(yd[i]);
	cudaFree(yd);
	freeFloatMat(xd, S);


	/*printf("Output Matrix");
	for (int sam = 0; sam < S; sam++){
		for (int u = 0; u < U; u++) {
			y_1[sam][u] = 0.0;
			for (int f = 0; f < F; f++) {
				y_1[sam][u] = y_1[sam][u] + x[sam][f] * w[f + 1][u];
			}
			y_1[sam][u] += w[0][u];
			y_1[sam][u] = sigmoid(y_1[sam][u]);
		}
	}*/
	/*float y_temp = 0.0;
	for (int f = 0; f < F; f++) {
		y_temp += x[0][f] * w[0][f+1];
	}
	y_temp += w[0][0];
	printf("y_temp = %f", y_temp);
	printf("sizes of x = %d, y_1 = %d, w = %d, y = %d",
	sizeof(x)/sizeof(x[0]), sizeof(y_1), sizeof(w), sizeof(y));*/

	//printf("\nsizes of x = %d x %d\n",
		//sizeof(x) , sizeof(x[0]));

	printf("Y_f1 [0][0] = %f\n", y_1[0][0]);
	printf("Y_f1 [0][1] = %f\n", y_1[0][1]);
	printf("Y_f1 [0][2] = %f\n", y_1[0][2]);
	printf("Y_f1 [0][3] = %f\n", y_1[0][3]);
	printf("\nY_O [0][0] = %f\n", y_O[0][0]);
	printf("Y_O [0][1] = %f\n", y_O[0][1]);
	printf("Y_O [0][2] = %f\n", y_O[0][2]);
	printf("Y_O [0][3] = %f\n", y_O[0][3]);
	printf("Y_O [0][3] = %f\n", y_O[0][4]);
	printf("\nC [1][0] = %f\n", C[1][0]);
	printf("C [2][0] = %f\n", C[2][0]);
	printf("C [3][0] = %f\n", C[3][0]);
	printf("C [4][0] = %f\n", C[4][0]);
	printf("C [5][0] = %f\n", C[5][0]);
	printf("\n W_O_updt[1][0] = %f\n", wO_updt[1][0]);
	printf("W_O_updt[2][0] = %f\n", wO_updt[2][0]);
	printf("W_O_updt[3][0] = %f\n", wO_updt[3][0]);
	printf("W_O_updt[4][0] = %f\n", wO_updt[4][0]);
	printf("W_O_updt[5][0] = %f\n", wO_updt[5][0]);
	float sum =0.0;
	for (int i = 0; i < 10; i++) {
		sum += y_O[0][i];
	}
	printf("\nSum of elements of y_O = %f\n", sum);
	freeFloatMat(x, S, true);
	free(y);
	freeFloatMat(C, S, true);
	freeFloatMat(w1, F);
	freeFloatMat(wO, U1);
	freeFloatMat(y_1, S);
	freeFloatMat(y_O, U1);
	freeFloatMat(z_1, S);
	freeFloatMat(z_O, U1);
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
	float** mat;// = (float**)malloc(i * sizeof(float*));
	if (reg)
		mat = (float**)malloc(i * sizeof(float*));
	else
		cudaMalloc((void ***) &mat, i * sizeof(float*));
	//if (reg)
		//cudaHostRegister(mat, i * sizeof(float*), cudaHostRegisterMapped);
	if (mat == NULL) {
		return mat;
	}
	for (int ii = 0; ii < i; ii++) {
		//mat[ii] = (float*)malloc(j * sizeof(float));
		if (reg)
			mat[ii] = (float*)malloc(j * sizeof(float));
		else
			cudaMalloc((void **) &(mat[ii]), j * sizeof(float));
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
			cudaFree(mat[ii]);
	}
	if (reg)
		free(mat);
	else
		cudaFree(mat);
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

