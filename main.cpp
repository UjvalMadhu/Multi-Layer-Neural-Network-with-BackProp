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

#ifndef __COMMON_H__
#include "common.h"
#endif

// Constant Declaration
const int Features = 784;     //Number of Input Features
const int Samples  = 60000;   //Number of Samples
const int U1       = 10;     //Number of Units of Layer 1
const int U2       = 256;     //Number of Units of Layer 2
const int UL       = 10;      //Number of Units in Output Layer
const int EPOCHS   = 2;       //Number of Epochs
const float eta    = 0.005;   //Learning Rate

//========== M A I N   F U N C T I O N================

int main(int argc, char* argv[]) {
	
	// ======== Feature Extraction ============ //

	// Training Dataset
	FILE* train_set = fopen("mnist_train.csv", "r");
	if (train_set== NULL) {
		fprintf(stderr, "Missing training data\n");
		exit(1);
	}


	// y: Output Vector Memory Allocation
	float* y_num = (float*)malloc(Samples * sizeof(float)); // y = 60000x1 Matrix
	if (y_num == NULL) {
		fprintf(stderr, "Missing training data\n");
		exit(1);
	}


	// x: Input Vector Memory Allocation
	float** X = allocFloatMat(Samples, Features);                    // x = 60000x784 Matrix
	if (X == NULL) {
		exit(1);
	}


	// Populating the input and Output Matrices
	uint8_t tmp ;
	for (int i = 0; i < Samples; i++) {
		fscanf(train_set, "%hhu", &tmp);
		y_num[i] = (float)(tmp);
		for (int j = 0; j < Features; j++) {
			fscanf(train_set, ",%hhu", &tmp);
			/*if (i == 0) {
				printf("\n x[0][%u ] = %u \n", j, tmp);
			}*/
			X[i][j] = (float)(tmp/255.0);    // Normalizaing x 
		}
	}
	printf("\nX[%d] = ", 1);
	for (int i = 0; i < Features; i++) {
		printf("%f \t", X[1][i] * 255);
	}
	printf("\nY[%d] = %f \n\n", 1, y_num[1]);
	 //Print Input and Output Values
	/*for (int i = 0; i < F / 4; i++) {
		printf("\n x[0][% d] = %f", i, x[0][i]);
	}
	for (int i = 0; i < Samples/300; i++) {
		printf("\n y_num[% d] = %f", i, y_num[i]);
	}*/

	
	//======= CATEGORICAL ENCODING OF OUTPUT =======//

	float** Y = (float **)malloc(Samples * sizeof(float *));
	if (Y == NULL) {
		printf("Memory Allocation Error for Y");
		exit(1);
	}
	for (int ii = 0; ii < Samples; ii++) {
		Y[ii] = (float*)malloc(10 * sizeof(float));
		if (Y[ii] == NULL) {
			printf("Memory Allocation Error for Y[%u]",ii);
			exit(1);
		}
	}
	for (int i = 0; i < Samples; i++) {
		for (int j = 0; j < 10; j++) {
			Y[i][j] = (j == y_num[i]) ? 1 : 0;
		}
	}

	//for (int i = 0; i < 20; i++) {
	//	printf("\n y[% d] = ", i);
	//	for (int j = 0; j < 10; j++) {
	//		printf("%f ,", y[i][j]);
	//	}
	//	printf("\n", i);
	//}

	const int F = 10;
	const int S = 50;
	clock_t start, end;
	float tCPU;

	// Selcting a Batch of the training data;
	float** x = allocFloatMat(S,F);
	
	for (int i = 0; i < S; i++) {
		for (int j = 0; j < F; j++) {
			x[i][j] = X[i][j+155];
		}
	}

	for (int sam = 0; sam < 2; sam++) {
		printf("\nX[%d] = ", sam);
		for (int i = 0; i < F; i++) {
			printf("%f \t", X[sam][i+155]*255);
		}
	}

	float** y = allocFloatMat(S, UL);

	for (int i = 0; i < S; i++) {
		for (int j = 0; j < UL; j++) {
			y[i][j] = Y[i][j];
		}
	}


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

	float** C = allocFloatMat(S, 1);
	if (C == NULL) {
		printf("Memory Allocation Error for Cost C");
		exit(1);
	}
	
	// The Implementation will be using Batch Gradient Descent Optimization
	// 
	// 
	//**********************  C P U    I m p l e m e n t at i o n  *******************
	//
	// 
	// Iterating over the number of Epochs
	start = clock();
	for (int ep = 0; ep < EPOCHS; ep++) {
		// 
		// ============================= Forward Pass ==================================// 
		// 
		//=============================== Layer One ===================================//

		printf("\n****** Layer 1 Epoch: %d ******\n", ep);

		for (int sam = 0; sam < 2; sam++) {
			printf("\nx[%d] = ", sam);
			for (int i = 0; i < F; i++) {
				printf("%f \t", x[sam][i]);
			}
		}

		for (int sam = 0; sam < F+1; sam++) {
			printf("\n w1[%d] = ", sam);
			for (int i = 0; i < 3; i++) {
				printf("%f \t", w1[sam][i]);
			}
		}

		ForwardCPU(x, z_1, w1, S, F, U1);       // Forward Pass with Sigmoid activation
		SigmoidAct(z_1, y_1, S, U1);

		for (int sam = 0; sam < 2; sam++) {
			printf("\nZ_1[%d] = ", sam);
			for (int i = 0; i < U1; i++) {
				printf("%f \t", z_1[sam][i]);
			}
		}

		printf("\n\n");

		for (int sam = 0; sam < 2; sam++) {
			printf("\nY_1[%d] = ", sam);
			for (int i = 0; i < U1; i++) {
				printf("%f \t", y_1[sam][i]);
			}
		}

		printf("\n\n");
		printf("\n****** Output layer Epoch: %d ******\n", ep);
		//============================ Output Layer =====================================//

		// The output layer does not use the sigmoid activation function but
		// uses the Normalized Exponential Activation so that the total sum of
		// all Output values of the network for a given sample = 1 and individual
		// output units will have different probabilites
		// Our goal would be to have the correct unit give the highest probability of 1.

		ForwardCPU(y_1, z_O, wO, S, U1, UL);			// Forward Pass
		NormExp(z_O, y_O, S, UL);                       // Normalized Exponential Activation

		for (int sam = 0; sam < 2; sam++) {
			printf("\nZ_O[%d] = ", sam);
			for (int i = 0; i < UL; i++) {
				printf("%f", z_O[sam][i]);
			}
		}
		printf("\n\n");

		for (int sam = 0; sam < 2; sam++) {
			printf("\nY_O[%d] = ", sam);
			for (int i = 0; i < UL; i++) {
				printf("%f", y_O[sam][i]);
			}
		}

		//============================== Backpropagation =================================//
		// In this implementation, we use categorical cross entropy cost
		// The categorical cross entropy cost  = sum over all classes {y*ln(y_O)}

		CatCrEnt(y, y_O, C, S, UL);						// Categorical Cross Entropy Cost for all samples


		float cost = 0.0;
		for (int sam = 0; sam < S; sam++) {
			cost += C[sam][0];
		}
		cost = cost / S;
		//Printing Cost value


		printf("\n\n****** Epoch %d: Cost = %f *******\n\n", ep, cost);

		/*for (int sam = 0; sam < S; sam++) {
			printf("\nCost = %f", C[sam][0]);
		}
		printf("\n\n");
		for (int sam = 0; sam < S; sam++) {
			for(int i = 0; i < 10; i++){
			printf("\nY[%d][%d] = %f",sam,i ,y[sam][i]);
			}
		}
		printf("\n\n");
		for (int sam = 0; sam < S; sam++) {
			for (int i = 0; i < 10; i++) {
				printf("\nY_O[%d][%d] = %f", sam, i, y_O[sam][i]);
			}
		}*/

		// Since we are using BGD optimization we will be accumulating all the errors from all the 
		// training samples and averaging them in the w_updt matices in the backProp step.

		// 1. Backpropagation at the Output Layer
		backProp_O(y, y_O, z_O, y_1, wO_updt, S, 10, U1);

		// 2. Backpropagation at the first Layer
		backProp_H(x, y, y_O, y_1, wO, w1_updt, S, F, U1, 10);



		//=============================  Weight Updation ================================//

		// 1. w1: F x U1

		updateW(w1, w1_updt, F, U1, eta);

		// 2. w2: U1 x 10

		updateW(wO, wO_updt, U1, UL, eta);


		if (ep == 1 || ep == 100) {
			printf("\n Predicted Output at Epoch = %d\n", ep);
			for (int i = 0; i < 10; i++) {
				for (int j = 0; j < UL; j++){
					printf("  %f  ", y_O[i][j]);
				}
				printf("\n");
			}

			printf("\n Target Output \n");
			for (int i = 0; i < 10; i++) {
				for (int j = 0; j < UL; j++) {
					printf("  %f  ", y[i][j]);
				}
				printf("\n");
			}
		}

	}
	end = clock();
	tCPU = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC;

	printf("\n\n Total Time taken for %d EPOCHS = %f ms", EPOCHS, tCPU);

	//
	// Free Memory Allocation
	freeFloatMat(X, Samples);
	freeFloatMat(Y, Samples);
	freeFloatMat(x, S);
	free(y);
	freeFloatMat(w1, F);
	freeFloatMat(w1_updt, F);
	freeFloatMat(wO, U1);
	freeFloatMat(wO_updt, U1);
	freeFloatMat(y_1, S);
	freeFloatMat(y_O, U1);
	freeFloatMat(C, S);
	return 0;
}
//==================================================
//     F U N C T I O N      D E F I N I T I O N S
// =================================================

// allocFloatMat: Allocates memory for a Matrix
// i = Number of Rows
// j = Number of Columns
// Returns a Double Pointer Matrix

float** allocFloatMat(int i, int j) {
	float** mat = (float**)malloc(i * sizeof(float*));
	if (mat == NULL) {
		return mat;
	}
	for (int ii = 0; ii < i; ii++) {
		mat[ii] = (float*)malloc(j * sizeof(float));
		if (mat[ii] == NULL) {
			return (float**)mat[ii];
		}
	}
	return mat;
}

// freeFloatMat: Frees the memory allocated for a Matrix
// i = Number of Rows
// mat = Double pointer of matrix

void freeFloatMat(float** mat, int i) {
	for (int ii = 0; ii < i; ii++) {
		free(mat[ii]);
	}
	free(mat);
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

/*printf("\n\n\nW1[1][0] = %f\n", w1[1][0]);
		printf("W1[2][0] = %f\n", w1[2][0]);
		printf("W1[3][0] = %f\n", w1[3][0]);
		printf("W1[4][0] = %f\n", w1[4][0]);
		printf("W1[5][0] = %f\n", w1[5][0]);

		printf("\nW_1_updt[1][0] = %f\n", w1_updt[1][0]);
		printf("W_1_updt[2][0] = %f\n", w1_updt[2][0]);
		printf("W_1_updt[3][0] = %f\n", w1_updt[3][0]);
		printf("W_1_updt[4][0] = %f\n", w1_updt[4][0]);
		printf("W_1_updt[5][0] = %f\n", w1_updt[5][0]);

		for (int i = 0; i < F; i++) {
			for (int j = 0; j < U1; j++) {
				printf("\nw1[%d][%d] = %f", i, j, w1[i][j]);
			}
		}
		printf("\n\n");

		for (int i = 0; i < F; i++) {
			for (int j = 0; j < U1; j++) {
				printf("\nw1_updt[%d][%d] = %f", i, j, w1_updt[i][j]);
			}
		}*/

		/*printf("\nWeights W1 after updation\n");

				for (int i = 0; i < F; i++) {
					for (int j = 0; j < U1; j++) {
						printf("\nw1[%d][%d] = %f", i, j, w1[i][j]);
					}
				}
				printf("\n\n");

				for (int i = 0; i < F; i++) {
					for (int j = 0; j < U1; j++) {
						printf("\nw1_updt[%d][%d] = %f", i, j, w1_updt[i][j]);
					}
				}*/

		/*printf("\nW1[1][0] = %f\n", w1[1][0]);
		printf("W1[2][0] = %f\n", w1[2][0]);
		printf("W1[3][0] = %f\n", w1[3][0]);
		printf("W1[4][0] = %f\n", w1[4][0]);
		printf("W1[5][0] = %f\n", w1[5][0]);

						printf("\n\n\nwO[1][0] = %f\n", wO[1][0]);
		printf("wO[2][0] = %f\n", wO[2][0]);
		printf("wO[3][0] = %f\n", wO[3][0]);
		printf("wO[4][0] = %f\n", wO[4][0]);
		printf("wO[5][0] = %f\n", wO[5][0]);

		printf("\nW_O_updt[1][0] = %f\n", wO_updt[1][0]);
		printf("W_O_updt[2][0] = %f\n", wO_updt[2][0]);
		printf("W_O_updt[3][0] = %f\n", wO_updt[3][0]);
		printf("W_O_updt[4][0] = %f\n", wO_updt[4][0]);
		printf("W_O_updt[5][0] = %f\n", wO_updt[5][0]);

		printf("\n\n");
		printf("\n\n");
		printf("\n\n");

		for (int i = 0; i < F; i++) {
			for (int j = 0; j < U1; j++) {
				printf("\nw1[%d][%d] = %f", i, j, w1[i][j]);
			}
		}
		printf("\n\n");

		for (int i = 0; i < F; i++) {
			for (int j = 0; j < U1; j++) {
				printf("\nw1_updt[%d][%d] = %f", i, j, w1_updt[i][j]);
			}
		}
		printf("\nWeights W2 after updation\n");

		for (int i = 0; i < F; i++) {
			for (int j = 0; j < U1; j++) {
				printf("\nw1[%d][%d] = %f", i, j, w1[i][j]);
			}
		}
		printf("\n\n");

		for (int i = 0; i < F; i++) {
			for (int j = 0; j < U1; j++) {
				printf("\nw1_updt[%d][%d] = %f", i, j, w1_updt[i][j]);
			}
		}
		/*
		printf("\nWO[1][0] = %f\n", wO[1][0]);
		printf("wO[2][0] = %f\n", wO[2][0]);
		printf("wO[3][0] = %f\n", wO[3][0]);
		printf("wO[4][0] = %f\n", wO[4][0]);
		printf("wO[5][0] = %f\n", wO[5][0]);*/

	// Print Statements for Debugg
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
	/*
	printf("\nY_f1 [0][0] = %f\n", y_1[0][0]);
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

	printf("\nW_O_updt[1][0] = %f\n", wO_updt[1][0]);
	printf("W_O_updt[2][0] = %f\n", wO_updt[2][0]);
	printf("W_O_updt[3][0] = %f\n", wO_updt[3][0]);
	printf("W_O_updt[4][0] = %f\n", wO_updt[4][0]);
	printf("W_O_updt[5][0] = %f\n", wO_updt[5][0]);

	printf("\n W_1_updt[1][0] = %f\n", w1_updt[1][0]);
	printf("W_1_updt[2][0] = %f\n", w1_updt[2][0]);
	printf("W_1_updt[3][0] = %f\n", w1_updt[3][0]);
	printf("W_1_updt[4][0] = %f\n", w1_updt[4][0]);
	printf("W_1_updt[5][0] = %f\n", w1_updt[5][0]);

	float sum =0.0;
	for (int i = 0; i < 10; i++) {
		sum += y_O[0][i];
	}
	printf("\nSum of elements of y_O = %f\n", sum);
	*/