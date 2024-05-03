//============================================
//  CMPE 755 High Performance Architectures
//============================================
//
// Projet: Multi Layer Neural Network
// Header File
//--------------------------------------------
// Authors: Ujval Madhu, Evan Ruttenberg
//============================================
#ifndef __COMMON_H__
#define __COMMON_H__

/*
*  CPU Functions
* 
*/
// allocFloatMat: Allocates memory for a Matrix
// i = Number of Rows
// j = Number of Columns
// Returns a Double Pointer Matrix
float** allocFloatMat(int i, int j);


// freeFloatMat: Frees the memory allocated for a Matrix
// i = Number of Rows
// mat = Double pointer of matrix
void freeFloatMat(float** mat, int i);


// mac: Returns the  Multiply Accumulate Result of Two Vectors, for input and weight MAC
// x = input vector array 1
// w = weight vector array without bias term
// F = Number of features of input layer
float mac(float* x, float* w, int F);


// Sigmoid: Returns the sigmoid of the input value
// x = float input
float sigmoid(float x);


// forward: Returns the y = w[0] + sum(x*w)
// x = input vector
// w = weight vector with bias
// F = number of features of x
float forward(float* x, float* w, int F);


// WeightGen(U , F): Creates a weight matrix and assigns random values to it
// U = number of units in current layer
// F = Number of features of input layer
float** WeightGen(int U, int F);


// ForwardCPU : Performs one formward pass of the input
// X     = SxinF Input Matrix
// Y     = SxUnits Output Matrix
// W     = (inF+1) x Units Weight Matrix
// S     = Number of Input Samples
// inF   = Number of Input Features/units
// Units = Number of Output Units
void ForwardCPU(float** X, float** Y, float** W, int S, int inF, int Units);


// SigmoidAct(float** X, float ** Y, int S, int Units) : Applies Sigmoid Activation to the SxUnits input matrix Y
// X = Input matrix
// Y = Output Matrix
// S = number of Samples 
// U = Number of Units
void SigmoidAct(float** X, float** Y, int S, int Units);


// expSum(Y, Units): calculates the sum of natural exponential of all the Units of Y
float expSum(float* Y, int Units);


// NormExp(float**  X, float** Y,int S, int U): returns the normalized exponential of the matrix X
// X is a a matrix of size SxU
// Y is the output Matrix of size SxU
// U is the number of units in X
// S = number of input samples
void NormExp(float** X, float** Y, int S, int U);


// CatCrEnt(Y, Y_O, S, U): Calculates the Categorical cross entropy Cost 
// Y   = Actual Output Matrix (SxU)
// Y_O = Calculated Output Matrix (SxU)
// C   = Cost Matrix (Sx1)
// S   = Number of Samples
// U   = Number of Units
void CatCrEnt(uint8_t** Y, float** Y_O, float** C, int S, int Units);


//void backProp_O(uint8_t** Y, float** Y_O, float** Z_O, float** Y_1, float** WO_updt, int S, int UL, int UL_m1)
// Calculates the derivative of the cost w.r.t the weights for the Output layer, i.e the weight update value averaged across the given number of samples
// Y       = Actual Output Matrix (S x UL);             Y_O     = Calculated Output Matrix (S x UL)
// Z_O     = Output Matrix before activation (S x UL) ; Y_1     = Input to output layer (S x UL_m1)
// WO_Updt = Matrix for storing derivatives w.r.t each weight (UL_m1+1 x UL)
// S = Number of Samples, UL = Number of units in output layer, UL_m1 = Number of units in Input Layer
void backProp_O(uint8_t** Y, float** Y_O, float** Z_O, float** Y_1, float** WO_updt, int S, int UL, int UL_m1);

// Void backProp_H()
// Calculates the derivative of the cost w.r.t the weights for the Hidden layer
// Y    = Actual Output Matrix (S x UL);             Y_O     = Calculated Output Matrix (S x UL)
// Z_O     = Output Matrix before activation (S x UL) ; Y_1     = Input to output layer (S x UL_m1)      
void backProp_H(float** X, uint8_t** Y, float** Y_O, float** Y_1, float** WO, float** W1_updt, int S, int UL_m2, int UL_m1, int UL);


// updateW(float ** W, float ** W_updt, int U1, int U2, int eta)
// W      = Weight Matrix to be updated (U1 x U2)
// W_updt = Weight Update Value Matrix (U1 x U2)
// eta    = Learning rate

void updateW(float** W, float** W_updt, int U1, int U2, int eta);

#endif