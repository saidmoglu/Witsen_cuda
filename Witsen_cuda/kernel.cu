
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "assertp.h"

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <time.h>
#include <cassert>

using namespace std;

#define filename "C:\\Users\\mustafa\\Dropbox\\MATLAB\\witsenhausen\\cuda\\k02_test6.m"
#define stdoutname "C:\\Users\\mustafa\\Dropbox\\MATLAB\\witsenhausen\\cuda\\k02_test6_stdout.txt"

#define maxNum 3
#define nm 0.95

#define Nx 2000
#define Nx_block_size 16
#define Nn 200
#define Nn_chunk_size 500
#define kstdv 5
#define kstdvn 5
#define kdfinitial 0.1
#define kdfmult 2
#define kdfmid 1
#define Xstdv 5.0
#define Nstdv 1.0

#define PI 3.141592653589793

double kcst;
double gdescent[Nx];
double g[Nx];
double gnew[Nx];
double X[Nx];
double px[Nx];
double sum;
double n[Nn];
double pn[Nn];
double sumn, sumx;

//For CUDA. Change all 3.
#define CUDA_TYPE double //double or float
#define CUDA_EXP exp //exp or expf
#define CUDA_POW pow //pow or powf
#define CUDA_METHOD  1 // 1:original or 2:Nn blocks

CUDA_TYPE *d_g, *d_kcst, *result, *d_X, *d_px, *d_n, *d_pn;
CUDA_TYPE *t_g, *t_kcst, *r_t;
CUDA_TYPE *t_x, *t_px, *t_n, *t_pn;

FILE *p;

double gausspdf(double n, double stdv);

double calcDP(void);
double calcDPnew(void);
double calcDP_par(int method);
double calcDP_par1(void);
double calcDP_par2(void);
double optimizeGsn(void);
void calcDescent(void);
void calcDescent_par(int method);
void calcDescent_par1(void);
void calcDescent_par2(void);
void printValuesToFile(void);
void initialize(void);

void runNCR(int fileSave);
double runDescent(int fileSave);
void getFunctions(void);

void search_nearby_g(void);

int main()
{
	initialize();

	kcst = 0.2;

	//printValuesToFile();
	
	double limits_org[] = { -27.966666666, -20.47333333, -13.38, -6.61, 0, 6.61, 13.38, 20.47333333, 27.966666666, 1000 };
	double as[] = { -0.9510011175, -0.9837561062, -0.9665655379, -0.9656027297, -0.9646104054, -0.9646104054, -0.9656027297, -0.9665655379, -0.9837561062, -0.9510011175 };
	double bs[] = { -30.2981039500, -23.6608471030, -16.2352064261, -9.5649838731, -3.1565578463, 3.1565578463, 9.5649838731, 16.2352064261, 23.6608471030, 30.2981039500 };
	srand((int)time(0));
	double x_step = 2.0*kstdv*Xstdv / (double)(Nx - 1);
	for (int i = 0; i < Nx; ++i)
	{
		int j = 0;
		while (X[i]>limits_org[j]) j++;
		g[i] = (as[j])* X[i] + bs[j] +(double)rand() / RAND_MAX;
	}

	runDescent(0);
	//search_nearby_g();
	//runNCR(1);

	//clock_t start, end;
	//double elapsed;
	//start = clock();
	//double DP2 = calcDP_par(CUDA_METHOD); //calcDescent_par();
	//end = clock();
	//elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
	//printf("elapsed : %g   DP : %.12f\n", elapsed, DP2);

	//Clean Up
	cudaFree(d_g);
	cudaFree(d_X);
	cudaFree(d_px);
	cudaFree(d_n);
	cudaFree(d_pn);
	cudaFree(d_kcst);
	cudaFree(result);

	cudaError_t cudaStatus = cudaDeviceReset();
	assertp(cudaStatus == cudaSuccess, "cudaDeviceReset failed!");

	delete[] t_g;
	delete[] r_t;
	delete[] t_x;
	delete[] t_px;
	delete[] t_n;
	delete[] t_pn;
	delete t_kcst;

    return 0;
}

void search_nearby_g(void)
{
	//Set mine here
	double limits_org[] = { -27.966666666, -20.47333333, -13.38, -6.61, 0, 6.61, 13.38, 20.47333333, 27.966666666, 1000 };
	double as[] = { -0.9510011175, -0.9837561062, -0.9665655379, -0.9656027297, -0.9646104054, -0.9646104054, -0.9656027297, -0.9665655379, -0.9837561062, -0.9510011175 };
	double bs[] = { -30.2981039500, -23.6608471030, -16.2352064261, -9.5649838731, -3.1565578463, 3.1565578463, 9.5649838731, 16.2352064261, 23.6608471030, 30.2981039500 };
	
	for (int i = 0; i < Nx; ++i)
	{
		int j = 0;
		while (X[i]>limits_org[j]) j++;
		g[i] = (as[j])* X[i] + bs[j];
	}

	double minDP = runDescent(0);

	printf("minDP= %.12f\n", minDP);

	double x_step = 2.0*kstdv*Xstdv / (double)(Nx - 1);
	double lim_descent[4];
	for (int limi = 0; limi < 4; limi++)
	{
		cout << "limi " << limi << endl;
		limits_org[limi] += x_step;
		limits_org[8 - limi] -= x_step;

		for (int i = 0; i < Nx; ++i)
		{
			int j = 0;
			while (X[i] > limits_org[j]) j++;
			g[i] = (as[j]) * X[i] + bs[j];
		}

		double DPnew1 = runDescent(0);

		limits_org[limi] -= 2*x_step;
		limits_org[8 - limi] += 2*x_step;

		for (int i = 0; i < Nx; ++i)
		{
			int j = 0;
			while (X[i] > limits_org[j]) j++;
			g[i] = (as[j]) * X[i] + bs[j];
		}

		double DPnew2 = runDescent(0);

		lim_descent[limi] = (DPnew1 - DPnew2) / (2 * x_step);

		minDP = min(min(DPnew1, DPnew2), minDP);

		limits_org[limi] += x_step;
		limits_org[8 - limi] -= x_step;
	}

	for (int i = 0; i < 4; i++)
	{
		printf("lim_descent[i] = %e\n", lim_descent[i]);
	}

	
	printf("minDP= %.12f\n", minDP);
}

double runDescent(int fileSave){
	double DPold = calcDP_par(CUDA_METHOD);
	printf("Descent kcst=%g  DP = %.12f\n", kcst, DPold);
	double DPnew;
	while (1) {
		DPnew = optimizeGsn();
		if (DPnew < DPold - 1e-13) {
			printf("DP = %.12f\n", DPnew);
			fileSave == 1 ? printValuesToFile() : 0;
			DPold = DPnew;
		}
		else{
			printf("Final DP = %.12f\n", DPnew);
			fileSave == 1 ? printValuesToFile() : 0;
			return DPnew;
		}
	}
}

void runNCR(int fileSave){
	kcst = kcst / (pow(nm, maxNum));
	/*for (int i = 0; i<Nx; i++) {
		g[i] = 0.1 * X[i];
	}*/

	double DPold;
	double DPnew;
	printf("NCR\n");
	for (int i = 0; i<maxNum; i++) {
		kcst = kcst*nm;

		DPold = calcDP_par(CUDA_METHOD);
		printf("Initi at kcst=%g  DP = %g\n", kcst, DPold);

		int counter = 0;
		while (1) {
			counter++;
			//DPnew = optimizeGsn();
			calcDescent_par(CUDA_METHOD);
			for (int jjj = 0; jjj<Nx; jjj++) {
				g[jjj] = g[jjj] - gdescent[jjj];
			}
			DPnew = calcDP_par(CUDA_METHOD);
			//if (counter % 10 == 0) {
				if (DPold-0.000000001>DPnew){//fabs(DPold - DPnew) / DPold>1e-5) {
					DPold = DPnew;
				}
				else{
					printf("Final at kcst=%g  DP = %g\n", kcst, DPold);
					fileSave == 1 ? printValuesToFile() : 0;
					break;
				}
			//}
		}
	}

	printf("finished NCR\n");
}

//FUNCTIONS
double calcDP(void)
{
	double result = 0;
	for (int ix = 0; ix<Nx; ix++) {
		for (int in = 0; in<Nn; in++) {
			double temp1 = 0;
			double temp2 = 0;
			for (int ix2 = 0; ix2<Nx; ix2++) {
				double tm = gausspdf(g[ix] + X[ix] + n[in] - g[ix2] - X[ix2], Nstdv)*px[ix2];
				temp1 += (g[ix2] + X[ix2])*tm;
				temp2 += tm;
			}
			result += (pow(X[ix] + g[ix] - temp1 / temp2, 2) + kcst*kcst*g[ix] * g[ix])*px[ix] * pn[in];
		}
	}
	return result;
}

double calcDPnew(void)
{
	double gkeep[Nx];
	for (int i = 0; i<Nx; i++) {
		gkeep[i] = g[i];
		g[i] = gnew[i];
	}
	double res = calcDP_par(CUDA_METHOD);
	for (int i = 0; i<Nx; i++) {
		g[i] = gkeep[i];
	}
	return res;
}

//calculate descent direction
void calcDescent(void)
{
	for (int i = 0; i<Nx; i++) {
		gdescent[i] = 0;
		for (int e = 0; e<Nn; e++) {
			double temp1 = 0;
			double temp2 = 0;
			double temp1der = 0;
			double temp2der = 0;
			for (int ix2 = 0; ix2<Nx; ix2++) {
				double tm = gausspdf(g[i] + X[i] + n[e] - g[ix2] - X[ix2], Nstdv)*px[ix2];
				temp1 += (g[ix2] + X[ix2])*tm;
				temp2 += tm;
				temp1der += (g[ix2] + X[ix2])*tm*(-(g[i] + X[i] + n[e] - g[ix2] - X[ix2]));
				temp2der += tm*(-(g[i] + X[i] + n[e] - g[ix2] - X[ix2]));
			}
			gdescent[i] += ((X[i] + g[i] - temp1 / temp2)*(1 - (temp1der / temp2 - (temp2der / temp2)*(temp1 / temp2))) + kcst*kcst*g[i])*px[i] * pn[e];
		}
	}
}

double optimizeGsn(void){
	double DPactold;
	for (int i = 0; i < 1; i++)
	{
		double DPnew;
		DPactold = calcDP_par(CUDA_METHOD);
		calcDescent_par(CUDA_METHOD);
		double kdf = kdfinitial;
		double kdfhold = 0;
		while (1) {
			kdf = kdf*kdfmult;
			for (int jjj = 0; jjj < Nx; jjj++) {
				gnew[jjj] = g[jjj] - kdf*gdescent[jjj];
			}
			DPnew = calcDPnew();
			if (DPnew < DPactold) {
				DPactold = DPnew;
				kdfhold = kdf;
			}
			else if (kdf>kdfmid)
				break;
		}
		for (int jjj = 0; jjj < Nx; jjj++) {
			g[jjj] = g[jjj] - kdfhold*gdescent[jjj];
		}
	}
	return DPactold;
}

//initialization
void initialize(void)
{
	//Create X,Z,pxz,px,pz_x,Pkx
	for (int i = 0; i<Nx; i++) {
		X[i] = -kstdv*Xstdv + i*2.0*kstdv*Xstdv / (double)(Nx - 1);
	}

	sum = 0;
	for (int i = 0; i<Nx; i++) {
		px[i] = gausspdf(X[i], Xstdv);
		sum += px[i];
	}
	for (int i = 0; i<Nx; i++) {
		px[i] = px[i] / sum;
	}

	//Noise
	for (int i = 0; i<Nn; i++) {
		n[i] = -kstdvn*Nstdv + i*2.0*kstdvn*Nstdv / (double)(Nn - 1);
	}

	sum = 0;
	for (int i = 0; i<Nn; i++) {
		pn[i] = gausspdf(n[i], Nstdv);
		sum += pn[i];
	}
	for (int i = 0; i<Nn; i++) {
		pn[i] = pn[i] / sum;
	}

	//For CUDA
	t_g = new CUDA_TYPE[Nx];
	t_kcst = new CUDA_TYPE(kcst);
	r_t = new CUDA_TYPE[Nx*Nn];
	//r_t = new CUDA_TYPE[Nx];

	//Allocate space
	cudaError_t cudaStatus = cudaMalloc((void **)&d_g, Nx * sizeof(CUDA_TYPE));
	assertp(cudaStatus == cudaSuccess, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMalloc((void **)&d_kcst, sizeof(CUDA_TYPE));
	assertp(cudaStatus == cudaSuccess, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMalloc((void **)&result, Nx * Nn * sizeof(CUDA_TYPE));
	assertp(cudaStatus == cudaSuccess, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMalloc((void **)&d_X, Nx * sizeof(CUDA_TYPE));
	assertp(cudaStatus == cudaSuccess, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMalloc((void **)&d_px, Nx * sizeof(CUDA_TYPE));
	assertp(cudaStatus == cudaSuccess, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMalloc((void **)&d_n, Nn * sizeof(CUDA_TYPE));
	assertp(cudaStatus == cudaSuccess, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMalloc((void **)&d_pn, Nn * sizeof(CUDA_TYPE));
	assertp(cudaStatus == cudaSuccess, "cudaMalloc failed!", cudaGetErrorString(cudaStatus));

	//Copy fixed values
	t_x = new CUDA_TYPE[Nx];
	for (int i = 0; i < Nx; i++){
		t_x[i] = (CUDA_TYPE)X[i];
	}
	t_px = new CUDA_TYPE[Nx];
	for (int i = 0; i < Nx; i++){
		t_px[i] = (CUDA_TYPE)px[i];
	}
	t_n = new CUDA_TYPE[Nn];
	for (int i = 0; i < Nn; i++){
		t_n[i] = (CUDA_TYPE)n[i];
	}
	t_pn = new CUDA_TYPE[Nn];
	for (int i = 0; i < Nn; i++){
		t_pn[i] = (CUDA_TYPE)pn[i];
	}
	//t_n = new CUDA_TYPE[Nn_chunk_size];
	//t_pn = new CUDA_TYPE[Nn_chunk_size];

	cudaStatus = cudaMemcpy(d_X, t_x, Nx*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMemcpy(d_px, t_px, Nx*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMemcpy(d_n, t_n, Nn*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMemcpy(d_pn, t_pn, Nn*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
}

//print the important values to file
void printValuesToFile(void)
{
	double dp = calcDP_par(CUDA_METHOD);
	p = fopen(filename, "w");

	//Print G
	fprintf(p, "G = [");
	for (int i = 0; i<Nx; i++) {
		fprintf(p, "%4.10f  ", g[i]);
	}
	fprintf(p, "];\n");

	//Print X
	fprintf(p, "X = [");
	for (int i = 0; i<Nx; i++) {
		fprintf(p, "%4.10f  ", X[i]);
	}
	fprintf(p, "];\n");

	//Print stuff
	fprintf(p, "Nx=%d; Nn=%d; dp=%.12f;\n", Nx, Nn, dp);

	//Print n
	fprintf(p, "n = [");
	for (int i = 0; i<Nn; i++) {
		fprintf(p, "%4.10f  ", n[i]);
	}
	fprintf(p, "];\n");

	fclose(p);
}

//Noise pdf
double gausspdf(double n, double stdv)
{
	return 1 / (sqrt(2 * PI)*stdv) * exp(-(n*n) / (2 * stdv*stdv));
}

__global__ void calcDescentKernel1(CUDA_TYPE *d_g, CUDA_TYPE *d_X, CUDA_TYPE *d_px, CUDA_TYPE *d_n, CUDA_TYPE *d_pn, CUDA_TYPE *d_kcst, CUDA_TYPE *d_result){
	unsigned int i = blockIdx.x;
	unsigned int e = threadIdx.x;
	CUDA_TYPE temp1 = 0;
	CUDA_TYPE temp2 = 0;
	CUDA_TYPE temp1der = 0;
	CUDA_TYPE temp2der = 0;
	CUDA_TYPE y_temp1, t_temp1, c_temp1 = 0.0;
	CUDA_TYPE y_temp2, t_temp2, c_temp2 = 0.0;
	CUDA_TYPE y_temp1der, t_temp1der, c_temp1der = 0.0;
	CUDA_TYPE y_temp2der, t_temp2der, c_temp2der = 0.0;
	for (int ix2 = 0; ix2<Nx; ix2++) {
		CUDA_TYPE tm = CUDA_EXP(-(d_g[i] + d_X[i] + d_n[e] - d_g[ix2] - d_X[ix2]) * (d_g[i] + d_X[i] + d_n[e] - d_g[ix2] - d_X[ix2]) / (2.0*Nstdv*Nstdv))*d_px[ix2];
		y_temp1 = (d_g[ix2] + d_X[ix2]) * tm - c_temp1;
		t_temp1 = temp1 + y_temp1;
		c_temp1 = (t_temp1 - temp1) - y_temp1;
		temp1 = t_temp1;
		y_temp2 = tm - c_temp2;
		t_temp2 = temp2 + y_temp2;
		c_temp2 = (t_temp2 - temp2) - y_temp2;
		temp2 = t_temp2;
		y_temp1der = (d_g[ix2] + d_X[ix2]) * tm * (-(d_g[i] + d_X[i] + d_n[e] - d_g[ix2] - d_X[ix2])) - c_temp1der;
		t_temp1der = temp1der + y_temp1der;
		c_temp1der = (t_temp1der - temp1der) - y_temp1der;
		temp1der = t_temp1der;
		y_temp2der = tm * (-(d_g[i] + d_X[i] + d_n[e] - d_g[ix2] - d_X[ix2])) - c_temp2der;
		t_temp2der = temp2der + y_temp2der;
		c_temp2der = (t_temp2der - temp2der) - y_temp2der;
		temp2der = t_temp2der;
	}
	d_result[threadIdx.x + blockIdx.x * blockDim.x] = ((d_X[i] + d_g[i] - temp1 / temp2)*(1 - (temp1der / temp2 - (temp2der / temp2)*(temp1 / temp2))) + (*d_kcst) * (*d_kcst)*d_g[i])*d_px[i] * d_pn[e];
}

__global__ void calcDescentKernel2(CUDA_TYPE *d_g, CUDA_TYPE *d_X, CUDA_TYPE *d_px, CUDA_TYPE *d_n, CUDA_TYPE *d_pn, CUDA_TYPE *d_kcst, CUDA_TYPE *d_result){
	unsigned int i = blockIdx.x * Nx_block_size + threadIdx.x;
	CUDA_TYPE tmp = 0;
	for (int e = 0; e < Nn; e++) {
		CUDA_TYPE temp1 = 0;
		CUDA_TYPE temp2 = 0;
		CUDA_TYPE temp1der = 0;
		CUDA_TYPE temp2der = 0;
		for (int ix2 = 0; ix2 < Nx; ix2++) {
			CUDA_TYPE tm = CUDA_EXP(-(d_g[i] + d_X[i] + d_n[e] - d_g[ix2] - d_X[ix2]) * (d_g[i] + d_X[i] + d_n[e] - d_g[ix2] - d_X[ix2]) / (2.0*Nstdv*Nstdv))*d_px[ix2];
			temp1 += (d_g[ix2] + d_X[ix2])*tm;
			temp2 += tm;
			temp1der += (d_g[ix2] + d_X[ix2])*tm*(-(d_g[i] + d_X[i] + d_n[e] - d_g[ix2] - d_X[ix2]));
			temp2der += tm*(-(d_g[i] + d_X[i] + d_n[e] - d_g[ix2] - d_X[ix2]));
		}
		tmp += ((d_X[i] + d_g[i] - temp1 / temp2)*(1 - (temp1der / temp2 - (temp2der / temp2)*(temp1 / temp2))) + (*d_kcst) * (*d_kcst)*d_g[i])*d_px[i] * d_pn[e];
	}
	d_result[i] = tmp;
}

__global__ void calcDPKernel1(CUDA_TYPE *d_g, CUDA_TYPE *d_X, CUDA_TYPE *d_px, CUDA_TYPE *d_n, CUDA_TYPE *d_pn, CUDA_TYPE *d_kcst, CUDA_TYPE *d_result)
{
	unsigned int ix = blockIdx.x;
	unsigned int in = threadIdx.x;
	
	CUDA_TYPE temp1 = 0;
	CUDA_TYPE temp2 = 0;
	CUDA_TYPE y_temp1, t_temp1, c_temp1 = 0.0;
	CUDA_TYPE y_temp2, t_temp2, c_temp2 = 0.0;
	for (int ix2 = 0; ix2<Nx; ix2++) {
		CUDA_TYPE tm = CUDA_EXP(-(d_g[ix] + d_X[ix] + d_n[in] - d_g[ix2] - d_X[ix2]) * (d_g[ix] + d_X[ix] + d_n[in] - d_g[ix2] - d_X[ix2]) / (2.0*Nstdv*Nstdv))* d_px[ix2];
		y_temp1 = (d_g[ix2] + d_X[ix2]) * tm - c_temp1;
		t_temp1 = temp1 + y_temp1;
		c_temp1 = (t_temp1 - temp1) - y_temp1;
		temp1 = t_temp1;
		y_temp2 = tm - c_temp2;
		t_temp2 = temp2 + y_temp2;
		c_temp2 = (t_temp2 - temp2) - y_temp2;
		temp2 = t_temp2;
	}
	d_result[threadIdx.x + blockIdx.x * blockDim.x] = (CUDA_POW(d_X[ix] + d_g[ix] - temp1 / temp2, 2) + (*d_kcst) * (*d_kcst) * d_g[ix] * d_g[ix])*d_px[ix] * d_pn[in];
}

__global__ void calcDPKernel2(CUDA_TYPE *d_g, CUDA_TYPE *d_X, CUDA_TYPE *d_px, CUDA_TYPE *d_n, CUDA_TYPE *d_pn, CUDA_TYPE *d_kcst, CUDA_TYPE *d_result)
{
	unsigned int ix = blockIdx.x * Nx_block_size + threadIdx.x;
	CUDA_TYPE tmp = 0;
	for (unsigned int in = 0; in < Nn_chunk_size; in++){
		CUDA_TYPE temp1 = 0;
		CUDA_TYPE temp2 = 0;
		for (int ix2 = 0; ix2 < Nx; ix2++) {
			CUDA_TYPE tm = CUDA_EXP(-(d_g[ix] + d_X[ix] + d_n[in] - d_g[ix2] - d_X[ix2]) * (d_g[ix] + d_X[ix] + d_n[in] - d_g[ix2] - d_X[ix2]) / (2.0*Nstdv*Nstdv))* d_px[ix2];
			temp1 += (d_g[ix2] + d_X[ix2])*tm;
			temp2 += tm;
		}
		tmp += (CUDA_POW(d_X[ix] + d_g[ix] - temp1 / temp2, 2) + (*d_kcst) * (*d_kcst) * d_g[ix] * d_g[ix])*d_px[ix] * d_pn[in];
	}
	d_result[ix] = tmp;
}

void calcDescent_par(int method) {
	if (method == 1)
		calcDescent_par1();
	else if (method == 2)
		calcDescent_par2();
}

void calcDescent_par1(void)
{
	//Copy values to GPU
	for (int i = 0; i < Nx; i++){
		t_g[i] = (CUDA_TYPE)g[i];
	}
	*t_kcst = (CUDA_TYPE)kcst;
	cudaError_t cudaStatus = cudaMemcpy(d_g, t_g, Nx*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMemcpy(d_kcst, t_kcst, sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));

	calcDescentKernel1 << <Nx, Nn >> > (d_g, d_X, d_px, d_n, d_pn, d_kcst, result);

	cudaStatus = cudaGetLastError();
	assertp(cudaStatus == cudaSuccess, "addKernel launch failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaDeviceSynchronize();
	assertp(cudaStatus == cudaSuccess, "cudaDeviceSynchronize returned error!", cudaGetErrorString(cudaStatus));
	cudaMemcpy(r_t, result, Nx*Nn*sizeof(CUDA_TYPE), cudaMemcpyDeviceToHost);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));

	for (int i = 0; i < Nx; i++){
		gdescent[i] = 0;
		double y, t, c = 0.0;
		for (int e = 0; e < Nn; e++){
			y = (double)r_t[i * Nn + e] - c;
			t = gdescent[i] + y;
			c = (t - gdescent[i]) - y;
			gdescent[i] = t;
		}
	}
}

void calcDescent_par2(void)
{
	//Copy values to GPU
	for (int i = 0; i < Nx; i++){
		t_g[i] = (CUDA_TYPE)g[i];
	}
	*t_kcst = (CUDA_TYPE)kcst;
	cudaError_t cudaStatus = cudaMemcpy(d_g, t_g, Nx*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMemcpy(d_kcst, t_kcst, sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
	assertp(Nn % Nn_chunk_size == 0, "Nn not multiple of Nn_chunk_size");

	for (int i = 0; i < Nx; i++){
		gdescent[i] = 0;
	}

	for (int ichunk = 0; ichunk < Nn / Nn_chunk_size; ichunk++)
	{
		for (int i = 0; i < Nn_chunk_size; i++){
			t_n[i] = (CUDA_TYPE)n[i + ichunk*Nn_chunk_size];
		}
		for (int i = 0; i < Nn_chunk_size; i++){
			t_pn[i] = (CUDA_TYPE)pn[i + ichunk*Nn_chunk_size];
		}
		cudaStatus = cudaMemcpy(d_n, t_n, Nn_chunk_size*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
		assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
		cudaStatus = cudaMemcpy(d_pn, t_pn, Nn_chunk_size*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
		assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));		
		assertp(Nx % Nx_block_size == 0, "Nx not multiple of Nx_block_size");
		
		calcDescentKernel2 << <Nx / Nx_block_size, Nx_block_size >> > (d_g, d_X, d_px, d_n, d_pn, d_kcst, result);

		cudaStatus = cudaGetLastError();
		assertp(cudaStatus == cudaSuccess, "addKernel launch failed!", cudaGetErrorString(cudaStatus));
		cudaStatus = cudaDeviceSynchronize();
		assertp(cudaStatus == cudaSuccess, "cudaDeviceSynchronize returned error!", cudaGetErrorString(cudaStatus));
		cudaMemcpy(r_t, result, Nx*sizeof(CUDA_TYPE), cudaMemcpyDeviceToHost);
		assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));

		for (int i = 0; i < Nx; i++){
			gdescent[i] += (double)r_t[i];
		}
		
	}
}

double calcDP_par(int method) {
	if (method == 1)
		return calcDP_par1();
	else if (method == 2)
		return calcDP_par2();
	return 0;
}

double calcDP_par1(void)
{
	//Copy values to GPU
	for (int i = 0; i < Nx; i++){
		t_g[i] = (CUDA_TYPE)g[i];
	}
	*t_kcst = (CUDA_TYPE)kcst;
	cudaError_t cudaStatus = cudaMemcpy(d_g, t_g, Nx*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMemcpy(d_kcst, t_kcst, sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));

	calcDPKernel1 <<<Nx, Nn>>> (d_g, d_X, d_px, d_n, d_pn, d_kcst, result);

	cudaStatus = cudaGetLastError();
	assertp(cudaStatus == cudaSuccess, "addKernel launch failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaDeviceSynchronize();
	assertp(cudaStatus == cudaSuccess, "cudaDeviceSynchronize returned error!", cudaGetErrorString(cudaStatus));
	cudaMemcpy(r_t, result, Nx*Nn*sizeof(CUDA_TYPE), cudaMemcpyDeviceToHost);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));

	double thisDP = 0;

	double y, t, c = 0.0;
	for (int i = 0; i < Nx*Nn; i++){
		//thisDP += (double)r_t[i];
		y = (double)r_t[i] - c;
		t = thisDP + y;
		c = (t - thisDP) - y;
		thisDP = t;
	}

	return thisDP;
}

double calcDP_par2(void)
{
	//Copy values to GPU
	for (int i = 0; i < Nx; i++){
		t_g[i] = (CUDA_TYPE)g[i];
	}
	*t_kcst = (CUDA_TYPE)kcst;
	cudaError_t cudaStatus = cudaMemcpy(d_g, t_g, Nx*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaMemcpy(d_kcst, t_kcst, sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
	assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));

	assertp(Nn % Nn_chunk_size == 0, "Nn not multiple of Nn_chunk_size");

	double thisDP = 0;
	for (int ichunk = 0; ichunk < Nn / Nn_chunk_size; ichunk++)
	{
		for (int i = 0; i < Nn_chunk_size; i++){
			t_n[i] = (CUDA_TYPE)n[i + ichunk*Nn_chunk_size];
		}
		for (int i = 0; i < Nn_chunk_size; i++){
			t_pn[i] = (CUDA_TYPE)pn[i + ichunk*Nn_chunk_size];
		}
		cudaStatus = cudaMemcpy(d_n, t_n, Nn_chunk_size*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
		assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
		cudaStatus = cudaMemcpy(d_pn, t_pn, Nn_chunk_size*sizeof(CUDA_TYPE), cudaMemcpyHostToDevice);
		assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));
		assertp(Nx % Nx_block_size == 0, "Nx not multiple of Nx_block_size");
		
		calcDPKernel2 << <Nx / Nx_block_size, Nx_block_size >> > (d_g, d_X, d_px, d_n, d_pn, d_kcst, result);

		cudaStatus = cudaGetLastError();
		assertp(cudaStatus == cudaSuccess, "addKernel launch failed!", cudaGetErrorString(cudaStatus));
		cudaStatus = cudaDeviceSynchronize();
		assertp(cudaStatus == cudaSuccess, "cudaDeviceSynchronize returned error!", cudaGetErrorString(cudaStatus));
		cudaMemcpy(r_t, result, Nx*sizeof(CUDA_TYPE), cudaMemcpyDeviceToHost);
		assertp(cudaStatus == cudaSuccess, "cudaMemcpy failed!", cudaGetErrorString(cudaStatus));

		double thisDP_chunk = 0;
		double y, t, c = 0.0;
		for (int i = 0; i < Nx; i++) {
			//thisDP += (double)r_t[i];
			y = (double)r_t[i] - c;
			t = thisDP_chunk + y;
			c = (t - thisDP_chunk) - y;
			thisDP_chunk = t;
		}

		thisDP += thisDP_chunk;
	}

	return thisDP;
}

void getFunctions(void)
{
	static const double G[] = { 0.93305555, 0.92998131, 0.92690708, 0.92383284, 0.92075860, 0.91768436, 0.91461012, 0.91153589, 0.90846165, 0.90538741, 0.90231317, 0.89923893, 0.89616470, 0.89309046, 0.89001622, 0.88694198,
		0.88386775, 0.88079351, 0.87771927, 0.87464503, 0.87157079, 0.86849656, 0.86542232, 0.86234808, 0.85927384, 0.85619961, 0.85312537, 0.85005113, 0.84697689, 0.84390265, 0.84082842, 0.83775418,
		0.83467994, 0.83160570, 0.82853146, 0.82545723, 0.82238299, 0.81930875, 0.81623451, 0.81316028, 0.81008604, 0.80701180, 0.80393756, 0.80086332, 0.79778909, 0.79471485, 0.79164061, 0.78856637,
		0.78549214, 0.78241790, 0.77934366, 0.77626942, 0.77319518, 0.77012095, 0.76704671, 0.76397247, 0.76089823, 0.75782399, 0.75474976, 0.75167552, 0.74860128, 0.74552704, 0.74245281, 0.73937857,
		0.73630433, 0.73323009, 0.73015585, 0.72708162, 0.72400738, 0.72093314, 0.71785890, 0.71478466, 0.71171043, 0.70863619, 0.70556195, 0.70248771, 0.69941348, 0.69633924, 0.69326500, 0.69019076,
		0.68711652, 0.68404229, 0.68096805, 0.67789381, 0.67481957, 0.67174534, 0.66867110, 0.66559686, 0.66252262, 0.65944838, 0.65637415, 0.65329991, 0.65022567, 0.64715143, 0.64407719, 0.64100296,
		0.63792872, 0.63485448, 0.63178024, 0.62870601, 0.62563177, 0.62255753, 0.61948329, 0.61640905, 0.61333482, 0.61026058, 0.60718634, 0.60411210, 0.60103786, 0.59796363, 0.59488939, 0.59181515,
		0.58874091, 0.58566668, 0.58259244, 0.57951820, 0.57644396, 0.57336972, 0.57029549, 0.56722125, 0.56414701, 0.56107277, 0.55799854, 0.55492430, 0.55185006, 0.54877582, 0.54570158, 0.54262735,
		0.53955311, 0.53647887, 0.53340463, 0.53033039, 0.52725616, 0.52418192, 0.52110768, 0.51803344, 0.51495921, 0.51188497, 0.50881073, 0.50573649, 0.50266225, 0.49958802, 0.49651378, 0.49343954,
		0.49036530, 0.48729107, 0.48421683, 0.48114259, 0.47806835, 0.47499411, 0.47191988, 0.46884564, 0.46577140, 0.46269716, 0.45962292, 0.45654869, 0.45347445, 0.45040021, 0.44732597, 0.44425174,
		0.44117750, 0.43810326, 0.43502902, 0.43195478, 0.42888055, 0.42580631, 0.42273207, 0.41965783, 0.41658359, 0.41350936, 0.41043512, 0.40736088, 0.40428664, 0.40121241, 0.39813817, 0.39506393,
		0.39198969, 0.38891545, 0.38584122, 0.38276698, 0.37969274, 0.37661850, 0.37354427, 0.37047003, 0.36739579, 0.36432155, 0.36124731, 0.35817308, 0.35509884, 0.35202460, 0.34895036, 0.34587612,
		0.34280189, 0.33972765, 0.33665341, 0.33357917, 0.33050494, 0.32743070, 0.32435646, 0.32128222, 0.31820798, 0.31513375, 0.31205951, 0.30898527, 0.30591103, 0.30283680, 0.29976256, 0.29668832,
		0.29361408, 0.29053984, 0.28746561, 0.28439137, 0.28131713, 0.27824289, 0.27516865, 0.27209442, 0.26902018, 0.26594594, 0.26287170, 0.25979747, 0.25672323, 0.25364899, 0.25057475, 0.24750051,
		0.24442628, 0.24135204, 0.23827780, 0.23520356, 0.23212932, 0.22905509, 0.22598085, 0.22290661, 0.21983237, 0.21675814, 0.21368390, 0.21060966, 0.20753542, 0.20446118, 0.20138695, 0.19831271,
		0.19523847, 0.19216423, 0.18909000, 0.18601576, 0.18294152, 0.17986728, 0.17679304, 0.17371881, 0.17064457, 0.16757033, 0.16449609, 0.16142185, 0.15834762, 0.15527338, 0.15219914, 0.14912490,
		0.14605067, 0.14297643, 0.13990219, 0.13682795, 0.13375371, 0.13067948, 0.12760524, 0.12453100, 0.12145676, 0.11838252, 0.11530829, 0.11223405, 0.10915981, 0.10608557, 0.10301134, 0.09993710,
		0.09686286, 0.09378862, 0.09071438, 0.08764015, 0.08456591, 0.08149167, 0.07841743, 0.07534320, 0.07226896, 0.06919472, 0.06612048, 0.06304624, 0.05997201, 0.05689777, 0.05382353, 0.05074929,
		0.04767505, 0.04460082, 0.04152658, 0.03845234, 0.03537810, 0.03230387, 0.02922963, 0.02615539, 0.02308115, 0.02000691, 0.01693268, 0.01385844, 0.01078420, 0.00770996, 0.00463573, 0.00156149, -
		0.00151275, -0.00458699, -0.00766123, -0.01073546, -0.01380970, -0.01688394, -0.01995818, -0.02303242, -0.02610665, -0.02918089, -0.03225513, -0.03532937, -0.03840360, -0.04147784, -0.04455208, -
		0.04762632, -0.05070056, -0.05377479, -0.05684903, -0.05992327, -0.06299751, -0.06607175, -0.06914598, -0.07222022, -0.07529446, -0.07836870, -0.08144293, -0.08451717, -0.08759141, -0.09066565, -
		0.09373989, -0.09681412, -0.09988836, -0.10296260, -0.10603684, -0.10911107, -0.11218531, -0.11525955, -0.11833379, -0.12140803, -0.12448226, -0.12755650, -0.13063074, -0.13370498, -0.13677922, -
		0.13985345, -0.14292769, -0.14600193, -0.14907617, -0.15215040, -0.15522464, -0.15829888, -0.16137312, -0.16444736, -0.16752159, -0.17059583, -0.17367007, -0.17674431, -0.17981855, -0.18289278, -
		0.18596702, -0.18904126, -0.19211550, -0.19518973, -0.19826397, -0.20133821, -0.20441245, -0.20748669, -0.21056092, -0.21363516, -0.21670940, -0.21978364, -0.22285787, -0.22593211, -0.22900635, -
		0.23208059, -0.23515483, -0.23822906, -0.24130330, -0.24437754, -0.24745178, -0.25052602, -0.25360025, -0.25667449, -0.25974873, -0.26282297, -0.26589720, -0.26897144, -0.27204568, -0.27511992, -
		0.27819416, -0.28126839, -0.28434263, -0.28741687, -0.29049111, -0.29356534, -0.29663958, -0.29971382, -0.30278806, -0.30586230, -0.30893653, -0.31201077, -0.31508501, -0.31815925, -0.32123349, -
		0.32430772, -0.32738196, -0.33045620, -0.33353044, -0.33660467, -0.33967891, -0.34275315, -0.34582739, -0.34890163, -0.35197586, -0.35505010, -0.35812434, -0.36119858, -0.36427282, -0.36734705, -
		0.37042129, -0.37349553, -0.37656977, -0.37964400, -0.38271824, -0.38579248, -0.38886672, -0.39194096, -0.39501519, -0.39808943, -0.40116367, -0.40423791, -0.40731214, -0.41038638, -0.41346062, -
		0.41653486, -0.41960910, -0.42268333, -0.42575757, -0.42883181, -0.43190605, -0.43498029, -0.43805452, -0.44112876, -0.44420300, -0.44727724, -0.45035147, -0.45342571, -0.45649995, -0.45957419, -
		0.46264843, -0.46572266, -0.46879690, -0.47187114, -0.47494538, -0.47801962, -0.48109385, -0.48416809, -0.48724233, -0.49031657, -0.49339080, -0.49646504, -0.49953928, -0.50261352, -0.50568776, -
		0.50876199, -0.51183623, -0.51491047, -0.51798471, -0.52105894, -0.52413318, -0.52720742, -0.53028166, -0.53335590, -0.53643013, -0.53950437, -0.54257861, -0.54565285, -0.54872709, -0.55180132, -
		0.55487556, -0.55794980, -0.56102404, -0.56409827, -0.56717251, -0.57024675, -0.57332099, -0.57639523, -0.57946946, -0.58254370, -0.58561794, -0.58869218, -0.59176641, -0.59484065, -0.59791489, -
		0.60098913, -0.60406337, -0.60713760, -0.61021184, -0.61328608, -0.61636032, -0.61943456, -0.62250879, -0.62558303, -0.62865727, -0.63173151, -0.63480574, -0.63787998, -0.64095422, -0.64402846, -
		0.64710270, -0.65017693, -0.65325117, -0.65632541, -0.65939965, -0.66247389, -0.66554812, -0.66862236, -0.67169660, -0.67477084, -0.67784507, -0.68091931, -0.68399355, -0.68706779, -0.69014203, -
		0.69321626, -0.69629050, -0.69936474, -0.70243898, -0.70551321, -0.70858745, -0.71166169, -0.71473593, -0.71781017, -0.72088440, -0.72395864, -0.72703288, -0.73010712, -0.73318136, -0.73625559, -
		0.73932983, -0.74240407, -0.74547831, -0.74855254, -0.75162678, -0.75470102, -0.75777526, -0.76084950, -0.76392373, -0.76699797, -0.77007221, -0.77314645, -0.77622068, -0.77929492, -0.78236916, -
		0.78544340, -0.78851764, -0.79159187, -0.79466611, -0.79774035, -0.80081459, -0.80388883, -0.80696306, -0.81003730, -0.81311154, -0.81618578, -0.81926001, -0.82233425, -0.82540849, -0.82848273, -
		0.83155697, -0.83463120, -0.83770544, -0.84077968, -0.84385392, -0.84692816, -0.85000239, -0.85307663, -0.85615087, -0.85922511, -0.86229934, -0.86537358, -0.86844782, -0.87152206, -0.87459630, -
		0.87767053, -0.88074477, -0.88381901, -0.88689325, -0.88996748, -0.89304172, -0.89611596, -0.89919020, -0.90226444, -0.90533867, -0.90841291, -0.91148715, -0.91456139, -0.91763563, -0.92070986, -
		0.92378410, -0.92685834, -0.92993258, -0.93300681, -0.93608105, -0.93915529, -0.94222953, -0.94530377, -0.94837800, -0.95145224, -0.95452648, -0.95760072, -0.96067496, -0.96374919, -0.96682343, -
		0.96989767, -0.97297191, -0.97604614, -0.97912038, -0.98219462, -0.98526886, -0.98834310, -0.99141733, -0.99449157, -0.99756581, -1.00064005, -1.00371428, -1.00678852, -1.00986276, -1.01293700, -
		1.01601124, -1.01908547, -1.02215971, -1.02523395, -1.02830819, -1.03138243, -1.03445666, -1.03753090, -1.04060514, -1.04367938, -1.04675361, -1.04982785, -1.05290209, -1.05597633, -1.05905057, -
		1.06212480, -1.06519904, -1.06827328, -1.07134752, -1.07442175, -1.07749599, -1.08057023, -1.08364447, -1.08671871, -1.08979294, -1.09286718, -1.09594142, -1.09901566, -1.10208990, -1.10516413, -
		1.10823837, -1.11131261, -1.11438685, -1.11746108, -1.12053532, -1.12360956, -1.12668380, -1.12975804, -1.13283227, -1.13590651, -1.13898075, -1.14205499, -1.14512923, -1.14820346, -1.15127770, -
		1.15435194, -1.15742618, -1.16050041, -1.16357465, -1.16664889, -1.16972313, -1.17279737, -1.17587160, -1.17894584, -1.18202008, -1.18509432, -1.18816855, -1.19124279, -1.19431703, -1.19739127, -
		1.20046551, -1.20353974, -1.20661398, -1.20968822, -1.21276246, -1.21583670, -1.21891093, -1.22198517, -1.22505941, -1.22813365, -1.23120788, -1.23428212, -1.23735636, -1.24043060, -1.24350484, -
		1.24657907, -1.24965331, -1.25272755, -1.25580179, -1.25887602, -1.26195026, -1.26502450, -1.26809874, -1.27117298, -1.27424721, -1.27732145, -1.28039569, -1.28346993, -1.28654417, -1.28961840, -
		1.29269264, -1.29576688, -1.29884112, -1.30191535, -1.30498959, -1.30806383, -1.31113807, -1.31421231, -1.31728654, -1.32036078, -1.32343502, -1.32650926, -1.32958349, -1.33265773, -1.33573197, -
		1.33880621, -1.34188045, -1.34495468, -1.34802892, -1.35110316, -1.35417740, -1.35725164, -1.36032587, -1.36340011, -1.36647435, -1.36954859, -1.37262282, -1.37569706, -1.37877130, -1.38184554, -
		1.38491978, -1.38799401, -1.39106825, -1.39414249, -1.39721673, -1.40029097, -1.40336520, -1.40643944, -1.40951368, -1.41258792, -1.41566215, -1.41873639, -1.42181063, -1.42488487, -1.42795911, -
		1.43103334, -1.43410758, -1.43718182, -1.44025606, -1.44333029, -1.44640453, -1.44947877, -1.45255301, -1.45562725, -1.45870148, -1.46177572, -1.46484996, -1.46792420, -1.47099844, -1.47407267, -
		1.47714691, -1.48022115, -1.48329539, -1.48636962, -1.48944386, -1.49251810, -1.49559234, -1.49866658, -1.50174081, -1.50481505, -1.50788929, -1.51096353, -1.51403776, -1.51711200, -1.52018624, -
		1.52326048, -1.52633472, -1.52940895, -1.53248319, -1.53555743, -1.53863167, -1.54170591, -1.54478014, -1.54785438, -1.55092862, -1.55400286, -1.55707709, -1.56015133, -1.56322557, -1.56629981, -
		1.56937405, -1.57244828, -1.57552252, -1.57859676, -1.58167100, -1.58474523, -1.58781947, -1.59089371, -1.59396795, -1.59704219, -1.60011642, -1.60319066, -1.60626490, -1.60933914, -1.61241338, -
		1.61548761, -1.61856185, -1.62163609, -1.62471033, -1.62778456, -1.63085880, -1.63393304, -1.63700728, -1.64008152, -1.64315575, -1.64622999, -1.64930423, -1.65237847, -1.65545270, -1.65852694, -
		1.66160118, -1.66467542, -1.66774966, -1.67082389, -1.67389813, -1.67697237, -1.68004661, -1.68312085, -1.68619508, -1.68926932, -1.69234356, -1.69541780, -1.69849203, -1.70156627, -1.70464051, -
		1.70771475, -1.71078899, -1.71386322, -1.71693746, -1.72001170, -1.72308594, -1.72616017, -1.72923441, -1.73230865, -1.73538289, -1.73845713, -1.74153136, -1.74460560, -1.74767984, -1.75075408, -
		1.75382832, -1.75690255, -1.75997679, -1.76305103, -1.76612527, -1.76919950, -1.77227374, -1.77534798, -1.77842222, -1.78149646, -1.78457069, -1.78764493, -1.79071917, -1.79379341, -1.79686764, -
		1.79994188, -1.80301612, -1.80609036, -1.80916460, -1.81223883, -1.81531307, -1.81838731, -1.82146155, -1.82453579, -1.82761002, -1.83068426, -1.83375850, -1.83683274, -1.83990697, -1.84298121, -
		1.84605545, -1.84912969, -1.85220393, -1.85527816, -1.85835240, -1.86142664, -1.86450088, -1.86757511, -1.87064935, -1.87372359, -1.87679783, -1.87987207, -1.88294630, -1.88602054, -1.88909478, -
		1.89216902, -1.89524326, -1.89831749, -1.90139173, -1.90446597, -1.90754021, -1.91061444, -1.91368868, -1.91676292, -1.91983716, -1.92291140, -1.92598563, -1.92905987, -1.93213411, -1.93520835, -
		1.93828258, -1.94135682, -1.94443106, -1.94750530, -1.95057954, -1.95365377, -1.95672801, -1.95980225, -1.96287649, -1.96595073, -1.96902496, -1.97209920, -1.97517344, -1.97824768, -1.98132191, -
		1.98439615, -1.98747039, -1.99054463, -1.99361887, -1.99669310, -1.99976734, -2.00284158, -2.00591582, -2.00899005, -2.01206429, -2.01513853, -2.01821277, -2.02128701, -2.02436124, -2.02743548, -
		2.03050972, -2.03358396, -2.03665820, -2.03973243, -2.04280667, -2.04588091, -2.04895515, -2.05202938, -2.05510362, -2.05817786, -2.06125210, -2.06432634, -2.06740057, -2.07047481, -2.07354905, -
		2.07662329, -2.07969752, -2.08277176, -2.08584600, -2.08892024, -2.09199448, -2.09506871, -2.09814295, -2.10121719, -2.10429143, -2.10736567, -2.11043990, -2.11351414, -2.11658838, -2.11966262, -
		2.12273685, -2.12581109, -2.12888533, -2.13195957, -2.13503381, -2.13810804, -2.14118228, -2.14425652, -2.14733076, -2.15040499, -2.15347923, -2.15655347, -2.15962771, -2.16270195, -2.16577618, -
		2.16885042, -2.17192466, -2.17499890, -2.17807313, -2.18114737, -2.18422161, -2.18729585, -2.19037009, -2.19344432, -2.19651856, -2.19959280, -2.20266704, -2.20574128, -2.20881551, -2.21188975, -
		2.21496399, -2.21803823, -2.22111246, -2.22418670, -2.22726094, -2.23033518, -2.23340942, -2.23648365, -2.23955789, -2.24263213, -2.24570637, -2.24878060, -2.25185484, -2.25492908, -2.25800332, -
		2.26107756, -2.26415179, -2.26722603, -2.27030027, -2.27337451, -2.27644875, -2.27952298, -2.28259722, -2.28567146, -2.28874570, -2.29181993, -2.29489417, -2.29796841, -2.30104265, -2.30411689, -
		2.30719112, -2.31026536, -2.31333960, -2.31641384, -2.31948807, -2.32256231, -2.32563655, -2.32871079, -2.33178503, -2.33485926, -2.33793350, -2.34100774, -2.34408198, -2.34715621, -2.35023045, -
		2.35330469, -2.35637893, -2.35945317, -2.36252740, -2.36560164, -2.36867588, -2.37175012, -2.37482436, -2.37789859, -2.38097283, -2.38404707, -2.38712131, -2.39019554, -2.39326978, -2.39634402, -
		2.39941826, -2.40249250, -2.40556673, -2.40864097, -2.41171521, -2.41478945, -2.41786368, -2.42093792, -2.42401216, -2.42708640, -2.43016064, -2.43323487, -2.43630911, -2.43938335, -2.44245759, -
		2.44553182, -2.44860606, -2.45168030, -2.45475454, -2.45782878, -2.46090301, -2.46397725, -2.46705149, -2.47012573, -2.47319997, -2.47627420, -2.47934844, -2.48242268, -2.48549692, -2.48857115, -
		2.49164539, -2.49471963, -2.49779387, -2.50086811, -2.50394234, -2.50701658, -2.51009082, -2.51316506, -2.51623929, -2.51931353, -2.52238777, -2.52546201, -2.52853625, -2.53161048, -2.53468472, -
		2.53775896, -2.54083320, -2.54390743, -2.54698167, -2.55005591, -2.55313015, -2.55620439, -2.55927862, -2.56235286, -2.56542710, -2.56850134, -2.57157557, -2.57464981, -2.57772405, -2.58079829, -
		2.58387253, -2.58694676, -2.59002100, -2.59309524, -2.59616948, -2.59924372, -2.60231795, -2.60539219, -2.60846643, -2.61154067, -2.61461490, -2.61768914, -2.62076338, -2.62383762, -2.62691186, -
		2.62998609, -2.63306033, -2.63613457, -2.63920881, -2.64228304, -2.64535728, -2.64843152, -2.65150576, -2.65458000, -2.65765423, -2.66072847, -2.66380271, -2.66687695, -2.66995118, -2.67302542, -
		2.67609966, -2.67917390, -2.68224814, -2.68532237, -2.68839661, -2.69147085, -2.69454509, -2.69761932, -2.70069356, -2.70376780, -2.70684204, -2.70991628, -2.71299051, -2.71606475, -2.71913899, -
		2.72221323, -2.72528746, -2.72836170, -2.73143594, -2.73451018, -2.73758442, -2.74065865, -2.74373289, -2.74680713, -2.74988137, -2.75295560, -2.75602984, -2.75910408, -2.76217832, -2.76525256, -
		2.76832679, -2.77140103, -2.77447527, -2.77754951, -2.78062375, -2.78369798, -2.78677222, -2.78984646, -2.79292070, -2.79599493, -2.79906917, -2.80214341, -2.80521765, -2.80829189, -2.81136612, -
		2.81444036, -2.81751460, -2.82058884, -2.82366307, -2.82673731, -2.82981155, -2.83288579, -2.83596003, -2.83903426, -2.84210850, -2.84518274, -2.84825698, -2.85133121, -2.85440545, -2.85747969, -
		2.86055393, -2.86362817, -2.86670240, -2.86977664, -2.87285088, -2.87592512, -2.87899935, -2.88207359, -2.88514783, -2.88822207, -2.89129631, -2.89437054, -2.89744478, -2.90051902, -2.90359326, -
		2.90666749, -2.90974173, -2.91281597, -2.91589021, -2.91896445, -2.92203868, -2.92511292, -2.92818716, -2.93126140, -2.93433563, -2.93740987, -2.94048411, -2.94355835, -2.94663259, -2.94970682, -
		2.95278106, -2.95585530, -2.95892954, -2.96200377, -2.96507801, -2.96815225, -2.97122649, -2.97430073, -2.97737496, -2.98044920, -2.98352344, -2.98659768, -2.98967191, -2.99274615, -2.99582039, -
		2.99889463, -3.00196887, -3.00504310, -3.00811734, -3.01119158, -3.01426582, -3.01734005, -3.02041429, -3.02348853, -3.02656277, -3.02963701, -3.03271124, -3.03578548, -3.03885972, -3.04193396, -
		3.04500819, -3.04808243, -3.05115667, -3.05423091, -3.05730515, -3.06037938, -3.06345362, -3.06652786, -3.06960210, -3.07267633, -3.07575057, -3.07882481, -3.08189905, -3.08497329, -3.08804752, -
		3.09112176, -3.09419600, -3.09727024, -3.10034447, -3.10341871, -3.10649295, -3.10956719, -3.11264143, -3.11571566, -3.11878990, -3.12186414, -3.12493838, -3.12801261, -3.13108685, -3.13416109, -
		3.13723533, -3.14030957, -3.14338380, -3.14645804, -3.14953228, -3.15260652, -3.15568075, -3.15875499, -3.16182923, -3.16490347, -3.16797771, -3.17105194, -3.17412618, -3.17720042, -3.18027466, -
		3.18334889, -3.18642313, -3.18949737, -3.19257161, -3.19564585, -3.19872008, -3.20179432, -3.20486856, -3.20794280, -3.21101703, -3.21409127, -3.21716551, -3.22023975, -3.22331399, -3.22638822, -
		3.22946246, -3.23253670, -3.23561094, -3.23868517, -3.24175941, -3.24483365, -3.24790789, -3.25098213, -3.25405636, -3.25713060, -3.26020484, -3.26327908, -3.26635331, -3.26942755, -3.27250179, -
		3.27557603, -3.27865026, -3.28172450, -3.28479874, -3.28787298, -3.29094722, -3.29402145, -3.29709569, -3.30016993, -3.30324417, -3.30631840, -3.30939264, -3.31246688, -3.31554112, -3.31861536, -
		3.32168959, -3.32476383, -3.32783807, -3.33091231, -3.33398654, -3.33706078, -3.34013502, -3.34320926, -3.34628350, -3.34935773, -3.35243197, -3.35550621, -3.35858045, -3.36165468, -3.36472892, -
		3.36780316, -3.37087740, -3.37395164, -3.37702587, -3.38010011, -3.38317435, -3.38624859, -3.38932282, -3.39239706, -3.39547130, -3.39854554, -3.40161977, -3.40469401, -3.40776825, -3.41084249, -
		3.41391673, -3.41699096, -3.42006520, -3.42313944, -3.42621368, -3.42928791, -3.43236215, -3.43543639, -3.43851063, -3.44158487, -3.44465910, -3.44773334, -3.45080758, -3.45388182, -3.45695605, -
		3.46003029, -3.46310453, -3.46617877, -3.46925300, -3.47232724, -3.47540148, -3.47847572, -3.48154996, -3.48462419, -3.48769843, -3.49077267, -3.49384691, -3.49692114, -3.49999538, -3.50306962, -
		3.50614386, -3.50921810, -3.51229233, -3.51536657, -3.51844081, 3.55220244, 3.54918193, 3.54616141, 3.54314089, 3.54012037, 3.53709986, 3.53407934, 3.53105882, 3.52803831, 3.52501779, 3.52199727,
		3.51897675, 3.51595624, 3.51293572, 3.50991520, 3.50689468, 3.50387417, 3.50085365, 3.49783313, 3.49481261, 3.49179210, 3.48877158, 3.48575106, 3.48273055, 3.47971003, 3.47668951, 3.47366899,
		3.47064848, 3.46762796, 3.46460744, 3.46158692, 3.45856641, 3.45554589, 3.45252537, 3.44950486, 3.44648434, 3.44346382, 3.44044330, 3.43742279, 3.43440227, 3.43138175, 3.42836123, 3.42534072,
		3.42232020, 3.41929968, 3.41627917, 3.41325865, 3.41023813, 3.40721761, 3.40419710, 3.40117658, 3.39815606, 3.39513554, 3.39211503, 3.38909451, 3.38607399, 3.38305347, 3.38003296, 3.37701244,
		3.37399192, 3.37097141, 3.36795089, 3.36493037, 3.36190985, 3.35888934, 3.35586882, 3.35284830, 3.34982778, 3.34680727, 3.34378675, 3.34076623, 3.33774572, 3.33472520, 3.33170468, 3.32868416,
		3.32566365, 3.32264313, 3.31962261, 3.31660209, 3.31358158, 3.31056106, 3.30754054, 3.30452002, 3.30149951, 3.29847899, 3.29545847, 3.29243796, 3.28941744, 3.28639692, 3.28337640, 3.28035589,
		3.27733537, 3.27431485, 3.27129433, 3.26827382, 3.26525330, 3.26223278, 3.25921227, 3.25619175, 3.25317123, 3.25015071, 3.24713020, 3.24410968, 3.24108916, 3.23806864, 3.23504813, 3.23202761,
		3.22900709, 3.22598657, 3.22296606, 3.21994554, 3.21692502, 3.21390451, 3.21088399, 3.20786347, 3.20484295, 3.20182244, 3.19880192, 3.19578140, 3.19276088, 3.18974037, 3.18671985, 3.18369933,
		3.18067882, 3.17765830, 3.17463778, 3.17161726, 3.16859675, 3.16557623, 3.16255571, 3.15953519, 3.15651468, 3.15349416, 3.15047364, 3.14745312, 3.14443261, 3.14141209, 3.13839157, 3.13537106,
		3.13235054, 3.12933002, 3.12630950, 3.12328899, 3.12026847, 3.11724795, 3.11422743, 3.11120692, 3.10818640, 3.10516588, 3.10214537, 3.09912485, 3.09610433, 3.09308381, 3.09006330, 3.08704278,
		3.08402226, 3.08100174, 3.07798123, 3.07496071, 3.07194019, 3.06891967, 3.06589916, 3.06287864, 3.05985812, 3.05683761, 3.05381709, 3.05079657, 3.04777605, 3.04475554, 3.04173502, 3.03871450,
		3.03569398, 3.03267347, 3.02965295, 3.02663243, 3.02361192, 3.02059140, 3.01757088, 3.01455036, 3.01152985, 3.00850933, 3.00548881, 3.00246829, 2.99944778, 2.99642726, 2.99340674, 2.99038622,
		2.98736571, 2.98434519, 2.98132467, 2.97830416, 2.97528364, 2.97226312, 2.96924260, 2.96622209, 2.96320157, 2.96018105, 2.95716053, 2.95414002, 2.95111950, 2.94809898, 2.94507847, 2.94205795,
		2.93903743, 2.93601691, 2.93299640, 2.92997588, 2.92695536, 2.92393484, 2.92091433, 2.91789381, 2.91487329, 2.91185278, 2.90883226, 2.90581174, 2.90279122, 2.89977071, 2.89675019, 2.89372967,
		2.89070915, 2.88768864, 2.88466812, 2.88164760, 2.87862708, 2.87560657, 2.87258605, 2.86956553, 2.86654502, 2.86352450, 2.86050398, 2.85748346, 2.85446295, 2.85144243, 2.84842191, 2.84540139,
		2.84238088, 2.83936036, 2.83633984, 2.83331933, 2.83029881, 2.82727829, 2.82425777, 2.82123726, 2.81821674, 2.81519622, 2.81217570, 2.80915519, 2.80613467, 2.80311415, 2.80009363, 2.79707312,
		2.79405260, 2.79103208, 2.78801157, 2.78499105, 2.78197053, 2.77895001, 2.77592950, 2.77290898, 2.76988846, 2.76686794, 2.76384743, 2.76082691, 2.75780639, 2.75478588, 2.75176536, 2.74874484,
		2.74572432, 2.74270381, 2.73968329, 2.73666277, 2.73364225, 2.73062174, 2.72760122, 2.72458070, 2.72156018, 2.71853967, 2.71551915, 2.71249863, 2.70947812, 2.70645760, 2.70343708, 2.70041656,
		2.69739605, 2.69437553, 2.69135501, 2.68833449, 2.68531398, 2.68229346, 2.67927294, 2.67625243, 2.67323191, 2.67021139, 2.66719087, 2.66417036, 2.66114984, 2.65812932, 2.65510880, 2.65208829,
		2.64906777, 2.64604725, 2.64302674, 2.64000622, 2.63698570, 2.63396518, 2.63094467, 2.62792415, 2.62490363, 2.62188311, 2.61886260, 2.61584208, 2.61282156, 2.60980104, 2.60678053, 2.60376001,
		2.60073949, 2.59771898, 2.59469846, 2.59167794, 2.58865742, 2.58563691, 2.58261639, 2.57959587, 2.57657535, 2.57355484, 2.57053432, 2.56751380, 2.56449329, 2.56147277, 2.55845225, 2.55543173,
		2.55241122, 2.54939070, 2.54637018, 2.54334966, 2.54032915, 2.53730863, 2.53428811, 2.53126759, 2.52824708, 2.52522656, 2.52220604, 2.51918553, 2.51616501, 2.51314449, 2.51012397, 2.50710346,
		2.50408294, 2.50106242, 2.49804190, 2.49502139, 2.49200087, 2.48898035, 2.48595984, 2.48293932, 2.47991880, 2.47689828, 2.47387777, 2.47085725, 2.46783673, 2.46481621, 2.46179570, 2.45877518,
		2.45575466, 2.45273415, 2.44971363, 2.44669311, 2.44367259, 2.44065208, 2.43763156, 2.43461104, 2.43159052, 2.42857001, 2.42554949, 2.42252897, 2.41950845, 2.41648794, 2.41346742, 2.41044690,
		2.40742639, 2.40440587, 2.40138535, 2.39836483, 2.39534432, 2.39232380, 2.38930328, 2.38628276, 2.38326225, 2.38024173, 2.37722121, 2.37420070, 2.37118018, 2.36815966, 2.36513914, 2.36211863,
		2.35909811, 2.35607759, 2.35305707, 2.35003656, 2.34701604, 2.34399552, 2.34097500, 2.33795449, 2.33493397, 2.33191345, 2.32889294, 2.32587242, 2.32285190, 2.31983138, 2.31681087, 2.31379035,
		2.31076983, 2.30774931, 2.30472880, 2.30170828, 2.29868776, 2.29566725, 2.29264673, 2.28962621, 2.28660569, 2.28358518, 2.28056466, 2.27754414, 2.27452362, 2.27150311, 2.26848259, 2.26546207,
		2.26244156, 2.25942104, 2.25640052, 2.25338000, 2.25035949, 2.24733897, 2.24431845, 2.24129793, 2.23827742, 2.23525690, 2.23223638, 2.22921586, 2.22619535, 2.22317483, 2.22015431, 2.21713380,
		2.21411328, 2.21109276, 2.20807224, 2.20505173, 2.20203121, 2.19901069, 2.19599017, 2.19296966, 2.18994914, 2.18692862, 2.18390811, 2.18088759, 2.17786707, 2.17484655, 2.17182604, 2.16880552,
		2.16578500, 2.16276448, 2.15974397, 2.15672345, 2.15370293, 2.15068241, 2.14766190, 2.14464138, 2.14162086, 2.13860035, 2.13557983, 2.13255931, 2.12953879, 2.12651828, 2.12349776, 2.12047724,
		2.11745672, 2.11443621, 2.11141569, 2.10839517, 2.10537466, 2.10235414, 2.09933362, 2.09631310, 2.09329259, 2.09027207, 2.08725155, 2.08423103, 2.08121052, 2.07819000, 2.07516948, 2.07214897,
		2.06912845, 2.06610793, 2.06308741, 2.06006690, 2.05704638, 2.05402586, 2.05100534, 2.04798483, 2.04496431, 2.04194379, 2.03892327, 2.03590276, 2.03288224, 2.02986172, 2.02684121, 2.02382069,
		2.02080017, 2.01777965, 2.01475914, 2.01173862, 2.00871810, 2.00569758, 2.00267707, 1.99965655, 1.99663603, 1.99361552, 1.99059500, 1.98757448, 1.98455396, 1.98153345, 1.97851293, 1.97549241,
		1.97247189, 1.96945138, 1.96643086, 1.96341034, 1.96038983, 1.95736931, 1.95434879, 1.95132827, 1.94830776, 1.94528724, 1.94226672, 1.93924620, 1.93622569, 1.93320517, 1.93018465, 1.92716413,
		1.92414362, 1.92112310, 1.91810258, 1.91508207, 1.91206155, 1.90904103, 1.90602051, 1.90300000, 1.89997948, 1.89695896, 1.89393844, 1.89091793, 1.88789741, 1.88487689, 1.88185638, 1.87883586,
		1.87581534, 1.87279482, 1.86977431, 1.86675379, 1.86373327, 1.86071275, 1.85769224, 1.85467172, 1.85165120, 1.84863069, 1.84561017, 1.84258965, 1.83956913, 1.83654862, 1.83352810, 1.83050758,
		1.82748706, 1.82446655, 1.82144603, 1.81842551, 1.81540499, 1.81238448, 1.80936396, 1.80634344, 1.80332293, 1.80030241, 1.79728189, 1.79426137, 1.79124086, 1.78822034, 1.78519982, 1.78217930,
		1.77915879, 1.77613827, 1.77311775, 1.77009724, 1.76707672, 1.76405620, 1.76103568, 1.75801517, 1.75499465, 1.75197413, 1.74895361, 1.74593310, 1.74291258, 1.73989206, 1.73687155, 1.73385103,
		1.73083051, 1.72780999, 1.72478948, 1.72176896, 1.71874844, 1.71572792, 1.71270741, 1.70968689, 1.70666637, 1.70364585, 1.70062534, 1.69760482, 1.69458430, 1.69156379, 1.68854327, 1.68552275,
		1.68250223, 1.67948172, 1.67646120, 1.67344068, 1.67042016, 1.66739965, 1.66437913, 1.66135861, 1.65833810, 1.65531758, 1.65229706, 1.64927654, 1.64625603, 1.64323551, 1.64021499, 1.63719447,
		1.63417396, 1.63115344, 1.62813292, 1.62511241, 1.62209189, 1.61907137, 1.61605085, 1.61303034, 1.61000982, 1.60698930, 1.60396878, 1.60094827, 1.59792775, 1.59490723, 1.59188671, 1.58886620,
		1.58584568, 1.58282516, 1.57980465, 1.57678413, 1.57376361, 1.57074309, 1.56772258, 1.56470206, 1.56168154, 1.55866102, 1.55564051, 1.55261999, 1.54959947, 1.54657896, 1.54355844, 1.54053792,
		1.53751740, 1.53449689, 1.53147637, 1.52845585, 1.52543533, 1.52241482, 1.51939430, 1.51637378, 1.51335327, 1.51033275, 1.50731223, 1.50429171, 1.50127120, 1.49825068, 1.49523016, 1.49220964,
		1.48918913, 1.48616861, 1.48314809, 1.48012758, 1.47710706, 1.47408654, 1.47106602, 1.46804551, 1.46502499, 1.46200447, 1.45898395, 1.45596344, 1.45294292, 1.44992240, 1.44690188, 1.44388137,
		1.44086085, 1.43784033, 1.43481982, 1.43179930, 1.42877878, 1.42575826, 1.42273775, 1.41971723, 1.41669671, 1.41367619, 1.41065568, 1.40763516, 1.40461464, 1.40159413, 1.39857361, 1.39555309,
		1.39253257, 1.38951206, 1.38649154, 1.38347102, 1.38045050, 1.37742999, 1.37440947, 1.37138895, 1.36836844, 1.36534792, 1.36232740, 1.35930688, 1.35628637, 1.35326585, 1.35024533, 1.34722481,
		1.34420430, 1.34118378, 1.33816326, 1.33514275, 1.33212223, 1.32910171, 1.32608119, 1.32306068, 1.32004016, 1.31701964, 1.31399912, 1.31097861, 1.30795809, 1.30493757, 1.30191705, 1.29889654,
		1.29587602, 1.29285550, 1.28983499, 1.28681447, 1.28379395, 1.28077343, 1.27775292, 1.27473240, 1.27171188, 1.26869136, 1.26567085, 1.26265033, 1.25962981, 1.25660930, 1.25358878, 1.25056826,
		1.24754774, 1.24452723, 1.24150671, 1.23848619, 1.23546567, 1.23244516, 1.22942464, 1.22640412, 1.22338361, 1.22036309, 1.21734257, 1.21432205, 1.21130154, 1.20828102, 1.20526050, 1.20223998,
		1.19921947, 1.19619895, 1.19317843, 1.19015792, 1.18713740, 1.18411688, 1.18109636, 1.17807585, 1.17505533, 1.17203481, 1.16901429, 1.16599378, 1.16297326, 1.15995274, 1.15693222, 1.15391171,
		1.15089119, 1.14787067, 1.14485016, 1.14182964, 1.13880912, 1.13578860, 1.13276809, 1.12974757, 1.12672705, 1.12370653, 1.12068602, 1.11766550, 1.11464498, 1.11162447, 1.10860395, 1.10558343,
		1.10256291, 1.09954240, 1.09652188, 1.09350136, 1.09048084, 1.08746033, 1.08443981, 1.08141929, 1.07839878, 1.07537826, 1.07235774, 1.06933722, 1.06631671, 1.06329619, 1.06027567, 1.05725515,
		1.05423464, 1.05121412, 1.04819360, 1.04517309, 1.04215257, 1.03913205, 1.03611153, 1.03309102, 1.03007050, 1.02704998, 1.02402946, 1.02100895, 1.01798843, 1.01496791, 1.01194740, 1.00892688,
		1.00590636, 1.00288584, 0.99986533, 0.99684481, 0.99382429, 0.99080377, 0.98778326, 0.98476274, 0.98174222, 0.97872170, 0.97570119, 0.97268067, 0.96966015, 0.96663964, 0.96361912, 0.96059860,
		0.95757808, 0.95455757, 0.95153705, 0.94851653, 0.94549601, 0.94247550, 0.93945498, 0.93643446, 0.93341395, 0.93039343, 0.92737291, 0.92435239, 0.92133188, 0.91831136, 0.91529084, 0.91227032,
		0.90924981, 0.90622929, 0.90320877, 0.90018826, 0.89716774, 0.89414722, 0.89112670, 0.88810619, 0.88508567, 0.88206515, 0.87904463, 0.87602412, 0.87300360, 0.86998308, 0.86696257, 0.86394205,
		0.86092153, 0.85790101, 0.85488050, 0.85185998, 0.84883946, 0.84581894, 0.84279843, 0.83977791, 0.83675739, 0.83373688, 0.83071636, 0.82769584, 0.82467532, 0.82165481, 0.81863429, 0.81561377,
		0.81259325, 0.80957274, 0.80655222, 0.80353170, 0.80051118, 0.79749067, 0.79447015, 0.79144963, 0.78842912, 0.78540860, 0.78238808, 0.77936756, 0.77634705, 0.77332653, 0.77030601, 0.76728549,
		0.76426498, 0.76124446, 0.75822394, 0.75520343, 0.75218291, 0.74916239, 0.74614187, 0.74312136, 0.74010084, 0.73708032, 0.73405980, 0.73103929, 0.72801877, 0.72499825, 0.72197774, 0.71895722,
		0.71593670, 0.71291618, 0.70989567, 0.70687515, 0.70385463, 0.70083411, 0.69781360, 0.69479308, 0.69177256, 0.68875205, 0.68573153, 0.68271101, 0.67969049, 0.67666998, 0.67364946, 0.67062894,
		0.66760842, 0.66458791, 0.66156739, 0.65854687, 0.65552636, 0.65250584, 0.64948532, 0.64646480, 0.64344429, 0.64042377, 0.63740325, 0.63438273, 0.63136222, 0.62834170, 0.62532118, 0.62230067,
		0.61928015, 0.61625963, 0.61323911, 0.61021860, 0.60719808, 0.60417756, 0.60115704, 0.59813653, 0.59511601, 0.59209549, 0.58907497, 0.58605446, 0.58303394, 0.58001342, 0.57699291, 0.57397239,
		0.57095187, 0.56793135, 0.56491084, 0.56189032, 0.55886980, 0.55584928, 0.55282877, 0.54980825, 0.54678773, 0.54376722, 0.54074670, 0.53772618, 0.53470566, 0.53168515, 0.52866463, 0.52564411,
		0.52262359, 0.51960308, 0.51658256, 0.51356204, 0.51054153, 0.50752101, 0.50450049, 0.50147997, 0.49845946, 0.49543894, 0.49241842, 0.48939790, 0.48637739, 0.48335687, 0.48033635, 0.47731584,
		0.47429532, 0.47127480, 0.46825428, 0.46523377, 0.46221325, 0.45919273, 0.45617221, 0.45315170, 0.45013118, 0.44711066, 0.44409015, 0.44106963, 0.43804911, 0.43502859, 0.43200808, 0.42898756,
		0.42596704, 0.42294652, 0.41992601, 0.41690549, 0.41388497, 0.41086446, 0.40784394, 0.40482342, 0.40180290, 0.39878239, 0.39576187, 0.39274135, 0.38972083, 0.38670032, 0.38367980, 0.38065928,
		0.37763877, 0.37461825, 0.37159773, 0.36857721, 0.36555670, 0.36253618, 0.35951566, 0.35649514, 0.35347463, 0.35045411, 0.34743359, 0.34441308, 0.34139256, 0.33837204, 0.33535152, 0.33233101,
		0.32931049, 0.32628997, 0.32326945, 0.32024894, 0.31722842, 0.31420790, 0.31118739, 0.30816687, 0.30514635, 0.30212583, 0.29910532, 0.29608480, 0.29306428, 0.29004376, 0.28702325, 0.28400273,
		0.28098221, 0.27796169, 0.27494118, 0.27192066, 0.26890014, 0.26587963, 0.26285911, 0.25983859, 0.25681807, 0.25379756, 0.25077704, 0.24775652, 0.24473600, 0.24171549, 0.23869497, 0.23567445,
		0.23265394, 0.22963342, 0.22661290, 0.22359238, 0.22057187, 0.21755135, 0.21453083, 0.21151031, 0.20848980, 0.20546928, 0.20244876, 0.19942825, 0.19640773, 0.19338721, 0.19036669, 0.18734618,
		0.18432566, 0.18130514, 0.17828462, 0.17526411, 0.17224359, 0.16922307, 0.16620256, 0.16318204, 0.16016152, 0.15714100, 0.15412049, 0.15109997, 0.14807945, 0.14505893, 0.14203842, 0.13901790,
		0.13599738, 0.13297687, 0.12995635, 0.12693583, 0.12391531, 0.12089480, 0.11787428, 0.11485376, 0.11183324, 0.10881273, 0.10579221, 0.10277169, 0.09975118, 0.09673066, 0.09371014, 0.09068962,
		0.08766911, 0.08464859, 0.08162807, 0.07860755, 0.07558704, 0.07256652, 0.06954600, 0.06652549, 0.06350497, 0.06048445, 0.05746393, 0.05444342, 0.05142290, 0.04840238, 0.04538186, 0.04236135,
		0.03934083, 0.03632031, 0.03329980, 0.03027928, 0.02725876, 0.02423824, 0.02121773, 0.01819721, 0.01517669, 0.01215617, 0.00913566, 0.00611514, 0.00309462, 0.00007411, -0.00294641, -0.00596693, -
		0.00898745, -0.01200796, -0.01502848, -0.01804900, -0.02106952, -0.02409003, -0.02711055, -0.03013107, -0.03315158, -0.03617210, -0.03919262, -0.04221314, -0.04523365, -0.04825417, -0.05127469, -
		0.05429521, -0.05731572, -0.06033624, -0.06335676, -0.06637727, -0.06939779, -0.07241831, -0.07543883, -0.07845934, -0.08147986, -0.08450038, -0.08752090, -0.09054141, -0.09356193, -0.09658245, -
		0.09960296, -0.10262348, -0.10564400, -0.10866452, -0.11168503, -0.11470555, -0.11772607, -0.12074659, -0.12376710, -0.12678762, -0.12980814, -0.13282866, -0.13584917, -0.13886969, -0.14189021, -
		0.14491072, -0.14793124, -0.15095176, -0.15397228, -0.15699279, -0.16001331, -0.16303383, -0.16605435, -0.16907486, -0.17209538, -0.17511590, -0.17813641, -0.18115693, -0.18417745, -0.18719797, -
		0.19021848, -0.19323900, -0.19625952, -0.19928004, -0.20230055, -0.20532107, -0.20834159, -0.21136210, -0.21438262, -0.21740314, -0.22042366, -0.22344417, -0.22646469, -0.22948521, -0.23250573, -
		0.23552624, -0.23854676, -0.24156728, -0.24458779, -0.24760831, -0.25062883, -0.25364935, -0.25666986, -0.25969038, -0.26271090, -0.26573142, -0.26875193, -0.27177245, -0.27479297, -0.27781348, -
		0.28083400, -0.28385452, -0.28687504, -0.28989555, -0.29291607, -0.29593659, -0.29895711, -0.30197762, -0.30499814, -0.30801866, -0.31103917, -0.31405969, -0.31708021, -0.32010073, -0.32312124, -
		0.32614176, -0.32916228, -0.33218280, -0.33520331, -0.33822383, -0.34124435, -0.34426486, -0.34728538, -0.35030590, -0.35332642, -0.35634693, -0.35936745, -0.36238797, -0.36540849, -0.36842900, -
		0.37144952, -0.37447004, -0.37749055, -0.38051107, -0.38353159, -0.38655211, -0.38957262, -0.39259314, -0.39561366, -0.39863418, -0.40165469, -0.40467521, -0.40769573, -0.41071624, -0.41373676, -
		0.41675728, -0.41977780, -0.42279831, -0.42581883, -0.42883935, -0.43185987, -0.43488038, -0.43790090, -0.44092142, -0.44394193, -0.44696245, -0.44998297, -0.45300349, -0.45602400, -0.45904452, -
		0.46206504, -0.46508556, -0.46810607, -0.47112659, -0.47414711, -0.47716762, -0.48018814, -0.48320866, -0.48622918, -0.48924969, -0.49227021, -0.49529073, -0.49831125, -0.50133176, -0.50435228, -
		0.50737280, -0.51039332, -0.51341383, -0.51643435, -0.51945487, -0.52247538, -0.52549590, -0.52851642, -0.53153694, -0.53455745, -0.53757797, -0.54059849, -0.54361901, -0.54663952, -0.54966004, -
		0.55268056, -0.55570107, -0.55872159, -0.56174211, -0.56476263, -0.56778314, -0.57080366, -0.57382418, -0.57684470, -0.57986521, -0.58288573, -0.58590625, -0.58892676, -0.59194728, -0.59496780, -
		0.59798832, -0.60100883, -0.60402935, -0.60704987, -0.61007039, -0.61309090, -0.61611142, -0.61913194, -0.62215245, -0.62517297, -0.62819349, -0.63121401, -0.63423452, -0.63725504, -0.64027556, -
		0.64329608, -0.64631659, -0.64933711, -0.65235763, -0.65537814, -0.65839866, -0.66141918, -0.66443970, -0.66746021, -0.67048073, -0.67350125, -0.67652177, -0.67954228, -0.68256280, -0.68558332, -
		0.68860383, -0.69162435, -0.69464487, -0.69766539, -0.70068590, -0.70370642, -0.70672694, -0.70974746, -0.71276797, -0.71578849, -0.71880901, -0.72182952, -0.72485004, -0.72787056, -0.73089108, -
		0.73391159, -0.73693211, -0.73995263, -0.74297315, -0.74599366, -0.74901418, -0.75203470, -0.75505522, -0.75807573, -0.76109625, -0.76411677, -0.76713728, -0.77015780, -0.77317832, -0.77619884, -
		0.77921935, -0.78223987, -0.78526039, -0.78828091, -0.79130142, -0.79432194, -0.79734246, -0.80036297, -0.80338349, -0.80640401, -0.80942453, -0.81244504, -0.81546556, -0.81848608, -0.82150660, -
		0.82452711, -0.82754763, -0.83056815, -0.83358866, -0.83660918, -0.83962970, -0.84265022, -0.84567073, -0.84869125, -0.85171177, -0.85473229, -0.85775280, -0.86077332, -0.86379384, -0.86681435, -
		0.86983487, -0.87285539, -0.87587591, -0.87889642, -0.88191694, -0.88493746, -0.88795798, -0.89097849, -0.89399901, -0.89701953, -0.90004004, -0.90306056, -0.90608108, -0.90910160, -0.91212211, -
		0.91514263, -0.91816315, -0.92118367, -0.92420418, -0.92722470, -0.93024522, -0.93326574, -0.93628625, -0.93930677, -0.94232729, -0.94534780, -0.94836832, -0.95138884, -0.95440936, -0.95742987, -
		0.96045039, -0.96347091, -0.96649143, -0.96951194, -0.97253246, -0.97555298, -0.97857349, -0.98159401, -0.98461453, -0.98763505, -0.99065556, -0.99367608, -0.99669660, -0.99971712, -1.00273763, -
		1.00575815, -1.00877867, -1.01179918, -1.01481970, -1.01784022, -1.02086074, -1.02388125, -1.02690177, -1.02992229, -1.03294281, -1.03596332, -1.03898384, -1.04200436, -1.04502487, -1.04804539, -
		1.05106591, -1.05408643, -1.05710694, -1.06012746, -1.06314798, -1.06616850, -1.06918901, -1.07220953, -1.07523005, -1.07825057, -1.08127108, -1.08429160, -1.08731212, -1.09033263, -1.09335315, -
		1.09637367, -1.09939419, -1.10241470, -1.10543522, -1.10845574, -1.11147626, -1.11449677, -1.11751729, -1.12053781, -1.12355832, -1.12657884, -1.12959936, -1.13261988, -1.13564039, -1.13866091, -
		1.14168143, -1.14470195, -1.14772246, -1.15074298, -1.15376350, -1.15678402, -1.15980453, -1.16282505, -1.16584557, -1.16886608, -1.17188660, -1.17490712, -1.17792764, -1.18094815, -1.18396867, -
		1.18698919, -1.19000971, -1.19303022, -1.19605074, -1.19907126, -1.20209177, -1.20511229, -1.20813281, -1.21115333, -1.21417384, -1.21719436, -1.22021488, -1.22323540, -1.22625591, -1.22927643, -
		1.23229695, -1.23531747, -1.23833798, -1.24135850, -1.24437902, -1.24739953, -1.25042005, -1.25344057, -1.25646109, -1.25948160, -1.26250212, -1.26552264, -1.26854316, -1.27156367, -1.27458419, -
		1.27760471, -1.28062522, -1.28364574, -1.28666626, -1.28968678, -1.29270729, -1.29572781, -1.29874833, -1.30176885, -1.30478936, -1.30780988, -1.31083040, -1.31385092, -1.31687143, -1.31989195, -
		1.32291247, -1.32593298, -1.32895350, -1.33197402, -1.33499454, -1.33801505, -1.34103557, -1.34405609, -1.34707661, -1.35009712, -1.35311764, -1.35613816, -1.35915867, -1.36217919, -1.36519971, -
		1.36822023, -1.37124074, -1.37426126, -1.37728178, -1.38030230, -1.38332281, -1.38634333, -1.38936385, -1.39238437, -1.39540488, -1.39842540, -1.40144592, -1.40446643, -1.40748695, -1.41050747, -
		1.41352799, -1.41654850, -1.41956902, -1.42258954, -1.42561006, -1.42863057, -1.43165109, -1.43467161, -1.43769213, -1.44071264, -1.44373316, -1.44675368, -1.44977419, -1.45279471, -1.45581523, -
		1.45883575, -1.46185626, -1.46487678, -1.46789730, -1.47091782, -1.47393833, -1.47695885, -1.47997937, -1.48299989, -1.48602040, -1.48904092, -1.49206144, -1.49508195, -1.49810247, -1.50112299, -
		1.50414351, -1.50716402, -1.51018454, -1.51320506, -1.51622558, -1.51924609, -1.52226661, -1.52528713, -1.52830765, -1.53132816, -1.53434868, -1.53736920, -1.54038971, -1.54341023, -1.54643075, -
		1.54945127, -1.55247178, -1.55549230, -1.55851282, -1.56153334, -1.56455385, -1.56757437, -1.57059489, -1.57361541, -1.57663592, -1.57965644, -1.58267696, -1.58569747, -1.58871799, -1.59173851, -
		1.59475903, -1.59777954, -1.60080006, -1.60382058, -1.60684110, -1.60986161, -1.61288213, -1.61590265, -1.61892317, -1.62194368, -1.62496420, -1.62798472, -1.63100523, -1.63402575, -1.63704627, -
		1.64006679, -1.64308730, -1.64610782, -1.64912834, -1.65214886, -1.65516937, -1.65818989, -1.66121041, -1.66423093, -1.66725144, -1.67027196, -1.67329248, -1.67631299, -1.67933351, -1.68235403, -
		1.68537455, -1.68839506, -1.69141558, -1.69443610, -1.69745662, -1.70047713, -1.70349765, -1.70651817, -1.70953869, -1.71255920, -1.71557972, -1.71860024, -1.72162076, -1.72464127, -1.72766179, -
		1.73068231, -1.73370282, -1.73672334, -1.73974386, -1.74276438, -1.74578489, -1.74880541, -1.75182593, -1.75484645, -1.75786696, -1.76088748, -1.76390800, -1.76692852, -1.76994903, -1.77296955, -
		1.77599007, -1.77901059, -1.78203110, -1.78505162, -1.78807214, -1.79109265, -1.79411317, -1.79713369, -1.80015421, -1.80317472, -1.80619524, -1.80921576, -1.81223628, -1.81525679, -1.81827731, -
		1.82129783, -1.82431835, -1.82733886, -1.83035938, -1.83337990, -1.83640042, -1.83942093, -1.84244145, -1.84546197, -1.84848248, -1.85150300, -1.85452352, -1.85754404, -1.86056455, -1.86358507, -
		1.86660559, -1.86962611, -1.87264662, -1.87566714, -1.87868766, -1.88170818, -1.88472869, -1.88774921, -1.89076973, -1.89379025, -1.89681076, -1.89983128, -1.90285180, -1.90587231, -1.90889283, -
		1.91191335, -1.91493387, -1.91795438, -1.92097490, -1.92399542, -1.92701594, -1.93003645, -1.93305697, -1.93607749, -1.93909801, -1.94211852, -1.94513904, -1.94815956, -1.95118008, -1.95420059, -
		1.95722111, -1.96024163, -1.96326215, -1.96628266, -1.96930318, -1.97232370, -1.97534421, -1.97836473, -1.98138525, -1.98440577, -1.98742628, -1.99044680, -1.99346732, -1.99648784, -1.99950835, -
		2.00252887, -2.00554939, -2.00856991, -2.01159042, -2.01461094, -2.01763146, -2.02065198, -2.02367249, -2.02669301, -2.02971353, -2.03273405, -2.03575456, -2.03877508, -2.04179560, -2.04481612, -
		2.04783663, -2.05085715, -2.05387767, -2.05689818, -2.05991870, -2.06293922, -2.06595974, -2.06898025, -2.07200077, -2.07502129, -2.07804181, -2.08106232, -2.08408284, -2.08710336, -2.09012388, -
		2.09314439, -2.09616491, -2.09918543, -2.10220595, -2.10522646, -2.10824698, -2.11126750, -2.11428802, -2.11730853, -2.12032905, -2.12334957, -2.12637009, -2.12939060, -2.13241112, -2.13543164, -
		2.13845216, -2.14147267, -2.14449319, -2.14751371, -2.15053422, -2.15355474, -2.15657526, -2.15959578, -2.16261629, -2.16563681, -2.16865733, -2.17167785, -2.17469836, -2.17771888, -2.18073940, -
		2.18375992, -2.18678043, -2.18980095, -2.19282147, -2.19584199, -2.19886250, -2.20188302, -2.20490354, -2.20792406, -2.21094457, -2.21396509, -2.21698561, -2.22000613, -2.22302664, -2.22604716, -
		2.22906768, -2.23208820, -2.23510871, -2.23812923, -2.24114975, -2.24417027, -2.24719078, -2.25021130, -2.25323182, -2.25625234, -2.25927285, -2.26229337, -2.26531389, -2.26833441, -2.27135492, -
		2.27437544, -2.27739596, -2.28041648, -2.28343699, -2.28645751, -2.28947803, -2.29249855, -2.29551906, -2.29853958, -2.30156010, -2.30458061, -2.30760113, -2.31062165, -2.31364217, -2.31666268, -
		2.31968320, -2.32270372, -2.32572424, -2.32874475, -2.33176527, -2.33478579, -2.33780631, -2.34082682, -2.34384734, -2.34686786, -2.34988838, -2.35290889, -2.35592941, -2.35894993, -2.36197045, -
		2.36499096, -2.36801148, -2.37103200, -2.37405252, -2.37707303, -2.38009355, -2.38311407, -2.38613459, -2.38915510, -2.39217562, -2.39519614, -2.39821666, -2.40123717, -2.40425769, -2.40727821, -
		2.41029873, -2.41331924, -2.41633976, -2.41936028, -2.42238080, -2.42540131, -2.42842183, -2.43144235, -2.43446287, -2.43748338, -2.44050390, -2.44352442, -2.44654494, -2.44956545, -2.45258597, -
		2.45560649, -2.45862701, -2.46164752, -2.46466804, -2.46768856, -2.47070908, -2.47372959, -2.47675011, -2.47977063, -2.48279115, -2.48581166, -2.48883218, -2.49185270, -2.49487322, -2.49789373, -
		2.50091425, -2.50393477, -2.50695529, -2.50997581, -2.51299632, -2.51601684, -2.51903736, -2.52205788, -2.52507839, -2.52809891, -2.53111943, -2.53413995, -2.53716046, -2.54018098, -2.54320150, -
		2.54622202, -2.54924253, -2.55226305, -2.55528357, -2.55830409, -2.56132460, -2.56434512, -2.56736564, -2.57038616, -2.57340667, -2.57642719, -2.57944771, -2.58246823, -2.58548874, -2.58850926, -
		2.59152978, -2.59455030, -2.59757081, -2.60059133, -2.60361185, -2.60663237, -2.60965288, -2.61267340, -2.61569392, -2.61871444, -2.62173495, -2.62475547, -2.62777599, -2.63079651, -2.63381702, -
		2.63683754, -2.63985806, -2.64287858, -2.64589910, -2.64891961, -2.65194013, -2.65496065, -2.65798117, -2.66100168, -2.66402220, -2.66704272, -2.67006324, -2.67308375, -2.67610427, -2.67912479, -
		2.68214531, -2.68516582, -2.68818634, -2.69120686, -2.69422738, -2.69724789, -2.70026841, -2.70328893, -2.70630945, -2.70932997, -2.71235048, -2.71537100, -2.71839152, -2.72141204, -2.72443255, -
		2.72745307, -2.73047359, -2.73349411, -2.73651462, -2.73953514, -2.74255566, -2.74557618, -2.74859669, -2.75161721, -2.75463773, -2.75765825, -2.76067876, -2.76369928, -2.76671980, -2.76974032, -
		2.77276084, -2.77578135, -2.77880187, -2.78182239, -2.78484291, -2.78786342, -2.79088394, -2.79390446, -2.79692498, -2.79994549, -2.80296601, -2.80598653, -2.80900705, -2.81202756, -2.81504808, -
		2.81806860, -2.82108912, -2.82410963, -2.82713015, -2.83015067, -2.83317119, -2.83619171, -2.83921222, -2.84223274, -2.84525326, -2.84827378, -2.85129429, -2.85431481, -2.85733533, -2.86035585, -
		2.86337636, -2.86639688, -2.86941740, -2.87243792, -2.87545844, -2.87847895, -2.88149947, -2.88451999, -2.88754051, -2.89056102, -2.89358154, -2.89660206, -2.89962258, -2.90264309, -2.90566361, -
		2.90868413, -2.91170465, -2.91472517, -2.91774568, -2.92076620, -2.92378672, -2.92680724, -2.92982775, -2.93284827, -2.93586879, -2.93888931, -2.94190982, -2.94493034, -2.94795086, -2.95097138, -
		2.95399190, -2.95701241, -2.96003293, -2.96305345, -2.96607397, -2.96909448, -2.97211500, -2.97513552, -2.97815604, -2.98117655, -2.98419707, -2.98721759, -2.99023811, -2.99325863, -2.99627914, -
		2.99929966, -3.00232018, -3.00534070, -3.00836121, -3.01138173, -3.01440225, -3.01742277, -3.02044329, -3.02346380, -3.02648432, -3.02950484, -3.03252536, -3.03554587, -3.03856639, -3.04158691, -
		3.04460743, -3.04762795, -3.05064846, -3.05366898, -3.05668950, -3.05971002, -3.06273053, -3.06575105, -3.06877157, -3.07179209, -3.07481261, -3.07783312, -3.08085364, -3.08387416, -3.08689468, -
		3.08991519, -3.09293571, -3.09595623, -3.09897675, -3.10199727, -3.10501778, -3.10803830, -3.11105882, -3.11407934, -3.11709985, -3.12012037, -3.12314089, -3.12616141, -3.12918193, -3.13220244, -
		3.13522296, -3.13824348, -3.14126400, -3.14428452, -3.14730503, -3.15032555, -3.15334607, -3.15636659, -3.15938710, -3.16240762, -3.16542814, -3.16844866, -3.17146918, -3.17448969, -3.17751021, -
		3.18053073, -3.18355125, -3.18657177, -3.18959228, -3.19261280, -3.19563332, -3.19865384, -3.20167435, -3.20469487, -3.20771539, -3.21073591, -3.21375643, -3.21677694, -3.21979746, -3.22281798, -
		3.22583850, -3.22885902, -3.23187953, -3.23490005, -3.23792057, -3.24094109, -3.24396160, -3.24698212, -3.25000264, -3.25302316, -3.25604368, -3.25906419, -3.26208471, -3.26510523, -3.26812575, -
		3.27114627, -3.27416678, -3.27718730, -3.28020782, -3.28322834, -3.28624886, -3.28926937, -3.29228989, -3.29531041, -3.29833093, -3.30135144, 3.35296993, 3.34995242, 3.34693491, 3.34391740,
		3.34089989, 3.33788238, 3.33486487, 3.33184737, 3.32882986, 3.32581235, 3.32279484, 3.31977733, 3.31675982, 3.31374231, 3.31072481, 3.30770730, 3.30468979, 3.30167228, 3.29865477, 3.29563726,
		3.29261975, 3.28960225, 3.28658474, 3.28356723, 3.28054972, 3.27753221, 3.27451470, 3.27149719, 3.26847969, 3.26546218, 3.26244467, 3.25942716, 3.25640965, 3.25339214, 3.25037463, 3.24735713,
		3.24433962, 3.24132211, 3.23830460, 3.23528709, 3.23226958, 3.22925207, 3.22623457, 3.22321706, 3.22019955, 3.21718204, 3.21416453, 3.21114702, 3.20812951, 3.20511201, 3.20209450, 3.19907699,
		3.19605948, 3.19304197, 3.19002446, 3.18700695, 3.18398945, 3.18097194, 3.17795443, 3.17493692, 3.17191941, 3.16890190, 3.16588439, 3.16286689, 3.15984938, 3.15683187, 3.15381436, 3.15079685,
		3.14777934, 3.14476184, 3.14174433, 3.13872682, 3.13570931, 3.13269180, 3.12967429, 3.12665678, 3.12363928, 3.12062177, 3.11760426, 3.11458675, 3.11156924, 3.10855173, 3.10553422, 3.10251672,
		3.09949921, 3.09648170, 3.09346419, 3.09044668, 3.08742917, 3.08441166, 3.08139416, 3.07837665, 3.07535914, 3.07234163, 3.06932412, 3.06630661, 3.06328910, 3.06027160, 3.05725409, 3.05423658,
		3.05121907, 3.04820156, 3.04518405, 3.04216654, 3.03914904, 3.03613153, 3.03311402, 3.03009651, 3.02707900, 3.02406149, 3.02104399, 3.01802648, 3.01500897, 3.01199146, 3.00897395, 3.00595644,
		3.00293893, 2.99992143, 2.99690392, 2.99388641, 2.99086890, 2.98785139, 2.98483388, 2.98181637, 2.97879887, 2.97578136, 2.97276385, 2.96974634, 2.96672883, 2.96371132, 2.96069381, 2.95767631,
		2.95465880, 2.95164129, 2.94862378, 2.94560627, 2.94258876, 2.93957126, 2.93655375, 2.93353624, 2.93051873, 2.92750122, 2.92448371, 2.92146620, 2.91844870, 2.91543119, 2.91241368, 2.90939617,
		2.90637866, 2.90336115, 2.90034364, 2.89732614, 2.89430863, 2.89129112, 2.88827361, 2.88525610, 2.88223859, 2.87922109, 2.87620358, 2.87318607, 2.87016856, 2.86715105, 2.86413354, 2.86111603,
		2.85809853, 2.85508102, 2.85206351, 2.84904600, 2.84602849, 2.84301098, 2.83999347, 2.83697597, 2.83395846, 2.83094095, 2.82792344, 2.82490593, 2.82188842, 2.81887092, 2.81585341, 2.81283590,
		2.80981839, 2.80680088, 2.80378337, 2.80076586, 2.79774836, 2.79473085, 2.79171334, 2.78869583, 2.78567832, 2.78266081, 2.77964331, 2.77662580, 2.77360829, 2.77059078, 2.76757327, 2.76455576,
		2.76153825, 2.75852075, 2.75550324, 2.75248573, 2.74946822, 2.74645071, 2.74343320, 2.74041569, 2.73739819, 2.73438068, 2.73136317, 2.72834566, 2.72532815, 2.72231064, 2.71929314, 2.71627563,
		2.71325812, 2.71024061, 2.70722310, 2.70420559, 2.70118808, 2.69817058, 2.69515307, 2.69213556, 2.68911805, 2.68610054, 2.68308303, 2.68006553, 2.67704802, 2.67403051, 2.67101300, 2.66799549,
		2.66497798, 2.66196047, 2.65894297, 2.65592546, 2.65290795, 2.64989044, 2.64687293, 2.64385542, 2.64083792, 2.63782041, 2.63480290, 2.63178539, 2.62876788, 2.62575037, 2.62273287, 2.61971536,
		2.61669785, 2.61368034, 2.61066283, 2.60764532, 2.60462781, 2.60161031, 2.59859280, 2.59557529, 2.59255778, 2.58954027, 2.58652276, 2.58350526, 2.58048775, 2.57747024, 2.57445273, 2.57143522,
		2.56841771, 2.56540020, 2.56238270, 2.55936519, 2.55634768, 2.55333017, 2.55031266, 2.54729515, 2.54427765, 2.54126014, 2.53824263, 2.53522512, 2.53220761, 2.52919010, 2.52617259, 2.52315509,
		2.52013758, 2.51712007, 2.51410256, 2.51108505, 2.50806754, 2.50505004, 2.50203253, 2.49901502, 2.49599751, 2.49298000, 2.48996249, 2.48694499, 2.48392748, 2.48090997, 2.47789246, 2.47487495,
		2.47185744, 2.46883993, 2.46582243, 2.46280492, 2.45978741, 2.45676990, 2.45375239, 2.45073488, 2.44771738, 2.44469987, 2.44168236, 2.43866485, 2.43564734, 2.43262983, 2.42961233, 2.42659482,
		2.42357731, 2.42055980, 2.41754229, 2.41452478, 2.41150727, 2.40848977, 2.40547226, 2.40245475, 2.39943724, 2.39641973, 2.39340222, 2.39038472, 2.38736721, 2.38434970, 2.38133219, 2.37831468,
		2.37529717, 2.37227967, 2.36926216, 2.36624465, 2.36322714, 2.36020963, 2.35719212, 2.35417462, 2.35115711, 2.34813960, 2.34512209, 2.34210458, 2.33908707, 2.33606956, 2.33305206, 2.33003455,
		2.32701704, 2.32399953, 2.32098202, 2.31796451, 2.31494701, 2.31192950, 2.30891199, 2.30589448, 2.30287697, 2.29985946, 2.29684196, 2.29382445, 2.29080694, 2.28778943, 2.28477192, 2.28175441,
		2.27873691, 2.27571940, 2.27270189, 2.26968438, 2.26666687, 2.26364936, 2.26063186, 2.25761435, 2.25459684, 2.25157933, 2.24856182, 2.24554431, 2.24252680, 2.23950930, 2.23649179, 2.23347428,
		2.23045677, 2.22743926, 2.22442175, 2.22140425, 2.21838674, 2.21536923, 2.21235172, 2.20933421, 2.20631670, 2.20329920, 2.20028169, 2.19726418, 2.19424667, 2.19122916, 2.18821165, 2.18519415,
		2.18217664, 2.17915913, 2.17614162, 2.17312411, 2.17010660, 2.16708910, 2.16407159, 2.16105408, 2.15803657, 2.15501906, 2.15200155, 2.14898405, 2.14596654, 2.14294903, 2.13993152, 2.13691401,
		2.13389650, 2.13087900, 2.12786149, 2.12484398, 2.12182647, 2.11880896, 2.11579145, 2.11277395, 2.10975644, 2.10673893, 2.10372142, 2.10070391, 2.09768640, 2.09466890, 2.09165139, 2.08863388,
		2.08561637, 2.08259886, 2.07958135, 2.07656384, 2.07354634, 2.07052883, 2.06751132, 2.06449381, 2.06147630, 2.05845880, 2.05544129, 2.05242378, 2.04940627, 2.04638876, 2.04337125, 2.04035374,
		2.03733624, 2.03431873, 2.03130122, 2.02828371, 2.02526620, 2.02224869, 2.01923119, 2.01621368, 2.01319617, 2.01017866, 2.00716115, 2.00414364, 2.00112614, 1.99810863, 1.99509112, 1.99207361,
		1.98905610, 1.98603860, 1.98302109, 1.98000358, 1.97698607, 1.97396856, 1.97095105, 1.96793355, 1.96491604, 1.96189853, 1.95888102, 1.95586351, 1.95284600, 1.94982850, 1.94681099, 1.94379348,
		1.94077597, 1.93775846, 1.93474095, 1.93172345, 1.92870594, 1.92568843, 1.92267092, 1.91965341, 1.91663590, 1.91361840, 1.91060089, 1.90758338, 1.90456587, 1.90154836, 1.89853085, 1.89551335,
		1.89249584, 1.88947833, 1.88646082, 1.88344331, 1.88042580, 1.87740830, 1.87439079, 1.87137328, 1.86835577, 1.86533826, 1.86232075, 1.85930325, 1.85628574, 1.85326823, 1.85025072, 1.84723321,
		1.84421570, 1.84119820, 1.83818069, 1.83516318, 1.83214567, 1.82912816, 1.82611066, 1.82309315, 1.82007564, 1.81705813, 1.81404062, 1.81102311, 1.80800561, 1.80498810, 1.80197059, 1.79895308,
		1.79593557, 1.79291806, 1.78990056, 1.78688305, 1.78386554, 1.78084803, 1.77783052, 1.77481301, 1.77179551, 1.76877800, 1.76576049, 1.76274298, 1.75972547, 1.75670796, 1.75369046, 1.75067295,
		1.74765544, 1.74463793, 1.74162042, 1.73860292, 1.73558541, 1.73256790, 1.72955039, 1.72653288, 1.72351537, 1.72049787, 1.71748036, 1.71446285, 1.71144534, 1.70842783, 1.70541033, 1.70239282,
		1.69937531, 1.69635780, 1.69334029, 1.69032278, 1.68730528, 1.68428777, 1.68127026, 1.67825275, 1.67523524, 1.67221773, 1.66920023, 1.66618272, 1.66316521, 1.66014770, 1.65713019, 1.65411268,
		1.65109518, 1.64807767, 1.64506016, 1.64204265, 1.63902514, 1.63600764, 1.63299013, 1.62997262, 1.62695511, 1.62393760, 1.62092009, 1.61790259, 1.61488508, 1.61186757, 1.60885006, 1.60583255,
		1.60281504, 1.59979754, 1.59678003, 1.59376252, 1.59074501, 1.58772750, 1.58471000, 1.58169249, 1.57867498, 1.57565747, 1.57263996, 1.56962245, 1.56660495, 1.56358744, 1.56056993, 1.55755242,
		1.55453491, 1.55151740, 1.54849990, 1.54548239, 1.54246488, 1.53944737, 1.53642986, 1.53341236, 1.53039485, 1.52737734, 1.52435983, 1.52134232, 1.51832481, 1.51530731, 1.51228980, 1.50927229,
		1.50625478, 1.50323727, 1.50021977, 1.49720226, 1.49418475, 1.49116724, 1.48814973, 1.48513222, 1.48211472, 1.47909721, 1.47607970, 1.47306219, 1.47004468, 1.46702718, 1.46400967, 1.46099216,
		1.45797465, 1.45495714, 1.45193963, 1.44892213, 1.44590462, 1.44288711, 1.43986960, 1.43685209, 1.43383459, 1.43081708, 1.42779957, 1.42478206, 1.42176455, 1.41874704, 1.41572954, 1.41271203,
		1.40969452, 1.40667701, 1.40365950, 1.40064200, 1.39762449, 1.39460698, 1.39158947, 1.38857196, 1.38555445, 1.38253695, 1.37951944, 1.37650193, 1.37348442, 1.37046691, 1.36744940, 1.36443190,
		1.36141439, 1.35839688, 1.35537937, 1.35236186, 1.34934436, 1.34632685, 1.34330934, 1.34029183, 1.33727432, 1.33425682, 1.33123931, 1.32822180, 1.32520429, 1.32218678, 1.31916928, 1.31615177,
		1.31313426, 1.31011675, 1.30709924, 1.30408173, 1.30106423, 1.29804672, 1.29502921, 1.29201170, 1.28899419, 1.28597669, 1.28295918, 1.27994167, 1.27692416, 1.27390665, 1.27088914, 1.26787164,
		1.26485413, 1.26183662, 1.25881911, 1.25580160, 1.25278409, 1.24976659, 1.24674908, 1.24373157, 1.24071406, 1.23769655, 1.23467905, 1.23166154, 1.22864403, 1.22562652, 1.22260901, 1.21959150,
		1.21657400, 1.21355649, 1.21053898, 1.20752147, 1.20450396, 1.20148646, 1.19846895, 1.19545144, 1.19243393, 1.18941642, 1.18639892, 1.18338141, 1.18036390, 1.17734639, 1.17432888, 1.17131138,
		1.16829387, 1.16527636, 1.16225885, 1.15924134, 1.15622383, 1.15320633, 1.15018882, 1.14717131, 1.14415380, 1.14113629, 1.13811879, 1.13510128, 1.13208377, 1.12906626, 1.12604875, 1.12303125,
		1.12001374, 1.11699623, 1.11397872, 1.11096121, 1.10794370, 1.10492620, 1.10190869, 1.09889118, 1.09587367, 1.09285616, 1.08983866, 1.08682115, 1.08380364, 1.08078613, 1.07776862, 1.07475112,
		1.07173361, 1.06871610, 1.06569859, 1.06268108, 1.05966357, 1.05664607, 1.05362856, 1.05061105, 1.04759354, 1.04457603, 1.04155853, 1.03854102, 1.03552351, 1.03250600, 1.02948849, 1.02647099,
		1.02345348, 1.02043597, 1.01741846, 1.01440095, 1.01138344, 1.00836594, 1.00534843, 1.00233092, 0.99931341, 0.99629590, 0.99327840, 0.99026089, 0.98724338, 0.98422587, 0.98120836, 0.97819086,
		0.97517335, 0.97215584, 0.96913833, 0.96612082, 0.96310332, 0.96008581, 0.95706830, 0.95405079, 0.95103328, 0.94801577, 0.94499827, 0.94198076, 0.93896325, 0.93594574, 0.93292823, 0.92991073,
		0.92689322, 0.92387571, 0.92085820, 0.91784069, 0.91482319, 0.91180568, 0.90878817, 0.90577066, 0.90275315, 0.89973565, 0.89671814, 0.89370063, 0.89068312, 0.88766561, 0.88464811, 0.88163060,
		0.87861309, 0.87559558, 0.87257807, 0.86956056, 0.86654306, 0.86352555, 0.86050804, 0.85749053, 0.85447302, 0.85145552, 0.84843801, 0.84542050, 0.84240299, 0.83938548, 0.83636798, 0.83335047,
		0.83033296, 0.82731545, 0.82429794, 0.82128044, 0.81826293, 0.81524542, 0.81222791, 0.80921040, 0.80619290, 0.80317539, 0.80015788, 0.79714037, 0.79412286, 0.79110536, 0.78808785, 0.78507034,
		0.78205283, 0.77903532, 0.77601782, 0.77300031, 0.76998280, 0.76696529, 0.76394778, 0.76093028, 0.75791277, 0.75489526, 0.75187775, 0.74886024, 0.74584273, 0.74282523, 0.73980772, 0.73679021,
		0.73377270, 0.73075520, 0.72773769, 0.72472018, 0.72170267, 0.71868516, 0.71566765, 0.71265015, 0.70963264, 0.70661513, 0.70359762, 0.70058011, 0.69756261, 0.69454510, 0.69152759, 0.68851008,
		0.68549258, 0.68247507, 0.67945756, 0.67644005, 0.67342254, 0.67040504, 0.66738753, 0.66437002, 0.66135251, 0.65833500, 0.65531750, 0.65229999, 0.64928248, 0.64626497, 0.64324746, 0.64022995,
		0.63721245, 0.63419494, 0.63117743, 0.62815992, 0.62514241, 0.62212491, 0.61910740, 0.61608989, 0.61307238, 0.61005487, 0.60703737, 0.60401986, 0.60100235, 0.59798484, 0.59496734, 0.59194983,
		0.58893232, 0.58591481, 0.58289730, 0.57987979, 0.57686229, 0.57384478, 0.57082727, 0.56780976, 0.56479225, 0.56177475, 0.55875724, 0.55573973, 0.55272222, 0.54970471, 0.54668721, 0.54366970,
		0.54065219, 0.53763468, 0.53461717, 0.53159967, 0.52858216, 0.52556465, 0.52254714, 0.51952963, 0.51651213, 0.51349462, 0.51047711, 0.50745960, 0.50444209, 0.50142459, 0.49840708, 0.49538957,
		0.49237206, 0.48935455, 0.48633705, 0.48331954, 0.48030203, 0.47728452, 0.47426701, 0.47124951, 0.46823200, 0.46521449, 0.46219698, 0.45917947, 0.45616197, 0.45314446, 0.45012695, 0.44710944,
		0.44409193, 0.44107443, 0.43805692, 0.43503941, 0.43202190, 0.42900439, 0.42598689, 0.42296938, 0.41995187, 0.41693436, 0.41391685, 0.41089935, 0.40788184, 0.40486433, 0.40184682, 0.39882931,
		0.39581181, 0.39279430, 0.38977679, 0.38675928, 0.38374177, 0.38072427, 0.37770676, 0.37468925, 0.37167174, 0.36865423, 0.36563673, 0.36261922, 0.35960171, 0.35658420, 0.35356669, 0.35054919,
		0.34753168, 0.34451417, 0.34149666, 0.33847915, 0.33546165, 0.33244414, 0.32942663, 0.32640912, 0.32339161, 0.32037411, 0.31735660, 0.31433909, 0.31132158, 0.30830407, 0.30528657, 0.30226906,
		0.29925155, 0.29623404, 0.29321653, 0.29019903, 0.28718152, 0.28416401, 0.28114650, 0.27812899, 0.27511149, 0.27209398, 0.26907647, 0.26605896, 0.26304145, 0.26002395, 0.25700644, 0.25398893,
		0.25097142, 0.24795391, 0.24493641, 0.24191890, 0.23890139, 0.23588388, 0.23286637, 0.22984887, 0.22683136, 0.22381385, 0.22079634, 0.21777883, 0.21476133, 0.21174382, 0.20872631, 0.20570880,
		0.20269129, 0.19967379, 0.19665628, 0.19363877, 0.19062126, 0.18760375, 0.18458625, 0.18156874, 0.17855123, 0.17553372, 0.17251621, 0.16949871, 0.16648120, 0.16346369, 0.16044618, 0.15742867,
		0.15441117, 0.15139366, 0.14837615, 0.14535864, 0.14234114, 0.13932363, 0.13630612, 0.13328861, 0.13027110, 0.12725360, 0.12423609, 0.12121858, 0.11820107, 0.11518356, 0.11216605, 0.10914855,
		0.10613104, 0.10311353, 0.10009602, 0.09707851, 0.09406101, 0.09104350, 0.08802599, 0.08500848, 0.08199097, 0.07897347, 0.07595596, 0.07293845, 0.06992094, 0.06690344, 0.06388593, 0.06086842,
		0.05785091, 0.05483340, 0.05181589, 0.04879839, 0.04578088, 0.04276337, 0.03974586, 0.03672836, 0.03371085, 0.03069334, 0.02767583, 0.02465832, 0.02164081, 0.01862331, 0.01560580, 0.01258829,
		0.00957078, 0.00655327, 0.00353577, 0.00051826, -0.00249925, -0.00551676, -0.00853426, -0.01155177, -0.01456928, -0.01758679, -0.02060430, -0.02362180, -0.02663931, -0.02965682, -0.03267433, -
		0.03569184, -0.03870934, -0.04172685, -0.04474436, -0.04776187, -0.05077938, -0.05379688, -0.05681439, -0.05983190, -0.06284941, -0.06586692, -0.06888442, -0.07190193, -0.07491944, -0.07793695, -
		0.08095446, -0.08397196, -0.08698947, -0.09000698, -0.09302449, -0.09604200, -0.09905950, -0.10207701, -0.10509452, -0.10811203, -0.11112954, -0.11414704, -0.11716455, -0.12018206, -0.12319957, -
		0.12621708, -0.12923458, -0.13225209, -0.13526960, -0.13828711, -0.14130462, -0.14432212, -0.14733963, -0.15035714, -0.15337465, -0.15639215, -0.15940966, -0.16242717, -0.16544468, -0.16846219, -
		0.17147969, -0.17449720, -0.17751471, -0.18053222, -0.18354973, -0.18656723, -0.18958474, -0.19260225, -0.19561976, -0.19863727, -0.20165477, -0.20467228, -0.20768979, -0.21070730, -0.21372481, -
		0.21674231, -0.21975982, -0.22277733, -0.22579484, -0.22881235, -0.23182985, -0.23484736, -0.23786487, -0.24088238, -0.24389988, -0.24691739, -0.24993490, -0.25295241, -0.25596992, -0.25898743, -
		0.26200493, -0.26502244, -0.26803995, -0.27105746, -0.27407497, -0.27709247, -0.28010998, -0.28312749, -0.28614500, -0.28916250, -0.29218001, -0.29519752, -0.29821503, -0.30123254, -0.30425004, -
		0.30726755, -0.31028506, -0.31330257, -0.31632008, -0.31933758, -0.32235509, -0.32537260, -0.32839011, -0.33140762, -0.33442513, -0.33744263, -0.34046014, -0.34347765, -0.34649516, -0.34951266, -
		0.35253017, -0.35554768, -0.35856519, -0.36158270, -0.36460020, -0.36761771, -0.37063522, -0.37365273, -0.37667024, -0.37968774, -0.38270525, -0.38572276, -0.38874027, -0.39175778, -0.39477529, -
		0.39779279, -0.40081030, -0.40382781, -0.40684532, -0.40986283, -0.41288033, -0.41589784, -0.41891535, -0.42193286, -0.42495036, -0.42796787, -0.43098538, -0.43400289, -0.43702040, -0.44003790, -
		0.44305541, -0.44607292, -0.44909043, -0.45210794, -0.45512544, -0.45814295, -0.46116046, -0.46417797, -0.46719548, -0.47021299, -0.47323049, -0.47624800, -0.47926551, -0.48228302, -0.48530052, -
		0.48831803, -0.49133554, -0.49435305, -0.49737056, -0.50038807, -0.50340557, -0.50642308, -0.50944059, -0.51245810, -0.51547561, -0.51849311, -0.52151062, -0.52452813, -0.52754564, -0.53056315, -
		0.53358065, -0.53659816, -0.53961567, -0.54263318, -0.54565069, -0.54866819, -0.55168570, -0.55470321, -0.55772072, -0.56073823, -0.56375573, -0.56677324, -0.56979075, -0.57280826, -0.57582577, -
		0.57884328, -0.58186078, -0.58487829, -0.58789580, -0.59091331, -0.59393081, -0.59694832, -0.59996583, -0.60298334, -0.60600085, -0.60901836, -0.61203586, -0.61505337, -0.61807088, -0.62108839, -
		0.62410590, -0.62712340, -0.63014091, -0.63315842, -0.63617593, -0.63919344, -0.64221094, -0.64522845, -0.64824596, -0.65126347, -0.65428098, -0.65729848, -0.66031599, -0.66333350, -0.66635101, -
		0.66936852, -0.67238603, -0.67540353, -0.67842104, -0.68143855, -0.68445606, -0.68747357, -0.69049107, -0.69350858, -0.69652609, -0.69954360, -0.70256111, -0.70557861, -0.70859612, -0.71161363, -
		0.71463114, -0.71764865, -0.72066615, -0.72368366, -0.72670117, -0.72971868, -0.73273619, -0.73575369, -0.73877120, -0.74178871, -0.74480622, -0.74782373, -0.75084124, -0.75385874, -0.75687625, -
		0.75989376, -0.76291127, -0.76592878, -0.76894628, -0.77196379, -0.77498130, -0.77799881, -0.78101632, -0.78403383, -0.78705133, -0.79006884, -0.79308635, -0.79610386, -0.79912137, -0.80213888, -
		0.80515638, -0.80817389, -0.81119140, -0.81420891, -0.81722642, -0.82024392, -0.82326143, -0.82627894, -0.82929645, -0.83231396, -0.83533147, -0.83834897, -0.84136648, -0.84438399, -0.84740150, -
		0.85041901, -0.85343651, -0.85645402, -0.85947153, -0.86248904, -0.86550655, -0.86852405, -0.87154156, -0.87455907, -0.87757658, -0.88059409, -0.88361160, -0.88662910, -0.88964661, -0.89266412, -
		0.89568163, -0.89869914, -0.90171664, -0.90473415, -0.90775166, -0.91076917, -0.91378668, -0.91680419, -0.91982169, -0.92283920, -0.92585671, -0.92887422, -0.93189173, -0.93490923, -0.93792674, -
		0.94094425, -0.94396176, -0.94697927, -0.94999677, -0.95301428, -0.95603179, -0.95904930, -0.96206681, -0.96508432, -0.96810182, -0.97111933, -0.97413684, -0.97715435, -0.98017186, -0.98318936, -
		0.98620687, -0.98922438, -0.99224189, -0.99525940, -0.99827691, -1.00129441, -1.00431192, -1.00732943, -1.01034694, -1.01336445, -1.01638196, -1.01939946, -1.02241697, -1.02543448, -1.02845199, -
		1.03146950, -1.03448700, -1.03750451, -1.04052202, -1.04353953, -1.04655704, -1.04957455, -1.05259205, -1.05560956, -1.05862707, -1.06164458, -1.06466209, -1.06767960, -1.07069710, -1.07371461, -
		1.07673212, -1.07974963, -1.08276714, -1.08578464, -1.08880215, -1.09181966, -1.09483717, -1.09785468, -1.10087219, -1.10388969, -1.10690720, -1.10992471, -1.11294222, -1.11595973, -1.11897724, -
		1.12199474, -1.12501225, -1.12802976, -1.13104727, -1.13406478, -1.13708228, -1.14009979, -1.14311730, -1.14613481, -1.14915232, -1.15216983, -1.15518733, -1.15820484, -1.16122235, -1.16423986, -
		1.16725737, -1.17027488, -1.17329238, -1.17630989, -1.17932740, -1.18234491, -1.18536242, -1.18837993, -1.19139743, -1.19441494, -1.19743245, -1.20044996, -1.20346747, -1.20648498, -1.20950248, -
		1.21251999, -1.21553750, -1.21855501, -1.22157252, -1.22459003, -1.22760754, -1.23062504, -1.23364255, -1.23666006, -1.23967757, -1.24269508, -1.24571258, -1.24873009, -1.25174760, -1.25476511, -
		1.25778262, -1.26080013, -1.26381763, -1.26683514, -1.26985265, -1.27287016, -1.27588767, -1.27890518, -1.28192268, -1.28494019, -1.28795770, -1.29097521, -1.29399272, -1.29701023, -1.30002774, -
		1.30304524, -1.30606275, -1.30908026, -1.31209777, -1.31511528, -1.31813279, -1.32115029, -1.32416780, -1.32718531, -1.33020282, -1.33322033, -1.33623784, -1.33925535, -1.34227285, -1.34529036, -
		1.34830787, -1.35132538, -1.35434289, -1.35736040, -1.36037790, -1.36339541, -1.36641292, -1.36943043, -1.37244794, -1.37546545, -1.37848296, -1.38150046, -1.38451797, -1.38753548, -1.39055299, -
		1.39357050, -1.39658800, -1.39960552, -1.40262302, -1.40564053, -1.40865804, -1.41167555, -1.41469306, -1.41771057, -1.42072807, -1.42374558, -1.42676309, -1.42978060, -1.43279811, -1.43581562, -
		1.43883313, -1.44185063, -1.44486814, -1.44788565, -1.45090316, -1.45392067, -1.45693818, -1.45995569, -1.46297319, -1.46599070, -1.46900821, -1.47202572, -1.47504323, -1.47806074, -1.48107825, -
		1.48409575, -1.48711326, -1.49013077, -1.49314828, -1.49616579, -1.49918330, -1.50220081, -1.50521832, -1.50823582, -1.51125333, -1.51427084, -1.51728835, -1.52030586, -1.52332337, -1.52634088, -
		1.52935838, -1.53237589, -1.53539340, -1.53841091, -1.54142842, -1.54444593, -1.54746344, -1.55048094, -1.55349845, -1.55651596, -1.55953347, -1.56255098, -1.56556849, -1.56858600, -1.57160350, -
		1.57462102, -1.57763852, -1.58065603, -1.58367354, -1.58669105, -1.58970856, -1.59272607, -1.59574357, -1.59876108, -1.60177859, -1.60479610, -1.60781361, -1.61083112, -1.61384863, -1.61686614, -
		1.61988365, -1.62290115, -1.62591866, -1.62893617, -1.63195368, -1.63497119, -1.63798870, -1.64100621, -1.64402372, -1.64704122, -1.65005873, -1.65307624, -1.65609375, -1.65911126, -1.66212877, -
		1.66514628, -1.66816379, -1.67118129, -1.67419880, -1.67721631, -1.68023382, -1.68325133, -1.68626884, -1.68928635, -1.69230386, -1.69532137, -1.69833887, -1.70135638, -1.70437389, -1.70739140, -
		1.71040891, -1.71342642, -1.71644393, -1.71946144, -1.72247895, -1.72549645, -1.72851396, -1.73153147, -1.73454898, -1.73756649, -1.74058400, -1.74360151, -1.74661902, -1.74963653, -1.75265404, -
		1.75567154, -1.75868905, -1.76170656, -1.76472407, -1.76774158, -1.77075909, -1.77377660, -1.77679411, -1.77981162, -1.78282912, -1.78584663, -1.78886414, -1.79188165, -1.79489916, -1.79791667, -
		1.80093418, -1.80395169, -1.80696919, -1.80998671, -1.81300421, -1.81602172, -1.81903923, -1.82205674, -1.82507425, -1.82809176, -1.83110927, -1.83412678, -1.83714429, -1.84016180, -1.84317930, -
		1.84619681, -1.84921432, -1.85223183, -1.85524934, -1.85826685, -1.86128436, -1.86430187, -1.86731937, -1.87033689, -1.87335440, -1.87637190, -1.87938941, -1.88240692, -1.88542443, -1.88844194, -
		1.89145945, -1.89447696, -1.89749446, -1.90051197, -1.90352948, -1.90654699, -1.90956450, -1.91258201, -1.91559952, -1.91861703, -1.92163454, -1.92465205, -1.92766956, -1.93068707, -1.93370457, -
		1.93672208, -1.93973959, -1.94275710, -1.94577461, -1.94879212, -1.95180963, -1.95482714, -1.95784465, -1.96086216, -1.96387967, -1.96689717, -1.96991468, -1.97293219, -1.97594970, -1.97896721, -
		1.98198472, -1.98500223, -1.98801974, -1.99103725, -1.99405476, -1.99707227, -2.00008978, -2.00310728, -2.00612479, -2.00914230, -2.01215981, -2.01517732, -2.01819483, -2.02121234, -2.02422985, -
		2.02724736, -2.03026487, -2.03328238, -2.03629989, -2.03931740, -2.04233490, -2.04535241, -2.04836992, -2.05138743, -2.05440494, -2.05742245, -2.06043996, -2.06345747, -2.06647498, -2.06949249, -
		2.07251000, -2.07552751, -2.07854501, -2.08156253, -2.08458004, -2.08759754, -2.09061505, -2.09363256, -2.09665007, -2.09966758, -2.10268509, -2.10570260, -2.10872011, -2.11173762, -2.11475513, -
		2.11777264, -2.12079015, -2.12380766, -2.12682517, -2.12984268, -2.13286018, -2.13587769, -2.13889520, -2.14191271, -2.14493022, -2.14794773, -2.15096524, -2.15398275, -2.15700026, -2.16001777, -
		2.16303528, -2.16605279, -2.16907030, -2.17208781, -2.17510532, -2.17812283, -2.18114034, -2.18415785, -2.18717535, -2.19019286, -2.19321038, -2.19622788, -2.19924539, -2.20226290, -2.20528041, -
		2.20829792, -2.21131543, -2.21433294, -2.21735045, -2.22036796, -2.22338547, -2.22640298, -2.22942049, -2.23243800, -2.23545551, -2.23847302, -2.24149053, -2.24450804, -2.24752555, -2.25054306, -
		2.25356057, -2.25657807, -2.25959559, -2.26261309, -2.26563060, -2.26864811, -2.27166562, -2.27468313, -2.27770064, -2.28071815, -2.28373566, -2.28675317, -2.28977068, -2.29278819, -2.29580570, -
		2.29882321, -2.30184072, -2.30485823, -2.30787574, -2.31089325, -2.31391076, -2.31692827, -2.31994578, -2.32296329, -2.32598080, -2.32899831, -2.33201582, -2.33503333, -2.33805084, -2.34106835, -
		2.34408586, -2.34710337, -2.35012088, -2.35313839, -2.35615590, -2.35917341, -2.36219092, -2.36520843, -2.36822594, -2.37124345, -2.37426096, -2.37727847, -2.38029598, -2.38331349, -2.38633100, -
		2.38934851, -2.39236602, -2.39538353, -2.39840104, -2.40141855, -2.40443606, -2.40745356, -2.41047108, -2.41348859, -2.41650610, -2.41952361, -2.42254112, -2.42555862, -2.42857614, -2.43159364, -
		2.43461116, -2.43762867, -2.44064618, -2.44366369, -2.44668120, -2.44969871, -2.45271622, -2.45573373, -2.45875124, -2.46176875, -2.46478626, -2.46780377, -2.47082128, -2.47383879, -2.47685630, -
		2.47987381, -2.48289132, -2.48590883, -2.48892634, -2.49194385, -2.49496136, -2.49797887, -2.50099638, -2.50401389, -2.50703140, -2.51004891, -2.51306642, -2.51608393, -2.51910144, -2.52211895, -
		2.52513646, -2.52815397, -2.53117148, -2.53418899, -2.53720650, -2.54022401, -2.54324152, -2.54625903, -2.54927654, -2.55229405, -2.55531156, -2.55832907, -2.56134658, -2.56436409, -2.56738160, -
		2.57039911, -2.57341663, -2.57643413, -2.57945165, -2.58246915, -2.58548667, -2.58850418, -2.59152169, -2.59453920, -2.59755671, -2.60057422, -2.60359173, -2.60660924, -2.60962675, -2.61264426, -
		2.61566177, -2.61867928, -2.62169679, -2.62471430, -2.62773181, -2.63074932, -2.63376683, -2.63678434, -2.63980185, -2.64281937, -2.64583687, -2.64885439, -2.65187190, -2.65488941, -2.65790692, -
		2.66092443, -2.66394194, -2.66695945, -2.66997696, -2.67299447, -2.67601198, -2.67902949, -2.68204700, -2.68506451, -2.68808202, -2.69109953, -2.69411705, -2.69713455, -2.70015207, -2.70316958, -
		2.70618709, -2.70920460, -2.71222211, -2.71523962, -2.71825713, -2.72127464, -2.72429215, -2.72730966, -2.73032717, -2.73334468, -2.73636219, -2.73937971, -2.74239722, -2.74541473, -2.74843224, -
		2.75144975, -2.75446726, -2.75748477, -2.76050228, -2.76351979, -2.76653730, -2.76955481, -2.77257232, -2.77558984, -2.77860734, -2.78162486, -2.78464237, -2.78765988, -2.79067739, -2.79369490, -
		2.79671241, -2.79972992, -2.80274743, -2.80576494, -2.80878246, -2.81179997, -2.81481747, -2.81783498, -2.82085250, -2.82387001, -2.82688752, -2.82990503, -2.83292254, -2.83594005, -2.83895756, -
		2.84197507, -2.84499258, -2.84801009, -2.85102760, -2.85404512, -2.85706263, -2.86008014, -2.86309765, -2.86611516, -2.86913267, -2.87215018, -2.87516769, -2.87818521, -2.88120271, -2.88422023, -
		2.88723774, -2.89025525, -2.89327276, -2.89629027, -2.89930778, -2.90232529, -2.90534280, -2.90836031, -2.91137783, -2.91439534, -2.91741285, -2.92043036, -2.92344787, -2.92646538, -2.92948289, -
		2.93250040, -2.93551792, -2.93853543, -2.94155294, -2.94457045, -2.94758796, -2.95060547, -2.95362298, -2.95664049, -2.95965800, -2.96267551, -2.96569303, -2.96871054, -2.97172805, -2.97474556, -
		2.97776307, -2.98078058, -2.98379810, -2.98681561, -2.98983312, -2.99285063, -2.99586814, -2.99888565, -3.00190316, -3.00492068, -3.00793818, -3.01095570, -3.01397321, -3.01699072, -3.02000823, -
		3.02302574, -3.02604325, -3.02906076, -3.03207827, -3.03509579, -3.03811330, -3.04113081, -3.04414832, -3.04716583, -3.05018334, -3.05320086, -3.05621837, -3.05923588, -3.06225339, -3.06527090, -
		3.06828841, -3.07130593, -3.07432344, -3.07734095, -3.08035846, -3.08337597, -3.08639349, -3.08941099, -3.09242851, -3.09544602, -3.09846353, -3.10148104, -3.10449855, -3.10751607, -3.11053358, -
		3.11355109, -3.11656860, -3.11958611, -3.12260362, -3.12562114, -3.12863865, -3.13165616, -3.13467367, -3.13769118, -3.14070870, -3.14372621, -3.14674372, -3.14976123, -3.15277874, -3.15579625, -
		3.15881377, -3.16183128, -3.16484879, -3.16786630, -3.17088381, -3.17390133, -3.17691884, -3.17993635, 3.21891365, 3.21589924, 3.21288484, 3.20987043, 3.20685602, 3.20384162, 3.20082721, 3.19781280,
		3.19479840, 3.19178399, 3.18876958, 3.18575518, 3.18274077, 3.17972636, 3.17671196, 3.17369755, 3.17068314, 3.16766874, 3.16465433, 3.16163992, 3.15862552, 3.15561111, 3.15259670, 3.14958229,
		3.14656789, 3.14355348, 3.14053908, 3.13752467, 3.13451026, 3.13149586, 3.12848145, 3.12546704, 3.12245264, 3.11943823, 3.11642382, 3.11340942, 3.11039501, 3.10738060, 3.10436620, 3.10135179,
		3.09833738, 3.09532298, 3.09230857, 3.08929416, 3.08627976, 3.08326535, 3.08025094, 3.07723654, 3.07422213, 3.07120772, 3.06819332, 3.06517891, 3.06216450, 3.05915010, 3.05613569, 3.05312128,
		3.05010688, 3.04709247, 3.04407806, 3.04106366, 3.03804925, 3.03503484, 3.03202044, 3.02900603, 3.02599162, 3.02297722, 3.01996281, 3.01694840, 3.01393400, 3.01091959, 3.00790518, 3.00489078,
		3.00187637, 2.99886196, 2.99584756, 2.99283315, 2.98981874, 2.98680434, 2.98378993, 2.98077552, 2.97776112, 2.97474671, 2.97173230, 2.96871790, 2.96570349, 2.96268909, 2.95967468, 2.95666027,
		2.95364586, 2.95063146, 2.94761705, 2.94460265, 2.94158824, 2.93857383, 2.93555943, 2.93254502, 2.92953061, 2.92651621, 2.92350180, 2.92048739, 2.91747299, 2.91445858, 2.91144417, 2.90842977,
		2.90541536, 2.90240095, 2.89938655, 2.89637214, 2.89335773, 2.89034333, 2.88732892, 2.88431451, 2.88130011, 2.87828570, 2.87527129, 2.87225689, 2.86924248, 2.86622808, 2.86321367, 2.86019926,
		2.85718486, 2.85417045, 2.85115604, 2.84814163, 2.84512723, 2.84211282, 2.83909842, 2.83608401, 2.83306960, 2.83005520, 2.82704079, 2.82402638, 2.82101198, 2.81799757, 2.81498316, 2.81196876,
		2.80895435, 2.80593994, 2.80292554, 2.79991113, 2.79689672, 2.79388232, 2.79086791, 2.78785351, 2.78483910, 2.78182469, 2.77881028, 2.77579588, 2.77278147, 2.76976707, 2.76675266, 2.76373825,
		2.76072385, 2.75770944, 2.75469503, 2.75168063, 2.74866622, 2.74565181, 2.74263741, 2.73962300, 2.73660859, 2.73359419, 2.73057978, 2.72756537, 2.72455097, 2.72153656, 2.71852215, 2.71550775,
		2.71249334, 2.70947894, 2.70646453, 2.70345012, 2.70043572, 2.69742131, 2.69440690, 2.69139250, 2.68837809, 2.68536368, 2.68234928, 2.67933487, 2.67632046, 2.67330606, 2.67029165, 2.66727724,
		2.66426284, 2.66124843, 2.65823403, 2.65521962, 2.65220521, 2.64919081, 2.64617640, 2.64316199, 2.64014759, 2.63713318, 2.63411877, 2.63110437, 2.62808996, 2.62507555, 2.62206115, 2.61904674,
		2.61603233, 2.61301793, 2.61000352, 2.60698911, 2.60397471, 2.60096030, 2.59794589, 2.59493149, 2.59191708, 2.58890268, 2.58588827, 2.58287386, 2.57985946, 2.57684505, 2.57383064, 2.57081624,
		2.56780183, 2.56478742, 2.56177302, 2.55875861, 2.55574420, 2.55272980, 2.54971539, 2.54670099, 2.54368658, 2.54067217, 2.53765777, 2.53464336, 2.53162895, 2.52861455, 2.52560014, 2.52258573,
		2.51957133, 2.51655692, 2.51354252, 2.51052811, 2.50751370, 2.50449930, 2.50148489, 2.49847048, 2.49545608, 2.49244167, 2.48942726, 2.48641286, 2.48339845, 2.48038404, 2.47736964, 2.47435523,
		2.47134082, 2.46832642, 2.46531201, 2.46229760, 2.45928320, 2.45626879, 2.45325439, 2.45023998, 2.44722557, 2.44421117, 2.44119676, 2.43818235, 2.43516795, 2.43215354, 2.42913913, 2.42612473,
		2.42311032, 2.42009592, 2.41708151, 2.41406710, 2.41105270, 2.40803829, 2.40502388, 2.40200948, 2.39899507, 2.39598066, 2.39296626, 2.38995185, 2.38693745, 2.38392304, 2.38090863, 2.37789423,
		2.37487982, 2.37186541, 2.36885101, 2.36583660, 2.36282219, 2.35980779, 2.35679338, 2.35377898, 2.35076457, 2.34775016, 2.34473576, 2.34172135, 2.33870694, 2.33569254, 2.33267813, 2.32966372,
		2.32664932, 2.32363491, 2.32062051, 2.31760610, 2.31459169, 2.31157729, 2.30856288, 2.30554847, 2.30253407, 2.29951966, 2.29650525, 2.29349085, 2.29047644, 2.28746203, 2.28444763, 2.28143322,
		2.27841882, 2.27540441, 2.27239000, 2.26937560, 2.26636119, 2.26334678, 2.26033238, 2.25731797, 2.25430356, 2.25128916, 2.24827475, 2.24526035, 2.24224594, 2.23923153, 2.23621713, 2.23320272,
		2.23018831, 2.22717391, 2.22415950, 2.22114509, 2.21813069, 2.21511628, 2.21210188, 2.20908747, 2.20607306, 2.20305866, 2.20004425, 2.19702984, 2.19401544, 2.19100103, 2.18798662, 2.18497222,
		2.18195781, 2.17894341, 2.17592900, 2.17291459, 2.16990019, 2.16688578, 2.16387137, 2.16085697, 2.15784256, 2.15482815, 2.15181375, 2.14879934, 2.14578494, 2.14277053, 2.13975612, 2.13674172,
		2.13372731, 2.13071290, 2.12769850, 2.12468409, 2.12166969, 2.11865528, 2.11564087, 2.11262647, 2.10961206, 2.10659765, 2.10358325, 2.10056884, 2.09755443, 2.09454003, 2.09152562, 2.08851122,
		2.08549681, 2.08248240, 2.07946800, 2.07645359, 2.07343918, 2.07042478, 2.06741037, 2.06439596, 2.06138156, 2.05836715, 2.05535275, 2.05233834, 2.04932393, 2.04630953, 2.04329512, 2.04028071,
		2.03726631, 2.03425190, 2.03123750, 2.02822309, 2.02520868, 2.02219428, 2.01917987, 2.01616546, 2.01315106, 2.01013665, 2.00712224, 2.00410784, 2.00109343, 1.99807903, 1.99506462, 1.99205021,
		1.98903581, 1.98602140, 1.98300699, 1.97999259, 1.97697818, 1.97396378, 1.97094937, 1.96793496, 1.96492056, 1.96190615, 1.95889174, 1.95587734, 1.95286293, 1.94984852, 1.94683412, 1.94381971,
		1.94080531, 1.93779090, 1.93477649, 1.93176209, 1.92874768, 1.92573328, 1.92271887, 1.91970446, 1.91669006, 1.91367565, 1.91066124, 1.90764684, 1.90463243, 1.90161802, 1.89860362, 1.89558921,
		1.89257480, 1.88956040, 1.88654599, 1.88353158, 1.88051718, 1.87750277, 1.87448837, 1.87147396, 1.86845956, 1.86544515, 1.86243074, 1.85941634, 1.85640193, 1.85338752, 1.85037312, 1.84735871,
		1.84434430, 1.84132990, 1.83831549, 1.83530109, 1.83228668, 1.82927227, 1.82625787, 1.82324346, 1.82022905, 1.81721465, 1.81420024, 1.81118583, 1.80817143, 1.80515702, 1.80214262, 1.79912821,
		1.79611380, 1.79309940, 1.79008499, 1.78707059, 1.78405618, 1.78104177, 1.77802736, 1.77501296, 1.77199855, 1.76898415, 1.76596974, 1.76295533, 1.75994093, 1.75692652, 1.75391211, 1.75089771,
		1.74788330, 1.74486890, 1.74185449, 1.73884008, 1.73582568, 1.73281127, 1.72979687, 1.72678246, 1.72376805, 1.72075365, 1.71773924, 1.71472483, 1.71171043, 1.70869602, 1.70568162, 1.70266721,
		1.69965280, 1.69663840, 1.69362399, 1.69060958, 1.68759518, 1.68458077, 1.68156636, 1.67855196, 1.67553755, 1.67252315, 1.66950874, 1.66649433, 1.66347993, 1.66046552, 1.65745111, 1.65443671,
		1.65142230, 1.64840790, 1.64539349, 1.64237908, 1.63936468, 1.63635027, 1.63333586, 1.63032146, 1.62730705, 1.62429265, 1.62127824, 1.61826383, 1.61524943, 1.61223502, 1.60922061, 1.60620621,
		1.60319180, 1.60017740, 1.59716299, 1.59414858, 1.59113418, 1.58811977, 1.58510536, 1.58209096, 1.57907655, 1.57606214, 1.57304774, 1.57003333, 1.56701893, 1.56400452, 1.56099011, 1.55797571,
		1.55496130, 1.55194690, 1.54893249, 1.54591808, 1.54290368, 1.53988927, 1.53687486, 1.53386046, 1.53084605, 1.52783164, 1.52481724, 1.52180283, 1.51878842, 1.51577402, 1.51275961, 1.50974521,
		1.50673080, 1.50371640, 1.50070199, 1.49768758, 1.49467317, 1.49165877, 1.48864436, 1.48562996, 1.48261555, 1.47960114, 1.47658674, 1.47357233, 1.47055793, 1.46754352, 1.46452911, 1.46151471,
		1.45850030, 1.45548589, 1.45247149, 1.44945708, 1.44644267, 1.44342827, 1.44041386, 1.43739946, 1.43438505, 1.43137064, 1.42835624, 1.42534183, 1.42232742, 1.41931302, 1.41629861, 1.41328421,
		1.41026980, 1.40725539, 1.40424099, 1.40122658, 1.39821217, 1.39519777, 1.39218336, 1.38916896, 1.38615455, 1.38314014, 1.38012574, 1.37711133, 1.37409692, 1.37108252, 1.36806811, 1.36505371,
		1.36203930, 1.35902489, 1.35601049, 1.35299608, 1.34998167, 1.34696727, 1.34395286, 1.34093845, 1.33792405, 1.33490964, 1.33189523, 1.32888083, 1.32586642, 1.32285202, 1.31983761, 1.31682320,
		1.31380880, 1.31079439, 1.30777999, 1.30476558, 1.30175117, 1.29873677, 1.29572236, 1.29270795, 1.28969355, 1.28667914, 1.28366473, 1.28065033, 1.27763592, 1.27462152, 1.27160711, 1.26859270,
		1.26557830, 1.26256389, 1.25954948, 1.25653508, 1.25352067, 1.25050627, 1.24749186, 1.24447745, 1.24146305, 1.23844864, 1.23543423, 1.23241983, 1.22940542, 1.22639101, 1.22337661, 1.22036220,
		1.21734780, 1.21433339, 1.21131898, 1.20830458, 1.20529017, 1.20227576, 1.19926136, 1.19624695, 1.19323254, 1.19021814, 1.18720373, 1.18418933, 1.18117492, 1.17816051, 1.17514611, 1.17213170,
		1.16911729, 1.16610289, 1.16308848, 1.16007408, 1.15705967, 1.15404526, 1.15103086, 1.14801645, 1.14500204, 1.14198764, 1.13897323, 1.13595883, 1.13294442, 1.12993001, 1.12691561, 1.12390120,
		1.12088679, 1.11787239, 1.11485798, 1.11184357, 1.10882917, 1.10581476, 1.10280036, 1.09978595, 1.09677154, 1.09375714, 1.09074273, 1.08772832, 1.08471392, 1.08169951, 1.07868510, 1.07567070,
		1.07265629, 1.06964189, 1.06662748, 1.06361307, 1.06059867, 1.05758426, 1.05456985, 1.05155545, 1.04854104, 1.04552664, 1.04251223, 1.03949782, 1.03648342, 1.03346901, 1.03045460, 1.02744020,
		1.02442579, 1.02141138, 1.01839698, 1.01538257, 1.01236816, 1.00935376, 1.00633935, 1.00332495, 1.00031054, 0.99729613, 0.99428173, 0.99126732, 0.98825291, 0.98523851, 0.98222410, 0.97920969,
		0.97619529, 0.97318088, 0.97016648, 0.96715207, 0.96413766, 0.96112326, 0.95810885, 0.95509444, 0.95208004, 0.94906563, 0.94605123, 0.94303682, 0.94002241, 0.93700801, 0.93399360, 0.93097919,
		0.92796479, 0.92495038, 0.92193597, 0.91892157, 0.91590716, 0.91289276, 0.90987835, 0.90686394, 0.90384954, 0.90083513, 0.89782072, 0.89480632, 0.89179191, 0.88877750, 0.88576310, 0.88274869,
		0.87973428, 0.87671988, 0.87370547, 0.87069107, 0.86767666, 0.86466225, 0.86164785, 0.85863344, 0.85561903, 0.85260463, 0.84959022, 0.84657581, 0.84356141, 0.84054700, 0.83753259, 0.83451819,
		0.83150378, 0.82848938, 0.82547497, 0.82246056, 0.81944616, 0.81643175, 0.81341734, 0.81040294, 0.80738853, 0.80437412, 0.80135972, 0.79834531, 0.79533091, 0.79231650, 0.78930209, 0.78628768,
		0.78327328, 0.78025887, 0.77724446, 0.77423006, 0.77121565, 0.76820125, 0.76518684, 0.76217243, 0.75915803, 0.75614362, 0.75312921, 0.75011481, 0.74710040, 0.74408600, 0.74107159, 0.73805718,
		0.73504278, 0.73202837, 0.72901396, 0.72599956, 0.72298515, 0.71997074, 0.71695634, 0.71394193, 0.71092753, 0.70791312, 0.70489871, 0.70188430, 0.69886990, 0.69585549, 0.69284108, 0.68982668,
		0.68681227, 0.68379787, 0.68078346, 0.67776905, 0.67475465, 0.67174024, 0.66872583, 0.66571143, 0.66269702, 0.65968261, 0.65666821, 0.65365380, 0.65063940, 0.64762499, 0.64461058, 0.64159618,
		0.63858177, 0.63556736, 0.63255296, 0.62953855, 0.62652414, 0.62350974, 0.62049533, 0.61748092, 0.61446652, 0.61145211, 0.60843770, 0.60542330, 0.60240889, 0.59939448, 0.59638008, 0.59336567,
		0.59035127, 0.58733686, 0.58432245, 0.58130805, 0.57829364, 0.57527923, 0.57226483, 0.56925042, 0.56623601, 0.56322161, 0.56020720, 0.55719279, 0.55417839, 0.55116398, 0.54814957, 0.54513517,
		0.54212076, 0.53910635, 0.53609195, 0.53307754, 0.53006314, 0.52704873, 0.52403432, 0.52101991, 0.51800551, 0.51499110, 0.51197669, 0.50896229, 0.50594788, 0.50293348, 0.49991907, 0.49690466,
		0.49389025, 0.49087585, 0.48786144, 0.48484704, 0.48183263, 0.47881822, 0.47580381, 0.47278941, 0.46977500, 0.46676060, 0.46374619, 0.46073178, 0.45771738, 0.45470297, 0.45168856, 0.44867416,
		0.44565975, 0.44264534, 0.43963094, 0.43661653, 0.43360212, 0.43058772, 0.42757331, 0.42455890, 0.42154450, 0.41853009, 0.41551568, 0.41250128, 0.40948687, 0.40647246, 0.40345806, 0.40044365,
		0.39742924, 0.39441484, 0.39140043, 0.38838602, 0.38537162, 0.38235721, 0.37934280, 0.37632840, 0.37331399, 0.37029958, 0.36728518, 0.36427077, 0.36125636, 0.35824196, 0.35522755, 0.35221314,
		0.34919874, 0.34618433, 0.34316993, 0.34015552, 0.33714111, 0.33412671, 0.33111230, 0.32809789, 0.32508348, 0.32206908, 0.31905467, 0.31604026, 0.31302586, 0.31001145, 0.30699705, 0.30398264,
		0.30096823, 0.29795382, 0.29493942, 0.29192501, 0.28891060, 0.28589620, 0.28288179, 0.27986739, 0.27685298, 0.27383857, 0.27082416, 0.26780976, 0.26479535, 0.26178094, 0.25876654, 0.25575213,
		0.25273772, 0.24972332, 0.24670891, 0.24369450, 0.24068010, 0.23766569, 0.23465128, 0.23163688, 0.22862247, 0.22560806, 0.22259366, 0.21957925, 0.21656484, 0.21355044, 0.21053603, 0.20752162,
		0.20450722, 0.20149281, 0.19847840, 0.19546400, 0.19244959, 0.18943518, 0.18642078, 0.18340637, 0.18039196, 0.17737756, 0.17436315, 0.17134874, 0.16833433, 0.16531993, 0.16230552, 0.15929112,
		0.15627671, 0.15326230, 0.15024789, 0.14723349, 0.14421908, 0.14120467, 0.13819027, 0.13517586, 0.13216145, 0.12914705, 0.12613264, 0.12311824, 0.12010383, 0.11708942, 0.11407501, 0.11106061,
		0.10804620, 0.10503179, 0.10201739, 0.09900298, 0.09598857, 0.09297416, 0.08995976, 0.08694535, 0.08393094, 0.08091654, 0.07790213, 0.07488772, 0.07187332, 0.06885891, 0.06584450, 0.06283010,
		0.05981569, 0.05680128, 0.05378688, 0.05077247, 0.04775806, 0.04474365, 0.04172925, 0.03871484, 0.03570044, 0.03268603, 0.02967162, 0.02665721, 0.02364281, 0.02062840, 0.01761399, 0.01459959,
		0.01158518, 0.00857077, 0.00555637, 0.00254196, -0.00047245, -0.00348686, -0.00650126, -0.00951567, -0.01253008, -0.01554448, -0.01855889, -0.02157330, -0.02458770, -0.02760211, -0.03061652, -
		0.03363092, -0.03664533, -0.03965974, -0.04267415, -0.04568855, -0.04870296, -0.05171737, -0.05473177, -0.05774618, -0.06076059, -0.06377499, -0.06678940, -0.06980381, -0.07281821, -0.07583262, -
		0.07884703, -0.08186144, -0.08487584, -0.08789025, -0.09090466, -0.09391906, -0.09693347, -0.09994788, -0.10296228, -0.10597669, -0.10899110, -0.11200551, -0.11501991, -0.11803432, -0.12104873, -
		0.12406314, -0.12707754, -0.13009195, -0.13310636, -0.13612076, -0.13913517, -0.14214958, -0.14516398, -0.14817839, -0.15119280, -0.15420720, -0.15722161, -0.16023602, -0.16325042, -0.16626483, -
		0.16927924, -0.17229365, -0.17530805, -0.17832246, -0.18133687, -0.18435128, -0.18736568, -0.19038009, -0.19339450, -0.19640890, -0.19942331, -0.20243772, -0.20545212, -0.20846653, -0.21148094, -
		0.21449535, -0.21750975, -0.22052416, -0.22353857, -0.22655297, -0.22956738, -0.23258179, -0.23559619, -0.23861060, -0.24162501, -0.24463942, -0.24765382, -0.25066823, -0.25368264, -0.25669705, -
		0.25971145, -0.26272586, -0.26574027, -0.26875467, -0.27176908, -0.27478349, -0.27779789, -0.28081230, -0.28382671, -0.28684112, -0.28985552, -0.29286993, -0.29588434, -0.29889875, -0.30191315, -
		0.30492756, -0.30794197, -0.31095637, -0.31397078, -0.31698519, -0.31999960, -0.32301400, -0.32602841, -0.32904282, -0.33205722, -0.33507163, -0.33808604, -0.34110045, -0.34411485, -0.34712926, -
		0.35014367, -0.35315808, -0.35617248, -0.35918689, -0.36220130, -0.36521570, -0.36823011, -0.37124452, -0.37425892, -0.37727333, -0.38028774, -0.38330215, -0.38631656, -0.38933096, -0.39234537, -
		0.39535978, -0.39837418, -0.40138859, -0.40440300, -0.40741741, -0.41043181, -0.41344622, -0.41646063, -0.41947503, -0.42248944, -0.42550385, -0.42851826, -0.43153267, -0.43454707, -0.43756148, -
		0.44057589, -0.44359029, -0.44660470, -0.44961911, -0.45263352, -0.45564792, -0.45866233, -0.46167674, -0.46469115, -0.46770555, -0.47071996, -0.47373437, -0.47674877, -0.47976318, -0.48277759, -
		0.48579200, -0.48880640, -0.49182081, -0.49483522, -0.49784963, -0.50086403, -0.50387844, -0.50689285, -0.50990726, -0.51292166, -0.51593607, -0.51895048, -0.52196489, -0.52497929, -0.52799370, -
		0.53100811, -0.53402251, -0.53703692, -0.54005133, -0.54306574, -0.54608015, -0.54909455, -0.55210896, -0.55512337, -0.55813778, -0.56115218, -0.56416659, -0.56718100, -0.57019541, -0.57320981, -
		0.57622422, -0.57923863, -0.58225304, -0.58526744, -0.58828185, -0.59129626, -0.59431067, -0.59732507, -0.60033948, -0.60335389, -0.60636829, -0.60938270, -0.61239711, -0.61541152, -0.61842593, -
		0.62144033, -0.62445474, -0.62746915, -0.63048355, -0.63349796, -0.63651237, -0.63952678, -0.64254118, -0.64555559, -0.64857000, -0.65158441, -0.65459882, -0.65761322, -0.66062763, -0.66364204, -
		0.66665645, -0.66967085, -0.67268526, -0.67569967, -0.67871408, -0.68172848, -0.68474289, -0.68775730, -0.69077170, -0.69378612, -0.69680052, -0.69981493, -0.70282934, -0.70584375, -0.70885815, -
		0.71187256, -0.71488697, -0.71790138, -0.72091578, -0.72393019, -0.72694460, -0.72995900, -0.73297341, -0.73598782, -0.73900223, -0.74201664, -0.74503104, -0.74804545, -0.75105986, -0.75407427, -
		0.75708868, -0.76010308, -0.76311749, -0.76613190, -0.76914631, -0.77216071, -0.77517512, -0.77818953, -0.78120394, -0.78421835, -0.78723275, -0.79024716, -0.79326157, -0.79627598, -0.79929038, -
		0.80230479, -0.80531920, -0.80833361, -0.81134801, -0.81436242, -0.81737683, -0.82039124, -0.82340565, -0.82642005, -0.82943446, -0.83244887, -0.83546328, -0.83847768, -0.84149209, -0.84450650, -
		0.84752091, -0.85053532, -0.85354973, -0.85656413, -0.85957854, -0.86259295, -0.86560735, -0.86862176, -0.87163617, -0.87465058, -0.87766499, -0.88067939, -0.88369380, -0.88670821, -0.88972262, -
		0.89273703, -0.89575144, -0.89876584, -0.90178025, -0.90479466, -0.90780907, -0.91082347, -0.91383788, -0.91685229, -0.91986670, -0.92288111, -0.92589551, -0.92890992, -0.93192433, -0.93493874, -
		0.93795314, -0.94096755, -0.94398196, -0.94699637, -0.95001078, -0.95302518, -0.95603959, -0.95905400, -0.96206841, -0.96508282, -0.96809722, -0.97111163, -0.97412604, -0.97714045, -0.98015486, -
		0.98316926, -0.98618367, -0.98919808, -0.99221249, -0.99522689, -0.99824130, -1.00125571, -1.00427012, -1.00728453, -1.01029894, -1.01331335, -1.01632775, -1.01934216, -1.02235657, -1.02537098, -
		1.02838538, -1.03139979, -1.03441420, -1.03742861, -1.04044302, -1.04345742, -1.04647183, -1.04948624, -1.05250065, -1.05551506, -1.05852947, -1.06154387, -1.06455828, -1.06757269, -1.07058710, -
		1.07360151, -1.07661592, -1.07963032, -1.08264473, -1.08565914, -1.08867355, -1.09168795, -1.09470236, -1.09771677, -1.10073118, -1.10374559, -1.10676000, -1.10977440, -1.11278881, -1.11580322, -
		1.11881763, -1.12183204, -1.12484644, -1.12786085, -1.13087526, -1.13388967, -1.13690408, -1.13991849, -1.14293289, -1.14594730, -1.14896171, -1.15197612, -1.15499053, -1.15800493, -1.16101934, -
		1.16403375, -1.16704816, -1.17006257, -1.17307698, -1.17609138, -1.17910579, -1.18212020, -1.18513461, -1.18814902, -1.19116343, -1.19417784, -1.19719224, -1.20020665, -1.20322106, -1.20623547, -
		1.20924988, -1.21226429, -1.21527869, -1.21829310, -1.22130751, -1.22432192, -1.22733633, -1.23035074, -1.23336514, -1.23637955, -1.23939396, -1.24240837, -1.24542278, -1.24843718, -1.25145159, -
		1.25446600, -1.25748041, -1.26049482, -1.26350923, -1.26652363, -1.26953804, -1.27255245, -1.27556686, -1.27858127, -1.28159567, -1.28461009, -1.28762449, -1.29063890, -1.29365331, -1.29666772, -
		1.29968213, -1.30269654, -1.30571095, -1.30872535, -1.31173976, -1.31475417, -1.31776858, -1.32078299, -1.32379740, -1.32681180, -1.32982621, -1.33284062, -1.33585503, -1.33886944, -1.34188385, -
		1.34489825, -1.34791266, -1.35092707, -1.35394148, -1.35695589, -1.35997030, -1.36298471, -1.36599911, -1.36901352, -1.37202793, -1.37504234, -1.37805675, -1.38107116, -1.38408557, -1.38709997, -
		1.39011438, -1.39312879, -1.39614320, -1.39915761, -1.40217202, -1.40518643, -1.40820084, -1.41121524, -1.41422965, -1.41724406, -1.42025847, -1.42327288, -1.42628729, -1.42930169, -1.43231610, -
		1.43533051, -1.43834492, -1.44135933, -1.44437374, -1.44738815, -1.45040256, -1.45341696, -1.45643137, -1.45944578, -1.46246019, -1.46547460, -1.46848901, -1.47150341, -1.47451782, -1.47753223, -
		1.48054664, -1.48356105, -1.48657546, -1.48958987, -1.49260428, -1.49561869, -1.49863310, -1.50164750, -1.50466191, -1.50767632, -1.51069073, -1.51370514, -1.51671955, -1.51973396, -1.52274837, -
		1.52576277, -1.52877718, -1.53179159, -1.53480600, -1.53782041, -1.54083482, -1.54384923, -1.54686364, -1.54987804, -1.55289245, -1.55590686, -1.55892127, -1.56193568, -1.56495009, -1.56796450, -
		1.57097891, -1.57399331, -1.57700772, -1.58002213, -1.58303654, -1.58605095, -1.58906536, -1.59207977, -1.59509418, -1.59810859, -1.60112300, -1.60413740, -1.60715181, -1.61016622, -1.61318063, -
		1.61619504, -1.61920945, -1.62222386, -1.62523827, -1.62825267, -1.63126708, -1.63428149, -1.63729590, -1.64031031, -1.64332472, -1.64633913, -1.64935354, -1.65236795, -1.65538236, -1.65839676, -
		1.66141117, -1.66442558, -1.66743999, -1.67045440, -1.67346881, -1.67648322, -1.67949763, -1.68251204, -1.68552644, -1.68854086, -1.69155527, -1.69456967, -1.69758408, -1.70059849, -1.70361290, -
		1.70662731, -1.70964172, -1.71265613, -1.71567054, -1.71868495, -1.72169935, -1.72471377, -1.72772817, -1.73074258, -1.73375699, -1.73677140, -1.73978581, -1.74280022, -1.74581463, -1.74882904, -
		1.75184345, -1.75485785, -1.75787226, -1.76088667, -1.76390108, -1.76691549, -1.76992990, -1.77294431, -1.77595872, -1.77897313, -1.78198754, -1.78500195, -1.78801636, -1.79103077, -1.79404517, -
		1.79705959, -1.80007399, -1.80308840, -1.80610281, -1.80911722, -1.81213163, -1.81514604, -1.81816045, -1.82117486, -1.82418927, -1.82720368, -1.83021809, -1.83323250, -1.83624691, -1.83926131, -
		1.84227572, -1.84529013, -1.84830454, -1.85131895, -1.85433336, -1.85734777, -1.86036218, -1.86337659, -1.86639100, -1.86940541, -1.87241982, -1.87543423, -1.87844864, -1.88146304, -1.88447745, -
		1.88749186, -1.89050627, -1.89352068, -1.89653509, -1.89954950, -1.90256391, -1.90557832, -1.90859273, -1.91160714, -1.91462155, -1.91763596, -1.92065037, -1.92366478, -1.92667919, -1.92969360, -
		1.93270800, -1.93572241, -1.93873682, -1.94175123, -1.94476565, -1.94778005, -1.95079446, -1.95380887, -1.95682328, -1.95983769, -1.96285210, -1.96586651, -1.96888092, -1.97189533, -1.97490974, -
		1.97792415, -1.98093856, -1.98395297, -1.98696738, -1.98998179, -1.99299620, -1.99601060, -1.99902501, -2.00203942, -2.00505383, -2.00806824, -2.01108265, -2.01409706, -2.01711147, -2.02012588, -
		2.02314029, -2.02615470, -2.02916911, -2.03218352, -2.03519793, -2.03821234, -2.04122675, -2.04424116, -2.04725557, -2.05026998, -2.05328439, -2.05629880, -2.05931321, -2.06232762, -2.06534203, -
		2.06835644, -2.07137085, -2.07438526, -2.07739966, -2.08041407, -2.08342848, -2.08644289, -2.08945731, -2.09247171, -2.09548612, -2.09850053, -2.10151494, -2.10452935, -2.10754376, -2.11055817, -
		2.11357258, -2.11658699, -2.11960140, -2.12261581, -2.12563022, -2.12864463, -2.13165904, -2.13467345, -2.13768786, -2.14070227, -2.14371668, -2.14673109, -2.14974550, -2.15275991, -2.15577432, -
		2.15878873, -2.16180314, -2.16481755, -2.16783196, -2.17084637, -2.17386078, -2.17687519, -2.17988960, -2.18290401, -2.18591842, -2.18893283, -2.19194724, -2.19496165, -2.19797606, -2.20099047, -
		2.20400488, -2.20701929, -2.21003370, -2.21304811, -2.21606252, -2.21907693, -2.22209134, -2.22510575, -2.22812016, -2.23113457, -2.23414898, -2.23716339, -2.24017779, -2.24319221, -2.24620662, -
		2.24922103, -2.25223544, -2.25524985, -2.25826426, -2.26127867, -2.26429308, -2.26730749, -2.27032190, -2.27333630, -2.27635072, -2.27936513, -2.28237954, -2.28539395, -2.28840836, -2.29142277, -
		2.29443718, -2.29745159, -2.30046600, -2.30348041, -2.30649482, -2.30950923, -2.31252364, -2.31553805, -2.31855246, -2.32156687, -2.32458128, -2.32759569, -2.33061010, -2.33362451, -2.33663892, -
		2.33965333, -2.34266774, -2.34568215, -2.34869656, -2.35171097, -2.35472538, -2.35773979, -2.36075420, -2.36376861, -2.36678302, -2.36979743, -2.37281184, -2.37582625, -2.37884066, -2.38185507, -
		2.38486948, -2.38788389, -2.39089830, -2.39391271, -2.39692712, -2.39994153, -2.40295594, -2.40597035, -2.40898476, -2.41199917, -2.41501358, -2.41802799, -2.42104240, -2.42405681, -2.42707122, -
		2.43008563, -2.43310005, -2.43611445, -2.43912886, -2.44214328, -2.44515768, -2.44817210, -2.45118651, -2.45420092, -2.45721533, -2.46022974, -2.46324415, -2.46625856, -2.46927297, -2.47228738, -
		2.47530179, -2.47831620, -2.48133061, -2.48434502, -2.48735943, -2.49037384, -2.49338825, -2.49640266, -2.49941707, -2.50243148, -2.50544590, -2.50846030, -2.51147471, -2.51448913, -2.51750354, -
		2.52051795, -2.52353236, -2.52654676, -2.52956118, -2.53257559, -2.53559000, -2.53860441, -2.54161882, -2.54463323, -2.54764764, -2.55066205, -2.55367646, -2.55669087, -2.55970528, -2.56271969, -
		2.56573411, -2.56874851, -2.57176292, -2.57477734, -2.57779175, -2.58080616, -2.58382057, -2.58683498, -2.58984939, -2.59286380, -2.59587821, -2.59889262, -2.60190703, -2.60492144, -2.60793585, -
		2.61095026, -2.61396467, -2.61697908, -2.61999350, -2.62300791, -2.62602232, -2.62903673, -2.63205114, -2.63506555, -2.63807996, -2.64109437, -2.64410878, -2.64712319, -2.65013760, -2.65315201, -
		2.65616642, -2.65918083, -2.66219524, -2.66520965, -2.66822406, -2.67123848, -2.67425288, -2.67726730, -2.68028171, -2.68329612, -2.68631053, -2.68932494, -2.69233935, -2.69535376, -2.69836817, -
		2.70138258, -2.70439699, -2.70741141, -2.71042581, -2.71344023, -2.71645464, -2.71946905, -2.72248346, -2.72549787, -2.72851228, -2.73152669, -2.73454110, -2.73755551, -2.74056992, -2.74358433, -
		2.74659874, -2.74961316, -2.75262757, -2.75564197, -2.75865639, -2.76167080, -2.76468521, -2.76769962, -2.77071403, -2.77372844, -2.77674285, -2.77975726, -2.78277168, -2.78578609, -2.78880050, -
		2.79181491, -2.79482932, -2.79784373, -2.80085814, -2.80387255, -2.80688696, -2.80990137, -2.81291578, -2.81593020, -2.81894461, -2.82195902, -2.82497343, -2.82798784, -2.83100225, -2.83401666, -
		2.83703107, -2.84004548, -2.84305989, -2.84607431, -2.84908872, -2.85210313, -2.85511754, -2.85813195, -2.86114636, -2.86416077, -2.86717518, -2.87018959, -2.87320400, -2.87621842, -2.87923283, -
		2.88224724, -2.88526165, -2.88827606, -2.89129047, -2.89430488, -2.89731930, -2.90033370, -2.90334811, -2.90636253, -2.90937694, -2.91239135, -2.91540576, -2.91842017, -2.92143458, -2.92444899, -
		2.92746341, -2.93047782, -2.93349223, -2.93650664, -2.93952105, -2.94253546, -2.94554987, -2.94856428, -2.95157869, -2.95459311, -2.95760752, -2.96062193, -2.96363634, -2.96665075, -2.96966516, -
		2.97267957, -2.97569398, -2.97870839, -2.98172281, -2.98473722, -2.98775163, -2.99076604, -2.99378045, -2.99679486, -2.99980927, -3.00282368, -3.00583810, -3.00885251, -3.01186692, -3.01488133, -
		3.01789574, -3.02091015, -3.02392457, -3.02693897, -3.02995339, -3.03296780, -3.03598221, -3.03899662, -3.04201103, -3.04502544, -3.04803986, -3.05105426, -3.05406868, -3.05708309, -3.06009750, -
		3.06311191, -3.06612632, -3.06914073, -3.07215514, -3.07516956, -3.07818397, -3.08119838, -3.08421279, -3.08722720, -3.09024161, -3.09325602, -3.09627044, -3.09928485, -3.10229926, -3.10531367, -
		3.10832808, -3.11134250, -3.11435691, -3.11737132, -3.12038573, -3.12340014, -3.12641455, -3.12942896, -3.13244337, -3.13545778, -3.13847220, -3.14148661, -3.14450102, -3.14751543, -3.15052984, -
		3.15354425, -3.15655867, 3.15354454, 3.15053013, 3.14751572, 3.14450131, 3.14148689, 3.13847249, 3.13545807, 3.13244366, 3.12942925, 3.12641484, 3.12340043, 3.12038601, 3.11737161, 3.11435719,
		3.11134278, 3.10832837, 3.10531396, 3.10229955, 3.09928514, 3.09627072, 3.09325631, 3.09024190, 3.08722749, 3.08421308, 3.08119867, 3.07818426, 3.07516984, 3.07215543, 3.06914102, 3.06612661,
		3.06311220, 3.06009778, 3.05708338, 3.05406896, 3.05105456, 3.04804015, 3.04502573, 3.04201132, 3.03899691, 3.03598250, 3.03296808, 3.02995367, 3.02693926, 3.02392485, 3.02091044, 3.01789603,
		3.01488161, 3.01186721, 3.00885280, 3.00583838, 3.00282397, 2.99980956, 2.99679515, 2.99378074, 2.99076632, 2.98775192, 2.98473750, 2.98172309, 2.97870868, 2.97569427, 2.97267986, 2.96966545,
		2.96665104, 2.96363663, 2.96062221, 2.95760780, 2.95459339, 2.95157898, 2.94856457, 2.94555016, 2.94253575, 2.93952134, 2.93650692, 2.93349251, 2.93047810, 2.92746369, 2.92444928, 2.92143487,
		2.91842046, 2.91540605, 2.91239164, 2.90937722, 2.90636281, 2.90334840, 2.90033399, 2.89731958, 2.89430517, 2.89129076, 2.88827635, 2.88526193, 2.88224753, 2.87923311, 2.87621870, 2.87320429,
		2.87018988, 2.86717547, 2.86416106, 2.86114665, 2.85813224, 2.85511783, 2.85210342, 2.84908900, 2.84607459, 2.84306018, 2.84004577, 2.83703136, 2.83401695, 2.83100254, 2.82798813, 2.82497371,
		2.82195930, 2.81894489, 2.81593048, 2.81291607, 2.80990166, 2.80688725, 2.80387284, 2.80085843, 2.79784402, 2.79482960, 2.79181519, 2.78880078, 2.78578637, 2.78277196, 2.77975755, 2.77674314,
		2.77372873, 2.77071432, 2.76769991, 2.76468550, 2.76167109, 2.75865667, 2.75564226, 2.75262785, 2.74961344, 2.74659903, 2.74358462, 2.74057021, 2.73755579, 2.73454138, 2.73152698, 2.72851257,
		2.72549815, 2.72248374, 2.71946933, 2.71645492, 2.71344051, 2.71042610, 2.70741169, 2.70439728, 2.70138287, 2.69836846, 2.69535404, 2.69233964, 2.68932522, 2.68631081, 2.68329640, 2.68028199,
		2.67726758, 2.67425317, 2.67123876, 2.66822435, 2.66520994, 2.66219553, 2.65918112, 2.65616671, 2.65315230, 2.65013788, 2.64712347, 2.64410906, 2.64109466, 2.63808024, 2.63506583, 2.63205142,
		2.62903701, 2.62602260, 2.62300819, 2.61999378, 2.61697937, 2.61396496, 2.61095055, 2.60793614, 2.60492172, 2.60190731, 2.59889291, 2.59587849, 2.59286408, 2.58984967, 2.58683526, 2.58382085,
		2.58080644, 2.57779203, 2.57477762, 2.57176321, 2.56874880, 2.56573439, 2.56271998, 2.55970557, 2.55669116, 2.55367675, 2.55066233, 2.54764792, 2.54463351, 2.54161910, 2.53860469, 2.53559028,
		2.53257587, 2.52956146, 2.52654705, 2.52353264, 2.52051823, 2.51750382, 2.51448941, 2.51147500, 2.50846059, 2.50544618, 2.50243176, 2.49941735, 2.49640294, 2.49338854, 2.49037412, 2.48735971,
		2.48434530, 2.48133089, 2.47831648, 2.47530207, 2.47228766, 2.46927325, 2.46625884, 2.46324443, 2.46023002, 2.45721561, 2.45420120, 2.45118679, 2.44817238, 2.44515797, 2.44214356, 2.43912915,
		2.43611474, 2.43310033, 2.43008592, 2.42707150, 2.42405710, 2.42104269, 2.41802827, 2.41501387, 2.41199945, 2.40898505, 2.40597063, 2.40295622, 2.39994182, 2.39692740, 2.39391299, 2.39089858,
		2.38788417, 2.38486976, 2.38185535, 2.37884094, 2.37582653, 2.37281212, 2.36979771, 2.36678330, 2.36376889, 2.36075448, 2.35774007, 2.35472566, 2.35171125, 2.34869684, 2.34568243, 2.34266802,
		2.33965361, 2.33663920, 2.33362479, 2.33061038, 2.32759597, 2.32458156, 2.32156715, 2.31855274, 2.31553833, 2.31252392, 2.30950951, 2.30649510, 2.30348069, 2.30046628, 2.29745187, 2.29443746,
		2.29142305, 2.28840864, 2.28539423, 2.28237982, 2.27936541, 2.27635100, 2.27333659, 2.27032218, 2.26730777, 2.26429336, 2.26127894, 2.25826454, 2.25525013, 2.25223572, 2.24922131, 2.24620690,
		2.24319249, 2.24017808, 2.23716367, 2.23414925, 2.23113484, 2.22812043, 2.22510603, 2.22209161, 2.21907721, 2.21606280, 2.21304839, 2.21003398, 2.20701957, 2.20400516, 2.20099075, 2.19797634,
		2.19496193, 2.19194752, 2.18893311, 2.18591870, 2.18290429, 2.17988988, 2.17687547, 2.17386106, 2.17084665, 2.16783224, 2.16481783, 2.16180342, 2.15878901, 2.15577460, 2.15276019, 2.14974577,
		2.14673137, 2.14371696, 2.14070255, 2.13768814, 2.13467373, 2.13165932, 2.12864491, 2.12563050, 2.12261609, 2.11960168, 2.11658727, 2.11357286, 2.11055845, 2.10754404, 2.10452963, 2.10151522,
		2.09850081, 2.09548640, 2.09247199, 2.08945758, 2.08644317, 2.08342876, 2.08041435, 2.07739994, 2.07438553, 2.07137112, 2.06835671, 2.06534230, 2.06232789, 2.05931348, 2.05629907, 2.05328466,
		2.05027025, 2.04725584, 2.04424143, 2.04122703, 2.03821261, 2.03519820, 2.03218380, 2.02916939, 2.02615498, 2.02314057, 2.02012616, 2.01711175, 2.01409734, 2.01108293, 2.00806852, 2.00505411,
		2.00203970, 1.99902529, 1.99601088, 1.99299647, 1.98998206, 1.98696765, 1.98395324, 1.98093883, 1.97792442, 1.97491001, 1.97189560, 1.96888119, 1.96586678, 1.96285237, 1.95983797, 1.95682356,
		1.95380914, 1.95079474, 1.94778033, 1.94476592, 1.94175151, 1.93873710, 1.93572269, 1.93270828, 1.92969387, 1.92667946, 1.92366505, 1.92065064, 1.91763623, 1.91462182, 1.91160742, 1.90859300,
		1.90557859, 1.90256419, 1.89954978, 1.89653537, 1.89352096, 1.89050655, 1.88749214, 1.88447773, 1.88146332, 1.87844891, 1.87543450, 1.87242009, 1.86940568, 1.86639127, 1.86337686, 1.86036245,
		1.85734805, 1.85433363, 1.85131922, 1.84830481, 1.84529041, 1.84227600, 1.83926159, 1.83624718, 1.83323277, 1.83021836, 1.82720395, 1.82418954, 1.82117513, 1.81816072, 1.81514631, 1.81213190,
		1.80911749, 1.80610309, 1.80308867, 1.80007427, 1.79705986, 1.79404545, 1.79103104, 1.78801663, 1.78500222, 1.78198781, 1.77897340, 1.77595899, 1.77294458, 1.76993018, 1.76691576, 1.76390135,
		1.76088695, 1.75787254, 1.75485813, 1.75184372, 1.74882931, 1.74581490, 1.74280049, 1.73978608, 1.73677167, 1.73375727, 1.73074286, 1.72772845, 1.72471404, 1.72169963, 1.71868522, 1.71567081,
		1.71265640, 1.70964199, 1.70662758, 1.70361317, 1.70059876, 1.69758435, 1.69456994, 1.69155554, 1.68854113, 1.68552672, 1.68251231, 1.67949790, 1.67648349, 1.67346908, 1.67045467, 1.66744026,
		1.66442585, 1.66141144, 1.65839703, 1.65538263, 1.65236822, 1.64935381, 1.64633940, 1.64332499, 1.64031058, 1.63729617, 1.63428176, 1.63126735, 1.62825294, 1.62523854, 1.62222412, 1.61920972,
		1.61619531, 1.61318090, 1.61016649, 1.60715208, 1.60413767, 1.60112326, 1.59810885, 1.59509444, 1.59208004, 1.58906562, 1.58605122, 1.58303681, 1.58002240, 1.57700799, 1.57399358, 1.57097918,
		1.56796476, 1.56495036, 1.56193595, 1.55892154, 1.55590713, 1.55289272, 1.54987831, 1.54686391, 1.54384949, 1.54083508, 1.53782068, 1.53480627, 1.53179186, 1.52877745, 1.52576304, 1.52274863,
		1.51973422, 1.51671981, 1.51370540, 1.51069100, 1.50767659, 1.50466218, 1.50164777, 1.49863336, 1.49561895, 1.49260454, 1.48959014, 1.48657572, 1.48356132, 1.48054691, 1.47753250, 1.47451809,
		1.47150368, 1.46848927, 1.46547486, 1.46246046, 1.45944605, 1.45643164, 1.45341723, 1.45040282, 1.44738841, 1.44437400, 1.44135960, 1.43834519, 1.43533078, 1.43231637, 1.42930196, 1.42628755,
		1.42327314, 1.42025873, 1.41724432, 1.41422992, 1.41121551, 1.40820110, 1.40518669, 1.40217228, 1.39915787, 1.39614346, 1.39312906, 1.39011465, 1.38710024, 1.38408583, 1.38107142, 1.37805701,
		1.37504260, 1.37202820, 1.36901379, 1.36599938, 1.36298497, 1.35997056, 1.35695615, 1.35394174, 1.35092733, 1.34791293, 1.34489852, 1.34188411, 1.33886970, 1.33585529, 1.33284088, 1.32982648,
		1.32681207, 1.32379766, 1.32078325, 1.31776884, 1.31475443, 1.31174002, 1.30872561, 1.30571121, 1.30269680, 1.29968239, 1.29666798, 1.29365357, 1.29063916, 1.28762476, 1.28461035, 1.28159594,
		1.27858153, 1.27556712, 1.27255271, 1.26953830, 1.26652390, 1.26350949, 1.26049508, 1.25748067, 1.25446626, 1.25145185, 1.24843745, 1.24542304, 1.24240863, 1.23939422, 1.23637981, 1.23336540,
		1.23035100, 1.22733659, 1.22432218, 1.22130777, 1.21829336, 1.21527895, 1.21226454, 1.20925013, 1.20623573, 1.20322132, 1.20020691, 1.19719250, 1.19417809, 1.19116368, 1.18814928, 1.18513487,
		1.18212046, 1.17910605, 1.17609164, 1.17307723, 1.17006283, 1.16704842, 1.16403401, 1.16101960, 1.15800519, 1.15499078, 1.15197638, 1.14896197, 1.14594756, 1.14293315, 1.13991874, 1.13690434,
		1.13388993, 1.13087552, 1.12786111, 1.12484670, 1.12183229, 1.11881789, 1.11580348, 1.11278907, 1.10977466, 1.10676025, 1.10374584, 1.10073143, 1.09771703, 1.09470262, 1.09168821, 1.08867380,
		1.08565939, 1.08264499, 1.07963058, 1.07661617, 1.07360176, 1.07058735, 1.06757294, 1.06455854, 1.06154413, 1.05852972, 1.05551531, 1.05250090, 1.04948650, 1.04647209, 1.04345768, 1.04044327,
		1.03742886, 1.03441445, 1.03140005, 1.02838564, 1.02537123, 1.02235682, 1.01934242, 1.01632801, 1.01331360, 1.01029919, 1.00728478, 1.00427037, 1.00125597, 0.99824156, 0.99522715, 0.99221274,
		0.98919833, 0.98618393, 0.98316952, 0.98015511, 0.97714070, 0.97412629, 0.97111188, 0.96809748, 0.96508307, 0.96206866, 0.95905425, 0.95603985, 0.95302544, 0.95001103, 0.94699662, 0.94398222,
		0.94096780, 0.93795340, 0.93493899, 0.93192458, 0.92891017, 0.92589577, 0.92288136, 0.91986695, 0.91685254, 0.91383813, 0.91082372, 0.90780932, 0.90479491, 0.90178050, 0.89876609, 0.89575168,
		0.89273728, 0.88972287, 0.88670846, 0.88369405, 0.88067964, 0.87766524, 0.87465083, 0.87163642, 0.86862201, 0.86560760, 0.86259320, 0.85957879, 0.85656438, 0.85354997, 0.85053557, 0.84752116,
		0.84450675, 0.84149234, 0.83847794, 0.83546353, 0.83244912, 0.82943471, 0.82642030, 0.82340590, 0.82039149, 0.81737708, 0.81436267, 0.81134826, 0.80833385, 0.80531945, 0.80230504, 0.79929063,
		0.79627622, 0.79326182, 0.79024741, 0.78723300, 0.78421859, 0.78120418, 0.77818978, 0.77517537, 0.77216096, 0.76914656, 0.76613214, 0.76311774, 0.76010333, 0.75708892, 0.75407452, 0.75106011,
		0.74804570, 0.74503129, 0.74201688, 0.73900248, 0.73598807, 0.73297366, 0.72995925, 0.72694484, 0.72393044, 0.72091603, 0.71790162, 0.71488721, 0.71187281, 0.70885840, 0.70584399, 0.70282958,
		0.69981517, 0.69680077, 0.69378636, 0.69077195, 0.68775754, 0.68474313, 0.68172873, 0.67871432, 0.67569991, 0.67268551, 0.66967110, 0.66665669, 0.66364228, 0.66062787, 0.65761347, 0.65459906,
		0.65158465, 0.64857024, 0.64555584, 0.64254143, 0.63952702, 0.63651262, 0.63349821, 0.63048380, 0.62746939, 0.62445498, 0.62144057, 0.61842617, 0.61541176, 0.61239735, 0.60938295, 0.60636854,
		0.60335413, 0.60033972, 0.59732531, 0.59431091, 0.59129650, 0.58828209, 0.58526769, 0.58225328, 0.57923887, 0.57622446, 0.57321005, 0.57019565, 0.56718124, 0.56416683, 0.56115242, 0.55813802,
		0.55512361, 0.55210920, 0.54909479, 0.54608039, 0.54306598, 0.54005157, 0.53703716, 0.53402276, 0.53100835, 0.52799394, 0.52497953, 0.52196513, 0.51895072, 0.51593631, 0.51292190, 0.50990750,
		0.50689309, 0.50387868, 0.50086427, 0.49784987, 0.49483546, 0.49182105, 0.48880664, 0.48579224, 0.48277783, 0.47976342, 0.47674901, 0.47373460, 0.47072020, 0.46770579, 0.46469139, 0.46167698,
		0.45866257, 0.45564816, 0.45263376, 0.44961935, 0.44660494, 0.44359053, 0.44057612, 0.43756172, 0.43454731, 0.43153291, 0.42851849, 0.42550409, 0.42248968, 0.41947527, 0.41646086, 0.41344646,
		0.41043205, 0.40741764, 0.40440323, 0.40138883, 0.39837442, 0.39536001, 0.39234560, 0.38933120, 0.38631679, 0.38330238, 0.38028798, 0.37727357, 0.37425916, 0.37124475, 0.36823035, 0.36521594,
		0.36220153, 0.35918712, 0.35617272, 0.35315831, 0.35014390, 0.34712950, 0.34411509, 0.34110068, 0.33808627, 0.33507187, 0.33205746, 0.32904305, 0.32602865, 0.32301424, 0.31999983, 0.31698542,
		0.31397102, 0.31095661, 0.30794220, 0.30492779, 0.30191339, 0.29889898, 0.29588457, 0.29287017, 0.28985576, 0.28684135, 0.28382694, 0.28081253, 0.27779813, 0.27478372, 0.27176931, 0.26875490,
		0.26574050, 0.26272609, 0.25971168, 0.25669728, 0.25368287, 0.25066846, 0.24765405, 0.24463965, 0.24162524, 0.23861083, 0.23559643, 0.23258202, 0.22956761, 0.22655321, 0.22353880, 0.22052439,
		0.21750998, 0.21449558, 0.21148117, 0.20846676, 0.20545235, 0.20243795, 0.19942354, 0.19640913, 0.19339473, 0.19038032, 0.18736591, 0.18435151, 0.18133710, 0.17832269, 0.17530828, 0.17229387,
		0.16927947, 0.16626506, 0.16325065, 0.16023625, 0.15722184, 0.15420743, 0.15119302, 0.14817862, 0.14516421, 0.14214980, 0.13913540, 0.13612099, 0.13310658, 0.13009218, 0.12707777, 0.12406336,
		0.12104895, 0.11803455, 0.11502014, 0.11200573, 0.10899133, 0.10597692, 0.10296251, 0.09994810, 0.09693370, 0.09391929, 0.09090488, 0.08789047, 0.08487607, 0.08186166, 0.07884726, 0.07583285,
		0.07281844, 0.06980403, 0.06678962, 0.06377522, 0.06076081, 0.05774640, 0.05473200, 0.05171759, 0.04870318, 0.04568878, 0.04267437, 0.03965996, 0.03664556, 0.03363115, 0.03061674, 0.02760233,
		0.02458793, 0.02157352, 0.01855912, 0.01554471, 0.01253030, 0.00951589, 0.00650149, 0.00348708, 0.00047267, -0.00254174, -0.00555614, -0.00857055, -0.01158496, -0.01459936, -0.01761377, -0.02062818,
		-0.02364259, -0.02665699, -0.02967140, -0.03268581, -0.03570021, -0.03871462, -0.04172903, -0.04474343, -0.04775784, -0.05077225, -0.05378665, -0.05680106, -0.05981547, -0.06282988, -0.06584428, -
		0.06885869, -0.07187309, -0.07488750, -0.07790191, -0.08091632, -0.08393072, -0.08694513, -0.08995954, -0.09297394, -0.09598835, -0.09900276, -0.10201716, -0.10503157, -0.10804598, -0.11106039, -
		0.11407479, -0.11708920, -0.12010361, -0.12311802, -0.12613242, -0.12914683, -0.13216124, -0.13517564, -0.13819005, -0.14120446, -0.14421886, -0.14723327, -0.15024768, -0.15326208, -0.15627649, -
		0.15929090, -0.16230530, -0.16531971, -0.16833412, -0.17134852, -0.17436293, -0.17737734, -0.18039174, -0.18340615, -0.18642056, -0.18943497, -0.19244937, -0.19546378, -0.19847819, -0.20149259, -
		0.20450700, -0.20752141, -0.21053581, -0.21355022, -0.21656463, -0.21957903, -0.22259344, -0.22560785, -0.22862226, -0.23163666, -0.23465107, -0.23766547, -0.24067988, -0.24369429, -0.24670870, -
		0.24972310, -0.25273751, -0.25575192, -0.25876632, -0.26178073, -0.26479514, -0.26780954, -0.27082395, -0.27383836, -0.27685276, -0.27986717, -0.28288158, -0.28589598, -0.28891039, -0.29192480, -
		0.29493920, -0.29795361, -0.30096802, -0.30398242, -0.30699683, -0.31001124, -0.31302564, -0.31604005, -0.31905446, -0.32206887, -0.32508327, -0.32809768, -0.33111209, -0.33412649, -0.33714090, -
		0.34015531, -0.34316971, -0.34618412, -0.34919853, -0.35221293, -0.35522734, -0.35824175, -0.36125615, -0.36427056, -0.36728497, -0.37029938, -0.37331378, -0.37632819, -0.37934259, -0.38235700, -
		0.38537141, -0.38838581, -0.39140022, -0.39441463, -0.39742903, -0.40044344, -0.40345785, -0.40647225, -0.40948666, -0.41250107, -0.41551547, -0.41852988, -0.42154429, -0.42455869, -0.42757310, -
		0.43058751, -0.43360192, -0.43661632, -0.43963073, -0.44264513, -0.44565954, -0.44867395, -0.45168835, -0.45470276, -0.45771717, -0.46073158, -0.46374598, -0.46676039, -0.46977479, -0.47278920, -
		0.47580361, -0.47881802, -0.48183242, -0.48484683, -0.48786124, -0.49087564, -0.49389005, -0.49690446, -0.49991886, -0.50293327, -0.50594768, -0.50896208, -0.51197649, -0.51499090, -0.51800530, -
		0.52101971, -0.52403412, -0.52704852, -0.53006293, -0.53307733, -0.53609174, -0.53910615, -0.54212055, -0.54513496, -0.54814937, -0.55116377, -0.55417818, -0.55719259, -0.56020700, -0.56322140, -
		0.56623581, -0.56925021, -0.57226462, -0.57527903, -0.57829344, -0.58130784, -0.58432225, -0.58733665, -0.59035106, -0.59336547, -0.59637988, -0.59939428, -0.60240869, -0.60542310, -0.60843750, -
		0.61145191, -0.61446631, -0.61748072, -0.62049513, -0.62350953, -0.62652394, -0.62953835, -0.63255275, -0.63556716, -0.63858157, -0.64159598, -0.64461038, -0.64762479, -0.65063919, -0.65365360, -
		0.65666801, -0.65968242, -0.66269682, -0.66571123, -0.66872564, -0.67174004, -0.67475445, -0.67776885, -0.68078326, -0.68379767, -0.68681207, -0.68982648, -0.69284089, -0.69585529, -0.69886970, -
		0.70188411, -0.70489851, -0.70791292, -0.71092733, -0.71394173, -0.71695614, -0.71997055, -0.72298495, -0.72599936, -0.72901377, -0.73202817, -0.73504258, -0.73805699, -0.74107139, -0.74408580, -
		0.74710020, -0.75011461, -0.75312902, -0.75614343, -0.75915783, -0.76217224, -0.76518665, -0.76820105, -0.77121546, -0.77422986, -0.77724427, -0.78025868, -0.78327308, -0.78628749, -0.78930190, -
		0.79231630, -0.79533071, -0.79834511, -0.80135952, -0.80437393, -0.80738834, -0.81040274, -0.81341715, -0.81643156, -0.81944596, -0.82246037, -0.82547478, -0.82848918, -0.83150359, -0.83451799, -
		0.83753240, -0.84054681, -0.84356121, -0.84657562, -0.84959003, -0.85260443, -0.85561884, -0.85863325, -0.86164765, -0.86466206, -0.86767647, -0.87069087, -0.87370528, -0.87671969, -0.87973409, -
		0.88274850, -0.88576291, -0.88877731, -0.89179172, -0.89480612, -0.89782053, -0.90083494, -0.90384934, -0.90686375, -0.90987816, -0.91289256, -0.91590697, -0.91892138, -0.92193578, -0.92495019, -
		0.92796460, -0.93097900, -0.93399341, -0.93700782, -0.94002222, -0.94303663, -0.94605104, -0.94906544, -0.95207985, -0.95509425, -0.95810866, -0.96112307, -0.96413747, -0.96715188, -0.97016629, -
		0.97318069, -0.97619510, -0.97920950, -0.98222391, -0.98523832, -0.98825273, -0.99126713, -0.99428154, -0.99729594, -1.00031035, -1.00332476, -1.00633916, -1.00935357, -1.01236798, -1.01538238, -
		1.01839679, -1.02141120, -1.02442560, -1.02744001, -1.03045442, -1.03346882, -1.03648323, -1.03949763, -1.04251204, -1.04552645, -1.04854085, -1.05155526, -1.05456967, -1.05758408, -1.06059848, -
		1.06361289, -1.06662729, -1.06964170, -1.07265611, -1.07567051, -1.07868492, -1.08169933, -1.08471373, -1.08772814, -1.09074255, -1.09375695, -1.09677136, -1.09978576, -1.10280017, -1.10581458, -
		1.10882898, -1.11184339, -1.11485780, -1.11787220, -1.12088661, -1.12390102, -1.12691542, -1.12992983, -1.13294424, -1.13595864, -1.13897305, -1.14198746, -1.14500186, -1.14801627, -1.15103067, -
		1.15404508, -1.15705949, -1.16007389, -1.16308830, -1.16610271, -1.16911711, -1.17213152, -1.17514593, -1.17816033, -1.18117474, -1.18418915, -1.18720355, -1.19021796, -1.19323237, -1.19624677, -
		1.19926118, -1.20227558, -1.20528999, -1.20830440, -1.21131880, -1.21433321, -1.21734762, -1.22036202, -1.22337643, -1.22639083, -1.22940524, -1.23241965, -1.23543405, -1.23844846, -1.24146287, -
		1.24447727, -1.24749168, -1.25050608, -1.25352049, -1.25653490, -1.25954930, -1.26256371, -1.26557812, -1.26859252, -1.27160693, -1.27462134, -1.27763574, -1.28065015, -1.28366456, -1.28667896, -
		1.28969337, -1.29270778, -1.29572218, -1.29873659, -1.30175099, -1.30476540, -1.30777981, -1.31079421, -1.31380862, -1.31682303, -1.31983743, -1.32285184, -1.32586625, -1.32888065, -1.33189506, -
		1.33490946, -1.33792387, -1.34093828, -1.34395269, -1.34696709, -1.34998150, -1.35299590, -1.35601031, -1.35902472, -1.36203912, -1.36505353, -1.36806794, -1.37108234, -1.37409675, -1.37711115, -
		1.38012556, -1.38313997, -1.38615437, -1.38916878, -1.39218319, -1.39519759, -1.39821200, -1.40122641, -1.40424081, -1.40725522, -1.41026963, -1.41328403, -1.41629844, -1.41931284, -1.42232725, -
		1.42534166, -1.42835606, -1.43137047, -1.43438488, -1.43739928, -1.44041369, -1.44342810, -1.44644250, -1.44945691, -1.45247131, -1.45548572, -1.45850013, -1.46151453, -1.46452894, -1.46754335, -
		1.47055775, -1.47357216, -1.47658657, -1.47960097, -1.48261538, -1.48562979, -1.48864419, -1.49165860, -1.49467300, -1.49768741, -1.50070182, -1.50371622, -1.50673063, -1.50974504, -1.51275944, -
		1.51577385, -1.51878826, -1.52180266, -1.52481707, -1.52783148, -1.53084588, -1.53386029, -1.53687469, -1.53988910, -1.54290351, -1.54591791, -1.54893232, -1.55194673, -1.55496113, -1.55797554, -
		1.56098995, -1.56400435, -1.56701876, -1.57003316, -1.57304757, -1.57606198, -1.57907638, -1.58209079, -1.58510520, -1.58811960, -1.59113401, -1.59414842, -1.59716282, -1.60017723, -1.60319163, -
		1.60620604, -1.60922045, -1.61223486, -1.61524926, -1.61826367, -1.62127807, -1.62429248, -1.62730689, -1.63032129, -1.63333570, -1.63635010, -1.63936451, -1.64237892, -1.64539333, -1.64840773, -
		1.65142214, -1.65443654, -1.65745095, -1.66046536, -1.66347976, -1.66649417, -1.66950858, -1.67252298, -1.67553739, -1.67855179, -1.68156620, -1.68458061, -1.68759501, -1.69060942, -1.69362383, -
		1.69663823, -1.69965264, -1.70266704, -1.70568145, -1.70869586, -1.71171026, -1.71472467, -1.71773908, -1.72075349, -1.72376789, -1.72678230, -1.72979671, -1.73281111, -1.73582552, -1.73883992, -
		1.74185433, -1.74486874, -1.74788314, -1.75089755, -1.75391195, -1.75692636, -1.75994077, -1.76295517, -1.76596958, -1.76898399, -1.77199839, -1.77501280, -1.77802720, -1.78104161, -1.78405602, -
		1.78707043, -1.79008483, -1.79309924, -1.79611364, -1.79912805, -1.80214246, -1.80515687, -1.80817127, -1.81118568, -1.81420008, -1.81721449, -1.82022890, -1.82324330, -1.82625771, -1.82927211, -
		1.83228652, -1.83530093, -1.83831533, -1.84132974, -1.84434415, -1.84735855, -1.85037296, -1.85338737, -1.85640177, -1.85941618, -1.86243059, -1.86544499, -1.86845940, -1.87147380, -1.87448821, -
		1.87750262, -1.88051703, -1.88353143, -1.88654584, -1.88956024, -1.89257465, -1.89558906, -1.89860346, -1.90161787, -1.90463227, -1.90764668, -1.91066109, -1.91367549, -1.91668990, -1.91970430, -
		1.92271871, -1.92573312, -1.92874753, -1.93176193, -1.93477634, -1.93779075, -1.94080515, -1.94381956, -1.94683396, -1.94984837, -1.95286278, -1.95587718, -1.95889159, -1.96190600, -1.96492040, -
		1.96793481, -1.97094922, -1.97396362, -1.97697803, -1.97999244, -1.98300684, -1.98602125, -1.98903565, -1.99205006, -1.99506447, -1.99807887, -2.00109328, -2.00410769, -2.00712209, -2.01013650, -
		2.01315091, -2.01616531, -2.01917972, -2.02219413, -2.02520853, -2.02822294, -2.03123734, -2.03425175, -2.03726616, -2.04028056, -2.04329497, -2.04630938, -2.04932378, -2.05233819, -2.05535260, -
		2.05836700, -2.06138141, -2.06439581, -2.06741022, -2.07042463, -2.07343903, -2.07645344, -2.07946785, -2.08248225, -2.08549666, -2.08851107, -2.09152547, -2.09453988, -2.09755429, -2.10056869, -
		2.10358310, -2.10659751, -2.10961191, -2.11262632, -2.11564073, -2.11865513, -2.12166954, -2.12468394, -2.12769835, -2.13071276, -2.13372716, -2.13674157, -2.13975598, -2.14277038, -2.14578479, -
		2.14879920, -2.15181360, -2.15482801, -2.15784241, -2.16085682, -2.16387123, -2.16688564, -2.16990004, -2.17291445, -2.17592885, -2.17894326, -2.18195767, -2.18497207, -2.18798648, -2.19100089, -
		2.19401529, -2.19702970, -2.20004410, -2.20305851, -2.20607292, -2.20908733, -2.21210173, -2.21511614, -2.21813054, -2.22114495, -2.22415936, -2.22717376, -2.23018817, -2.23320258, -2.23621698, -
		2.23923139, -2.24224580, -2.24526020, -2.24827461, -2.25128902, -2.25430342, -2.25731783, -2.26033223, -2.26334664, -2.26636105, -2.26937545, -2.27238986, -2.27540427, -2.27841867, -2.28143308, -
		2.28444749, -2.28746189, -2.29047630, -2.29349070, -2.29650511, -2.29951952, -2.30253392, -2.30554833, -2.30856274, -2.31157714, -2.31459155, -2.31760596, -2.32062037, -2.32363477, -2.32664918, -
		2.32966358, -2.33267799, -2.33569239, -2.33870680, -2.34172121, -2.34473562, -2.34775002, -2.35076443, -2.35377884, -2.35679324, -2.35980765, -2.36282205, -2.36583646, -2.36885087, -2.37186527, -
		2.37487968, -2.37789409, -2.38090849, -2.38392290, -2.38693731, -2.38995171, -2.39296612, -2.39598053, -2.39899493, -2.40200934, -2.40502375, -2.40803815, -2.41105256, -2.41406697, -2.41708137, -
		2.42009578, -2.42311019, -2.42612459, -2.42913900, -2.43215340, -2.43516781, -2.43818222, -2.44119662, -2.44421103, -2.44722544, -2.45023984, -2.45325425, -2.45626866, -2.45928306, -2.46229747, -
		2.46531188, -2.46832628, -2.47134069, -2.47435510, -2.47736950, -2.48038391, -2.48339832, -2.48641272, -2.48942713, -2.49244154, -2.49545594, -2.49847035, -2.50148475, -2.50449916, -2.50751357, -
		2.51052797, -2.51354238, -2.51655679, -2.51957120, -2.52258560, -2.52560001, -2.52861442, -2.53162882, -2.53464323, -2.53765763, -2.54067204, -2.54368645, -2.54670085, -2.54971526, -2.55272967, -
		2.55574407, -2.55875848, -2.56177289, -2.56478729, -2.56780170, -2.57081611, -2.57383051, -2.57684492, -2.57985932, -2.58287373, -2.58588814, -2.58890255, -2.59191695, -2.59493136, -2.59794576, -
		2.60096017, -2.60397458, -2.60698898, -2.61000339, -2.61301780, -2.61603220, -2.61904661, -2.62206102, -2.62507542, -2.62808983, -2.63110424, -2.63411864, -2.63713305, -2.64014746, -2.64316186, -
		2.64617627, -2.64919068, -2.65220508, -2.65521949, -2.65823390, -2.66124830, -2.66426271, -2.66727712, -2.67029152, -2.67330593, -2.67632034, -2.67933474, -2.68234915, -2.68536355, -2.68837796, -
		2.69139237, -2.69440678, -2.69742118, -2.70043559, -2.70344999, -2.70646440, -2.70947881, -2.71249322, -2.71550762, -2.71852203, -2.72153643, -2.72455084, -2.72756525, -2.73057965, -2.73359406, -
		2.73660847, -2.73962288, -2.74263728, -2.74565169, -2.74866609, -2.75168050, -2.75469491, -2.75770931, -2.76072372, -2.76373813, -2.76675253, -2.76976694, -2.77278135, -2.77579575, -2.77881016, -
		2.78182457, -2.78483897, -2.78785338, -2.79086779, -2.79388219, -2.79689660, -2.79991101, -2.80292541, -2.80593982, -2.80895423, -2.81196863, -2.81498304, -2.81799745, -2.82101185, -2.82402626, -
		2.82704067, -2.83005507, -2.83306948, -2.83608389, -2.83909829, -2.84211270, -2.84512711, -2.84814151, -2.85115592, -2.85417033, -2.85718473, -2.86019914, -2.86321355, -2.86622795, -2.86924236, -
		2.87225677, -2.87527117, -2.87828558, -2.88129999, -2.88431439, -2.88732880, -2.89034321, -2.89335761, -2.89637202, -2.89938643, -2.90240083, -2.90541524, -2.90842965, -2.91144405, -2.91445846, -
		2.91747287, -2.92048727, -2.92350168, -2.92651609, -2.92953049, -2.93254490, -2.93555931, -2.93857371, -2.94158812, -2.94460253, -2.94761693, -2.95063134, -2.95364575, -2.95666015, -2.95967456, -
		2.96268897, -2.96570337, -2.96871778, -2.97173219, -2.97474659, -2.97776100, -2.98077541, -2.98378981, -2.98680422, -2.98981863, -2.99283303, -2.99584744, -2.99886185, -3.00187625, -3.00489066, -
		3.00790507, -3.01091947, -3.01393388, -3.01694829, -3.01996269, -3.02297710, -3.02599151, -3.02900591, -3.03202032, -3.03503473, -3.03804913, -3.04106354, -3.04407795, -3.04709235, -3.05010676, -
		3.05312117, -3.05613558, -3.05914998, -3.06216439, -3.06517880, -3.06819320, -3.07120761, -3.07422202, -3.07723642, -3.08025083, -3.08326524, -3.08627964, -3.08929405, -3.09230846, -3.09532286, -
		3.09833727, -3.10135168, -3.10436608, -3.10738049, -3.11039490, -3.11340930, -3.11642371, -3.11943812, -3.12245252, -3.12546693, -3.12848134, -3.13149574, -3.13451015, -3.13752456, -3.14053896, -
		3.14355337, -3.14656778, -3.14958218, -3.15259659, -3.15561100, -3.15862540, -3.16163981, -3.16465422, -3.16766862, -3.17068303, -3.17369744, -3.17671184, -3.17972625, -3.18274066, -3.18575507, -
		3.18876947, -3.19178388, -3.19479828, -3.19781269, -3.20082710, -3.20384150, -3.20685591, -3.20987032, -3.21288473, -3.21589913, -3.21891354, 3.17993632, 3.17691881, 3.17390130, 3.17088379,
		3.16786627, 3.16484876, 3.16183125, 3.15881374, 3.15579623, 3.15277872, 3.14976121, 3.14674370, 3.14372618, 3.14070867, 3.13769116, 3.13467365, 3.13165613, 3.12863862, 3.12562111, 3.12260359,
		3.11958609, 3.11656858, 3.11355106, 3.11053355, 3.10751604, 3.10449853, 3.10148101, 3.09846350, 3.09544599, 3.09242848, 3.08941097, 3.08639346, 3.08337595, 3.08035844, 3.07734092, 3.07432341,
		3.07130590, 3.06828839, 3.06527087, 3.06225337, 3.05923585, 3.05621834, 3.05320083, 3.05018332, 3.04716581, 3.04414829, 3.04113079, 3.03811327, 3.03509576, 3.03207825, 3.02906074, 3.02604323,
		3.02302572, 3.02000821, 3.01699069, 3.01397318, 3.01095567, 3.00793816, 3.00492065, 3.00190314, 2.99888562, 2.99586812, 2.99285060, 2.98983309, 2.98681558, 2.98379807, 2.98078056, 2.97776304,
		2.97474553, 2.97172802, 2.96871051, 2.96569300, 2.96267549, 2.95965798, 2.95664047, 2.95362295, 2.95060545, 2.94758794, 2.94457042, 2.94155291, 2.93853540, 2.93551789, 2.93250038, 2.92948287,
		2.92646536, 2.92344784, 2.92043033, 2.91741282, 2.91439531, 2.91137780, 2.90836029, 2.90534278, 2.90232526, 2.89930776, 2.89629024, 2.89327273, 2.89025522, 2.88723771, 2.88422020, 2.88120269,
		2.87818518, 2.87516767, 2.87215016, 2.86913265, 2.86611513, 2.86309762, 2.86008011, 2.85706260, 2.85404509, 2.85102758, 2.84801007, 2.84499256, 2.84197504, 2.83895753, 2.83594002, 2.83292251,
		2.82990500, 2.82688749, 2.82386998, 2.82085247, 2.81783496, 2.81481745, 2.81179994, 2.80878243, 2.80576491, 2.80274740, 2.79972989, 2.79671238, 2.79369487, 2.79067736, 2.78765985, 2.78464234,
		2.78162483, 2.77860732, 2.77558981, 2.77257230, 2.76955479, 2.76653727, 2.76351976, 2.76050225, 2.75748474, 2.75446723, 2.75144972, 2.74843221, 2.74541470, 2.74239719, 2.73937968, 2.73636217,
		2.73334466, 2.73032714, 2.72730964, 2.72429212, 2.72127461, 2.71825710, 2.71523959, 2.71222208, 2.70920457, 2.70618706, 2.70316955, 2.70015204, 2.69713453, 2.69411702, 2.69109951, 2.68808200,
		2.68506449, 2.68204698, 2.67902946, 2.67601196, 2.67299444, 2.66997694, 2.66695942, 2.66394191, 2.66092440, 2.65790690, 2.65488938, 2.65187187, 2.64885436, 2.64583685, 2.64281934, 2.63980183,
		2.63678432, 2.63376681, 2.63074930, 2.62773179, 2.62471428, 2.62169677, 2.61867926, 2.61566174, 2.61264424, 2.60962672, 2.60660922, 2.60359171, 2.60057420, 2.59755669, 2.59453917, 2.59152166,
		2.58850415, 2.58548664, 2.58246913, 2.57945162, 2.57643411, 2.57341660, 2.57039909, 2.56738158, 2.56436407, 2.56134656, 2.55832905, 2.55531154, 2.55229403, 2.54927652, 2.54625901, 2.54324150,
		2.54022399, 2.53720648, 2.53418897, 2.53117146, 2.52815395, 2.52513644, 2.52211893, 2.51910142, 2.51608390, 2.51306640, 2.51004889, 2.50703138, 2.50401387, 2.50099636, 2.49797885, 2.49496134,
		2.49194383, 2.48892632, 2.48590881, 2.48289130, 2.47987378, 2.47685628, 2.47383876, 2.47082126, 2.46780375, 2.46478624, 2.46176873, 2.45875122, 2.45573371, 2.45271619, 2.44969869, 2.44668118,
		2.44366367, 2.44064615, 2.43762865, 2.43461113, 2.43159362, 2.42857611, 2.42555860, 2.42254110, 2.41952359, 2.41650608, 2.41348857, 2.41047106, 2.40745354, 2.40443603, 2.40141852, 2.39840102,
		2.39538351, 2.39236600, 2.38934849, 2.38633098, 2.38331347, 2.38029596, 2.37727845, 2.37426094, 2.37124343, 2.36822592, 2.36520841, 2.36219090, 2.35917339, 2.35615588, 2.35313837, 2.35012086,
		2.34710335, 2.34408584, 2.34106833, 2.33805082, 2.33503331, 2.33201580, 2.32899829, 2.32598078, 2.32296327, 2.31994576, 2.31692825, 2.31391074, 2.31089323, 2.30787572, 2.30485821, 2.30184070,
		2.29882319, 2.29580568, 2.29278817, 2.28977066, 2.28675316, 2.28373564, 2.28071813, 2.27770062, 2.27468311, 2.27166560, 2.26864810, 2.26563059, 2.26261308, 2.25959557, 2.25657806, 2.25356055,
		2.25054304, 2.24752553, 2.24450802, 2.24149051, 2.23847300, 2.23545549, 2.23243798, 2.22942047, 2.22640296, 2.22338545, 2.22036794, 2.21735043, 2.21433292, 2.21131542, 2.20829791, 2.20528040,
		2.20226288, 2.19924538, 2.19622787, 2.19321036, 2.19019284, 2.18717533, 2.18415783, 2.18114032, 2.17812281, 2.17510530, 2.17208779, 2.16907028, 2.16605277, 2.16303526, 2.16001775, 2.15700024,
		2.15398273, 2.15096522, 2.14794771, 2.14493020, 2.14191269, 2.13889518, 2.13587767, 2.13286017, 2.12984266, 2.12682515, 2.12380764, 2.12079013, 2.11777262, 2.11475511, 2.11173760, 2.10872009,
		2.10570258, 2.10268507, 2.09966756, 2.09665006, 2.09363255, 2.09061504, 2.08759753, 2.08458002, 2.08156250, 2.07854500, 2.07552749, 2.07250998, 2.06949247, 2.06647496, 2.06345745, 2.06043994,
		2.05742244, 2.05440492, 2.05138741, 2.04836991, 2.04535240, 2.04233489, 2.03931738, 2.03629987, 2.03328236, 2.03026485, 2.02724734, 2.02422983, 2.02121232, 2.01819481, 2.01517731, 2.01215980,
		2.00914228, 2.00612477, 2.00310727, 2.00008976, 1.99707225, 1.99405474, 1.99103723, 1.98801972, 1.98500221, 1.98198470, 1.97896720, 1.97594968, 1.97293218, 1.96991467, 1.96689715, 1.96387965,
		1.96086214, 1.95784463, 1.95482712, 1.95180961, 1.94879210, 1.94577459, 1.94275708, 1.93973957, 1.93672207, 1.93370456, 1.93068705, 1.92766954, 1.92465203, 1.92163452, 1.91861701, 1.91559950,
		1.91258199, 1.90956448, 1.90654697, 1.90352946, 1.90051195, 1.89749445, 1.89447694, 1.89145943, 1.88844192, 1.88542441, 1.88240690, 1.87938940, 1.87637188, 1.87335438, 1.87033686, 1.86731935,
		1.86430185, 1.86128434, 1.85826683, 1.85524932, 1.85223181, 1.84921430, 1.84619679, 1.84317928, 1.84016178, 1.83714427, 1.83412676, 1.83110925, 1.82809174, 1.82507423, 1.82205672, 1.81903921,
		1.81602170, 1.81300419, 1.80998669, 1.80696918, 1.80395167, 1.80093416, 1.79791665, 1.79489914, 1.79188163, 1.78886412, 1.78584661, 1.78282911, 1.77981160, 1.77679409, 1.77377658, 1.77075907,
		1.76774156, 1.76472405, 1.76170655, 1.75868903, 1.75567153, 1.75265402, 1.74963651, 1.74661900, 1.74360149, 1.74058398, 1.73756647, 1.73454896, 1.73153145, 1.72851395, 1.72549644, 1.72247893,
		1.71946142, 1.71644391, 1.71342640, 1.71040890, 1.70739138, 1.70437388, 1.70135637, 1.69833886, 1.69532135, 1.69230384, 1.68928633, 1.68626882, 1.68325131, 1.68023380, 1.67721630, 1.67419879,
		1.67118128, 1.66816377, 1.66514626, 1.66212875, 1.65911124, 1.65609374, 1.65307623, 1.65005872, 1.64704121, 1.64402370, 1.64100619, 1.63798868, 1.63497117, 1.63195366, 1.62893615, 1.62591865,
		1.62290114, 1.61988363, 1.61686612, 1.61384861, 1.61083110, 1.60781359, 1.60479609, 1.60177858, 1.59876107, 1.59574356, 1.59272605, 1.58970854, 1.58669104, 1.58367353, 1.58065601, 1.57763851,
		1.57462100, 1.57160349, 1.56858598, 1.56556847, 1.56255097, 1.55953346, 1.55651595, 1.55349844, 1.55048093, 1.54746342, 1.54444591, 1.54142840, 1.53841090, 1.53539339, 1.53237588, 1.52935837,
		1.52634086, 1.52332335, 1.52030584, 1.51728834, 1.51427083, 1.51125332, 1.50823581, 1.50521830, 1.50220079, 1.49918328, 1.49616577, 1.49314827, 1.49013076, 1.48711325, 1.48409574, 1.48107823,
		1.47806072, 1.47504322, 1.47202571, 1.46900820, 1.46599069, 1.46297318, 1.45995567, 1.45693816, 1.45392066, 1.45090315, 1.44788564, 1.44486813, 1.44185062, 1.43883311, 1.43581561, 1.43279810,
		1.42978059, 1.42676308, 1.42374557, 1.42072806, 1.41771055, 1.41469304, 1.41167554, 1.40865803, 1.40564052, 1.40262301, 1.39960550, 1.39658799, 1.39357049, 1.39055298, 1.38753547, 1.38451796,
		1.38150045, 1.37848294, 1.37546543, 1.37244792, 1.36943042, 1.36641291, 1.36339540, 1.36037789, 1.35736038, 1.35434288, 1.35132537, 1.34830786, 1.34529035, 1.34227284, 1.33925533, 1.33623782,
		1.33322032, 1.33020281, 1.32718530, 1.32416779, 1.32115028, 1.31813278, 1.31511527, 1.31209776, 1.30908025, 1.30606274, 1.30304523, 1.30002772, 1.29701021, 1.29399271, 1.29097520, 1.28795769,
		1.28494018, 1.28192267, 1.27890517, 1.27588766, 1.27287015, 1.26985264, 1.26683513, 1.26381762, 1.26080011, 1.25778261, 1.25476510, 1.25174759, 1.24873008, 1.24571257, 1.24269506, 1.23967756,
		1.23666005, 1.23364254, 1.23062503, 1.22760752, 1.22459001, 1.22157250, 1.21855500, 1.21553749, 1.21251998, 1.20950247, 1.20648496, 1.20346745, 1.20044995, 1.19743244, 1.19441493, 1.19139742,
		1.18837991, 1.18536240, 1.18234490, 1.17932739, 1.17630988, 1.17329237, 1.17027486, 1.16725736, 1.16423985, 1.16122234, 1.15820483, 1.15518732, 1.15216981, 1.14915231, 1.14613480, 1.14311729,
		1.14009978, 1.13708227, 1.13406476, 1.13104726, 1.12802975, 1.12501224, 1.12199473, 1.11897722, 1.11595971, 1.11294221, 1.10992470, 1.10690719, 1.10388968, 1.10087217, 1.09785466, 1.09483716,
		1.09181965, 1.08880214, 1.08578463, 1.08276712, 1.07974962, 1.07673211, 1.07371460, 1.07069709, 1.06767958, 1.06466207, 1.06164457, 1.05862706, 1.05560955, 1.05259204, 1.04957453, 1.04655703,
		1.04353952, 1.04052201, 1.03750450, 1.03448699, 1.03146948, 1.02845197, 1.02543447, 1.02241696, 1.01939945, 1.01638194, 1.01336443, 1.01034692, 1.00732942, 1.00431191, 1.00129440, 0.99827689,
		0.99525939, 0.99224188, 0.98922437, 0.98620686, 0.98318935, 0.98017184, 0.97715434, 0.97413683, 0.97111932, 0.96810181, 0.96508430, 0.96206679, 0.95904929, 0.95603178, 0.95301427, 0.94999676,
		0.94697925, 0.94396175, 0.94094424, 0.93792673, 0.93490922, 0.93189171, 0.92887421, 0.92585670, 0.92283919, 0.91982168, 0.91680417, 0.91378666, 0.91076916, 0.90775165, 0.90473414, 0.90171663,
		0.89869912, 0.89568162, 0.89266411, 0.88964660, 0.88662909, 0.88361158, 0.88059407, 0.87757657, 0.87455906, 0.87154155, 0.86852404, 0.86550653, 0.86248903, 0.85947152, 0.85645401, 0.85343650,
		0.85041899, 0.84740149, 0.84438398, 0.84136647, 0.83834896, 0.83533145, 0.83231394, 0.82929644, 0.82627893, 0.82326142, 0.82024391, 0.81722640, 0.81420890, 0.81119139, 0.80817388, 0.80515637,
		0.80213886, 0.79912135, 0.79610385, 0.79308634, 0.79006883, 0.78705132, 0.78403382, 0.78101631, 0.77799880, 0.77498129, 0.77196378, 0.76894627, 0.76592877, 0.76291126, 0.75989375, 0.75687624,
		0.75385873, 0.75084123, 0.74782372, 0.74480621, 0.74178870, 0.73877119, 0.73575368, 0.73273618, 0.72971867, 0.72670116, 0.72368365, 0.72066614, 0.71764864, 0.71463113, 0.71161362, 0.70859611,
		0.70557860, 0.70256110, 0.69954359, 0.69652608, 0.69350857, 0.69049106, 0.68747356, 0.68445605, 0.68143854, 0.67842103, 0.67540352, 0.67238601, 0.66936851, 0.66635100, 0.66333349, 0.66031598,
		0.65729847, 0.65428097, 0.65126346, 0.64824595, 0.64522844, 0.64221093, 0.63919343, 0.63617592, 0.63315841, 0.63014090, 0.62712339, 0.62410589, 0.62108838, 0.61807087, 0.61505336, 0.61203585,
		0.60901835, 0.60600084, 0.60298333, 0.59996582, 0.59694831, 0.59393081, 0.59091330, 0.58789579, 0.58487828, 0.58186077, 0.57884327, 0.57582576, 0.57280825, 0.56979074, 0.56677323, 0.56375572,
		0.56073822, 0.55772071, 0.55470320, 0.55168569, 0.54866819, 0.54565068, 0.54263317, 0.53961566, 0.53659815, 0.53358064, 0.53056314, 0.52754563, 0.52452812, 0.52151061, 0.51849311, 0.51547560,
		0.51245809, 0.50944058, 0.50642307, 0.50340557, 0.50038806, 0.49737055, 0.49435304, 0.49133553, 0.48831802, 0.48530052, 0.48228301, 0.47926550, 0.47624799, 0.47323048, 0.47021298, 0.46719547,
		0.46417796, 0.46116045, 0.45814295, 0.45512544, 0.45210793, 0.44909042, 0.44607291, 0.44305540, 0.44003790, 0.43702039, 0.43400288, 0.43098537, 0.42796787, 0.42495036, 0.42193285, 0.41891534,
		0.41589783, 0.41288032, 0.40986282, 0.40684531, 0.40382780, 0.40081029, 0.39779278, 0.39477528, 0.39175777, 0.38874026, 0.38572275, 0.38270525, 0.37968774, 0.37667023, 0.37365272, 0.37063521,
		0.36761770, 0.36460020, 0.36158269, 0.35856518, 0.35554767, 0.35253016, 0.34951266, 0.34649515, 0.34347764, 0.34046013, 0.33744263, 0.33442512, 0.33140761, 0.32839010, 0.32537259, 0.32235508,
		0.31933758, 0.31632007, 0.31330256, 0.31028505, 0.30726754, 0.30425004, 0.30123253, 0.29821502, 0.29519751, 0.29218001, 0.28916250, 0.28614499, 0.28312748, 0.28010997, 0.27709246, 0.27407496,
		0.27105745, 0.26803994, 0.26502243, 0.26200493, 0.25898742, 0.25596991, 0.25295240, 0.24993489, 0.24691739, 0.24389988, 0.24088237, 0.23786486, 0.23484735, 0.23182985, 0.22881234, 0.22579483,
		0.22277732, 0.21975981, 0.21674231, 0.21372480, 0.21070729, 0.20768978, 0.20467227, 0.20165477, 0.19863726, 0.19561975, 0.19260224, 0.18958473, 0.18656723, 0.18354972, 0.18053221, 0.17751470,
		0.17449719, 0.17147969, 0.16846218, 0.16544467, 0.16242716, 0.15940965, 0.15639215, 0.15337464, 0.15035713, 0.14733962, 0.14432211, 0.14130461, 0.13828710, 0.13526959, 0.13225208, 0.12923457,
		0.12621707, 0.12319956, 0.12018205, 0.11716454, 0.11414703, 0.11112953, 0.10811202, 0.10509451, 0.10207700, 0.09905950, 0.09604199, 0.09302448, 0.09000697, 0.08698946, 0.08397195, 0.08095445,
		0.07793694, 0.07491943, 0.07190192, 0.06888441, 0.06586691, 0.06284940, 0.05983189, 0.05681438, 0.05379688, 0.05077937, 0.04776186, 0.04474435, 0.04172684, 0.03870934, 0.03569183, 0.03267432,
		0.02965681, 0.02663930, 0.02362180, 0.02060429, 0.01758678, 0.01456927, 0.01155176, 0.00853426, 0.00551675, 0.00249924, -0.00051827, -0.00353578, -0.00655328, -0.00957079, -0.01258830, -0.01560581, -
		0.01862332, -0.02164082, -0.02465833, -0.02767584, -0.03069335, -0.03371086, -0.03672836, -0.03974587, -0.04276338, -0.04578089, -0.04879840, -0.05181590, -0.05483341, -0.05785092, -0.06086843, -
		0.06388594, -0.06690344, -0.06992095, -0.07293846, -0.07595597, -0.07897348, -0.08199098, -0.08500849, -0.08802600, -0.09104351, -0.09406102, -0.09707852, -0.10009603, -0.10311354, -0.10613105, -
		0.10914855, -0.11216606, -0.11518357, -0.11820108, -0.12121859, -0.12423609, -0.12725360, -0.13027111, -0.13328862, -0.13630613, -0.13932363, -0.14234114, -0.14535865, -0.14837616, -0.15139366, -
		0.15441117, -0.15742868, -0.16044619, -0.16346370, -0.16648121, -0.16949871, -0.17251622, -0.17553373, -0.17855124, -0.18156874, -0.18458625, -0.18760376, -0.19062127, -0.19363878, -0.19665629, -
		0.19967379, -0.20269130, -0.20570881, -0.20872632, -0.21174383, -0.21476133, -0.21777884, -0.22079635, -0.22381386, -0.22683136, -0.22984887, -0.23286638, -0.23588389, -0.23890140, -0.24191891, -
		0.24493641, -0.24795392, -0.25097143, -0.25398894, -0.25700644, -0.26002395, -0.26304146, -0.26605897, -0.26907648, -0.27209398, -0.27511149, -0.27812900, -0.28114651, -0.28416402, -0.28718152, -
		0.29019903, -0.29321654, -0.29623405, -0.29925156, -0.30226906, -0.30528657, -0.30830408, -0.31132159, -0.31433910, -0.31735660, -0.32037411, -0.32339162, -0.32640913, -0.32942664, -0.33244414, -
		0.33546165, -0.33847916, -0.34149667, -0.34451418, -0.34753168, -0.35054919, -0.35356670, -0.35658421, -0.35960172, -0.36261922, -0.36563673, -0.36865424, -0.37167175, -0.37468926, -0.37770676, -
		0.38072427, -0.38374178, -0.38675929, -0.38977680, -0.39279430, -0.39581181, -0.39882932, -0.40184683, -0.40486434, -0.40788184, -0.41089935, -0.41391686, -0.41693437, -0.41995188, -0.42296938, -
		0.42598689, -0.42900440, -0.43202191, -0.43503942, -0.43805692, -0.44107443, -0.44409194, -0.44710945, -0.45012695, -0.45314446, -0.45616197, -0.45917948, -0.46219699, -0.46521450, -0.46823200, -
		0.47124951, -0.47426702, -0.47728453, -0.48030203, -0.48331954, -0.48633705, -0.48935456, -0.49237207, -0.49538958, -0.49840708, -0.50142459, -0.50444210, -0.50745961, -0.51047712, -0.51349462, -
		0.51651213, -0.51952964, -0.52254715, -0.52556466, -0.52858216, -0.53159967, -0.53461718, -0.53763469, -0.54065219, -0.54366970, -0.54668721, -0.54970472, -0.55272223, -0.55573974, -0.55875724, -
		0.56177475, -0.56479226, -0.56780977, -0.57082727, -0.57384478, -0.57686229, -0.57987980, -0.58289731, -0.58591482, -0.58893232, -0.59194983, -0.59496734, -0.59798485, -0.60100235, -0.60401986, -
		0.60703737, -0.61005488, -0.61307239, -0.61608990, -0.61910740, -0.62212491, -0.62514242, -0.62815993, -0.63117744, -0.63419494, -0.63721245, -0.64022996, -0.64324747, -0.64626498, -0.64928248, -
		0.65229999, -0.65531750, -0.65833501, -0.66135252, -0.66437002, -0.66738753, -0.67040504, -0.67342255, -0.67644006, -0.67945756, -0.68247507, -0.68549258, -0.68851009, -0.69152760, -0.69454510, -
		0.69756261, -0.70058012, -0.70359763, -0.70661514, -0.70963264, -0.71265015, -0.71566766, -0.71868517, -0.72170268, -0.72472018, -0.72773769, -0.73075520, -0.73377271, -0.73679022, -0.73980772, -
		0.74282523, -0.74584274, -0.74886025, -0.75187776, -0.75489526, -0.75791277, -0.76093028, -0.76394779, -0.76696530, -0.76998281, -0.77300031, -0.77601782, -0.77903533, -0.78205284, -0.78507034, -
		0.78808785, -0.79110536, -0.79412287, -0.79714038, -0.80015789, -0.80317539, -0.80619290, -0.80921041, -0.81222792, -0.81524543, -0.81826293, -0.82128044, -0.82429795, -0.82731546, -0.83033297, -
		0.83335047, -0.83636798, -0.83938549, -0.84240300, -0.84542051, -0.84843801, -0.85145552, -0.85447303, -0.85749054, -0.86050805, -0.86352555, -0.86654306, -0.86956057, -0.87257808, -0.87559559, -
		0.87861309, -0.88163060, -0.88464811, -0.88766562, -0.89068313, -0.89370063, -0.89671814, -0.89973565, -0.90275316, -0.90577067, -0.90878818, -0.91180568, -0.91482319, -0.91784070, -0.92085821, -
		0.92387572, -0.92689322, -0.92991073, -0.93292824, -0.93594575, -0.93896326, -0.94198076, -0.94499827, -0.94801578, -0.95103329, -0.95405080, -0.95706830, -0.96008581, -0.96310332, -0.96612083, -
		0.96913834, -0.97215584, -0.97517335, -0.97819086, -0.98120837, -0.98422588, -0.98724339, -0.99026089, -0.99327840, -0.99629591, -0.99931342, -1.00233093, -1.00534843, -1.00836594, -1.01138345, -
		1.01440096, -1.01741847, -1.02043597, -1.02345348, -1.02647099, -1.02948850, -1.03250601, -1.03552351, -1.03854102, -1.04155853, -1.04457604, -1.04759355, -1.05061105, -1.05362856, -1.05664607, -
		1.05966358, -1.06268109, -1.06569860, -1.06871610, -1.07173361, -1.07475112, -1.07776863, -1.08078614, -1.08380364, -1.08682115, -1.08983866, -1.09285617, -1.09587368, -1.09889118, -1.10190869, -
		1.10492620, -1.10794371, -1.11096122, -1.11397873, -1.11699623, -1.12001374, -1.12303125, -1.12604876, -1.12906627, -1.13208377, -1.13510128, -1.13811879, -1.14113630, -1.14415381, -1.14717131, -
		1.15018882, -1.15320633, -1.15622384, -1.15924135, -1.16225885, -1.16527636, -1.16829387, -1.17131138, -1.17432889, -1.17734640, -1.18036390, -1.18338141, -1.18639892, -1.18941643, -1.19243394, -
		1.19545144, -1.19846895, -1.20148646, -1.20450397, -1.20752148, -1.21053898, -1.21355649, -1.21657400, -1.21959151, -1.22260902, -1.22562653, -1.22864403, -1.23166154, -1.23467905, -1.23769656, -
		1.24071407, -1.24373157, -1.24674908, -1.24976659, -1.25278410, -1.25580161, -1.25881911, -1.26183662, -1.26485413, -1.26787164, -1.27088915, -1.27390666, -1.27692416, -1.27994167, -1.28295918, -
		1.28597669, -1.28899420, -1.29201170, -1.29502921, -1.29804672, -1.30106423, -1.30408174, -1.30709925, -1.31011675, -1.31313426, -1.31615177, -1.31916928, -1.32218679, -1.32520429, -1.32822180, -
		1.33123931, -1.33425682, -1.33727433, -1.34029184, -1.34330934, -1.34632685, -1.34934436, -1.35236187, -1.35537938, -1.35839688, -1.36141439, -1.36443190, -1.36744941, -1.37046692, -1.37348442, -
		1.37650193, -1.37951944, -1.38253695, -1.38555446, -1.38857197, -1.39158947, -1.39460698, -1.39762449, -1.40064200, -1.40365951, -1.40667701, -1.40969452, -1.41271203, -1.41572954, -1.41874705, -
		1.42176456, -1.42478206, -1.42779957, -1.43081708, -1.43383459, -1.43685210, -1.43986960, -1.44288711, -1.44590462, -1.44892213, -1.45193964, -1.45495715, -1.45797465, -1.46099216, -1.46400967, -
		1.46702718, -1.47004469, -1.47306219, -1.47607970, -1.47909721, -1.48211472, -1.48513223, -1.48814974, -1.49116724, -1.49418475, -1.49720226, -1.50021977, -1.50323728, -1.50625478, -1.50927229, -
		1.51228980, -1.51530731, -1.51832482, -1.52134233, -1.52435983, -1.52737734, -1.53039485, -1.53341236, -1.53642987, -1.53944737, -1.54246488, -1.54548239, -1.54849990, -1.55151741, -1.55453492, -
		1.55755242, -1.56056993, -1.56358744, -1.56660495, -1.56962246, -1.57263997, -1.57565747, -1.57867498, -1.58169249, -1.58471000, -1.58772751, -1.59074502, -1.59376252, -1.59678003, -1.59979754, -
		1.60281505, -1.60583256, -1.60885006, -1.61186757, -1.61488508, -1.61790259, -1.62092010, -1.62393761, -1.62695511, -1.62997262, -1.63299013, -1.63600764, -1.63902515, -1.64204265, -1.64506016, -
		1.64807767, -1.65109518, -1.65411269, -1.65713020, -1.66014770, -1.66316521, -1.66618272, -1.66920023, -1.67221774, -1.67523525, -1.67825275, -1.68127026, -1.68428777, -1.68730528, -1.69032279, -
		1.69334029, -1.69635780, -1.69937531, -1.70239282, -1.70541033, -1.70842784, -1.71144534, -1.71446285, -1.71748036, -1.72049787, -1.72351538, -1.72653288, -1.72955039, -1.73256790, -1.73558541, -
		1.73860292, -1.74162043, -1.74463793, -1.74765544, -1.75067295, -1.75369046, -1.75670797, -1.75972548, -1.76274298, -1.76576049, -1.76877800, -1.77179551, -1.77481302, -1.77783053, -1.78084803, -
		1.78386554, -1.78688305, -1.78990056, -1.79291807, -1.79593558, -1.79895308, -1.80197059, -1.80498810, -1.80800561, -1.81102312, -1.81404062, -1.81705813, -1.82007564, -1.82309315, -1.82611066, -
		1.82912817, -1.83214567, -1.83516318, -1.83818069, -1.84119820, -1.84421571, -1.84723322, -1.85025072, -1.85326823, -1.85628574, -1.85930325, -1.86232076, -1.86533827, -1.86835577, -1.87137328, -
		1.87439079, -1.87740830, -1.88042581, -1.88344332, -1.88646082, -1.88947833, -1.89249584, -1.89551335, -1.89853086, -1.90154837, -1.90456587, -1.90758338, -1.91060089, -1.91361840, -1.91663591, -
		1.91965342, -1.92267092, -1.92568843, -1.92870594, -1.93172345, -1.93474096, -1.93775847, -1.94077597, -1.94379348, -1.94681099, -1.94982850, -1.95284601, -1.95586351, -1.95888102, -1.96189853, -
		1.96491604, -1.96793355, -1.97095106, -1.97396856, -1.97698607, -1.98000358, -1.98302109, -1.98603860, -1.98905611, -1.99207361, -1.99509112, -1.99810863, -2.00112614, -2.00414365, -2.00716116, -
		2.01017866, -2.01319617, -2.01621368, -2.01923119, -2.02224870, -2.02526621, -2.02828371, -2.03130122, -2.03431873, -2.03733624, -2.04035375, -2.04337126, -2.04638876, -2.04940627, -2.05242378, -
		2.05544129, -2.05845880, -2.06147631, -2.06449381, -2.06751132, -2.07052883, -2.07354634, -2.07656385, -2.07958136, -2.08259886, -2.08561637, -2.08863388, -2.09165139, -2.09466890, -2.09768641, -
		2.10070391, -2.10372142, -2.10673893, -2.10975644, -2.11277395, -2.11579146, -2.11880896, -2.12182647, -2.12484398, -2.12786149, -2.13087900, -2.13389651, -2.13691401, -2.13993152, -2.14294903, -
		2.14596654, -2.14898405, -2.15200156, -2.15501906, -2.15803657, -2.16105408, -2.16407159, -2.16708910, -2.17010661, -2.17312411, -2.17614162, -2.17915913, -2.18217664, -2.18519415, -2.18821166, -
		2.19122916, -2.19424667, -2.19726418, -2.20028169, -2.20329920, -2.20631671, -2.20933421, -2.21235172, -2.21536923, -2.21838674, -2.22140425, -2.22442176, -2.22743926, -2.23045677, -2.23347428, -
		2.23649179, -2.23950930, -2.24252681, -2.24554432, -2.24856182, -2.25157933, -2.25459684, -2.25761435, -2.26063186, -2.26364937, -2.26666687, -2.26968438, -2.27270189, -2.27571940, -2.27873691, -
		2.28175442, -2.28477192, -2.28778943, -2.29080694, -2.29382445, -2.29684196, -2.29985947, -2.30287697, -2.30589448, -2.30891199, -2.31192950, -2.31494701, -2.31796452, -2.32098202, -2.32399953, -
		2.32701704, -2.33003455, -2.33305206, -2.33606957, -2.33908708, -2.34210458, -2.34512209, -2.34813960, -2.35115711, -2.35417462, -2.35719213, -2.36020963, -2.36322714, -2.36624465, -2.36926216, -
		2.37227967, -2.37529718, -2.37831468, -2.38133219, -2.38434970, -2.38736721, -2.39038472, -2.39340223, -2.39641973, -2.39943724, -2.40245475, -2.40547226, -2.40848977, -2.41150728, -2.41452478, -
		2.41754229, -2.42055980, -2.42357731, -2.42659482, -2.42961233, -2.43262984, -2.43564734, -2.43866485, -2.44168236, -2.44469987, -2.44771738, -2.45073489, -2.45375239, -2.45676990, -2.45978741, -
		2.46280492, -2.46582243, -2.46883994, -2.47185744, -2.47487495, -2.47789246, -2.48090997, -2.48392748, -2.48694499, -2.48996250, -2.49298000, -2.49599751, -2.49901502, -2.50203253, -2.50505004, -
		2.50806755, -2.51108505, -2.51410256, -2.51712007, -2.52013758, -2.52315509, -2.52617260, -2.52919011, -2.53220761, -2.53522512, -2.53824263, -2.54126014, -2.54427765, -2.54729516, -2.55031266, -
		2.55333017, -2.55634768, -2.55936519, -2.56238270, -2.56540021, -2.56841771, -2.57143522, -2.57445273, -2.57747024, -2.58048775, -2.58350526, -2.58652277, -2.58954027, -2.59255778, -2.59557529, -
		2.59859280, -2.60161031, -2.60462782, -2.60764532, -2.61066283, -2.61368034, -2.61669785, -2.61971536, -2.62273287, -2.62575038, -2.62876788, -2.63178539, -2.63480290, -2.63782041, -2.64083792, -
		2.64385543, -2.64687293, -2.64989044, -2.65290795, -2.65592546, -2.65894297, -2.66196048, -2.66497798, -2.66799549, -2.67101300, -2.67403051, -2.67704802, -2.68006553, -2.68308304, -2.68610054, -
		2.68911805, -2.69213556, -2.69515307, -2.69817058, -2.70118809, -2.70420559, -2.70722310, -2.71024061, -2.71325812, -2.71627563, -2.71929314, -2.72231065, -2.72532815, -2.72834566, -2.73136317, -
		2.73438068, -2.73739819, -2.74041570, -2.74343320, -2.74645071, -2.74946822, -2.75248573, -2.75550324, -2.75852075, -2.76153826, -2.76455576, -2.76757327, -2.77059078, -2.77360829, -2.77662580, -
		2.77964331, -2.78266082, -2.78567832, -2.78869583, -2.79171334, -2.79473085, -2.79774836, -2.80076587, -2.80378337, -2.80680088, -2.80981839, -2.81283590, -2.81585341, -2.81887092, -2.82188843, -
		2.82490593, -2.82792344, -2.83094095, -2.83395846, -2.83697597, -2.83999348, -2.84301098, -2.84602849, -2.84904600, -2.85206351, -2.85508102, -2.85809853, -2.86111604, -2.86413354, -2.86715105, -
		2.87016856, -2.87318607, -2.87620358, -2.87922109, -2.88223860, -2.88525610, -2.88827361, -2.89129112, -2.89430863, -2.89732614, -2.90034365, -2.90336115, -2.90637866, -2.90939617, -2.91241368, -
		2.91543119, -2.91844870, -2.92146621, -2.92448371, -2.92750122, -2.93051873, -2.93353624, -2.93655375, -2.93957126, -2.94258877, -2.94560627, -2.94862378, -2.95164129, -2.95465880, -2.95767631, -
		2.96069382, -2.96371132, -2.96672883, -2.96974634, -2.97276385, -2.97578136, -2.97879887, -2.98181638, -2.98483388, -2.98785139, -2.99086890, -2.99388641, -2.99690392, -2.99992143, -3.00293894, -
		3.00595644, -3.00897395, -3.01199146, -3.01500897, -3.01802648, -3.02104399, -3.02406150, -3.02707900, -3.03009651, -3.03311402, -3.03613153, -3.03914904, -3.04216655, -3.04518405, -3.04820156, -
		3.05121907, -3.05423658, -3.05725409, -3.06027160, -3.06328911, -3.06630661, -3.06932412, -3.07234163, -3.07535914, -3.07837665, -3.08139416, -3.08441167, -3.08742917, -3.09044668, -3.09346419, -
		3.09648170, -3.09949921, -3.10251672, -3.10553423, -3.10855173, -3.11156924, -3.11458675, -3.11760426, -3.12062177, -3.12363928, -3.12665679, -3.12967429, -3.13269180, -3.13570931, -3.13872682, -
		3.14174433, -3.14476184, -3.14777934, -3.15079685, -3.15381436, -3.15683187, -3.15984938, -3.16286689, -3.16588440, -3.16890190, -3.17191941, -3.17493692, -3.17795443, -3.18097194, -3.18398945, -
		3.18700696, -3.19002446, -3.19304197, -3.19605948, -3.19907699, -3.20209450, -3.20511201, -3.20812952, -3.21114702, -3.21416453, -3.21718204, -3.22019955, -3.22321706, -3.22623457, -3.22925208, -
		3.23226958, -3.23528709, -3.23830460, -3.24132211, -3.24433962, -3.24735713, -3.25037464, -3.25339214, -3.25640965, -3.25942716, -3.26244467, -3.26546218, -3.26847969, -3.27149720, -3.27451470, -
		3.27753221, -3.28054972, -3.28356723, -3.28658474, -3.28960225, -3.29261975, -3.29563726, -3.29865477, -3.30167228, -3.30468979, -3.30770730, -3.31072481, -3.31374232, -3.31675982, -3.31977733, -
		3.32279484, -3.32581235, -3.32882986, -3.33184737, -3.33486488, -3.33788238, -3.34089989, -3.34391740, -3.34693491, -3.34995242, -3.35296993, 3.30135144, 3.29833093, 3.29531041, 3.29228989,
		3.28926937, 3.28624885, 3.28322834, 3.28020782, 3.27718730, 3.27416678, 3.27114626, 3.26812575, 3.26510523, 3.26208471, 3.25906419, 3.25604368, 3.25302316, 3.25000264, 3.24698212, 3.24396160,
		3.24094109, 3.23792057, 3.23490005, 3.23187953, 3.22885901, 3.22583850, 3.22281798, 3.21979746, 3.21677694, 3.21375642, 3.21073591, 3.20771539, 3.20469487, 3.20167435, 3.19865384, 3.19563332,
		3.19261280, 3.18959228, 3.18657176, 3.18355125, 3.18053073, 3.17751021, 3.17448969, 3.17146918, 3.16844866, 3.16542814, 3.16240762, 3.15938710, 3.15636659, 3.15334607, 3.15032555, 3.14730503,
		3.14428451, 3.14126400, 3.13824348, 3.13522296, 3.13220244, 3.12918193, 3.12616141, 3.12314089, 3.12012037, 3.11709985, 3.11407934, 3.11105882, 3.10803830, 3.10501778, 3.10199727, 3.09897675,
		3.09595623, 3.09293571, 3.08991519, 3.08689468, 3.08387416, 3.08085364, 3.07783312, 3.07481260, 3.07179209, 3.06877157, 3.06575105, 3.06273053, 3.05971002, 3.05668950, 3.05366898, 3.05064846,
		3.04762795, 3.04460743, 3.04158691, 3.03856639, 3.03554587, 3.03252536, 3.02950484, 3.02648432, 3.02346380, 3.02044329, 3.01742277, 3.01440225, 3.01138173, 3.00836121, 3.00534070, 3.00232018,
		2.99929966, 2.99627914, 2.99325863, 2.99023811, 2.98721759, 2.98419707, 2.98117655, 2.97815604, 2.97513552, 2.97211500, 2.96909448, 2.96607397, 2.96305345, 2.96003293, 2.95701241, 2.95399189,
		2.95097138, 2.94795086, 2.94493034, 2.94190982, 2.93888931, 2.93586879, 2.93284827, 2.92982775, 2.92680724, 2.92378672, 2.92076620, 2.91774568, 2.91472516, 2.91170465, 2.90868413, 2.90566361,
		2.90264309, 2.89962258, 2.89660206, 2.89358154, 2.89056102, 2.88754051, 2.88451999, 2.88149947, 2.87847895, 2.87545843, 2.87243792, 2.86941740, 2.86639688, 2.86337636, 2.86035585, 2.85733533,
		2.85431481, 2.85129429, 2.84827378, 2.84525326, 2.84223274, 2.83921222, 2.83619171, 2.83317119, 2.83015067, 2.82713015, 2.82410963, 2.82108912, 2.81806860, 2.81504808, 2.81202756, 2.80900705,
		2.80598653, 2.80296601, 2.79994549, 2.79692498, 2.79390446, 2.79088394, 2.78786342, 2.78484291, 2.78182239, 2.77880187, 2.77578135, 2.77276083, 2.76974032, 2.76671980, 2.76369928, 2.76067876,
		2.75765825, 2.75463773, 2.75161721, 2.74859669, 2.74557618, 2.74255566, 2.73953514, 2.73651462, 2.73349411, 2.73047359, 2.72745307, 2.72443255, 2.72141203, 2.71839152, 2.71537100, 2.71235048,
		2.70932996, 2.70630945, 2.70328893, 2.70026841, 2.69724789, 2.69422738, 2.69120686, 2.68818634, 2.68516582, 2.68214531, 2.67912479, 2.67610427, 2.67308375, 2.67006324, 2.66704272, 2.66402220,
		2.66100168, 2.65798117, 2.65496065, 2.65194013, 2.64891961, 2.64589909, 2.64287858, 2.63985806, 2.63683754, 2.63381702, 2.63079651, 2.62777599, 2.62475547, 2.62173495, 2.61871444, 2.61569392,
		2.61267340, 2.60965288, 2.60663237, 2.60361185, 2.60059133, 2.59757081, 2.59455030, 2.59152978, 2.58850926, 2.58548874, 2.58246823, 2.57944771, 2.57642719, 2.57340667, 2.57038616, 2.56736564,
		2.56434512, 2.56132460, 2.55830409, 2.55528357, 2.55226305, 2.54924253, 2.54622202, 2.54320150, 2.54018098, 2.53716046, 2.53413995, 2.53111943, 2.52809891, 2.52507839, 2.52205787, 2.51903736,
		2.51601684, 2.51299632, 2.50997580, 2.50695529, 2.50393477, 2.50091425, 2.49789373, 2.49487322, 2.49185270, 2.48883218, 2.48581166, 2.48279115, 2.47977063, 2.47675011, 2.47372959, 2.47070908,
		2.46768856, 2.46466804, 2.46164752, 2.45862701, 2.45560649, 2.45258597, 2.44956545, 2.44654494, 2.44352442, 2.44050390, 2.43748338, 2.43446287, 2.43144235, 2.42842183, 2.42540131, 2.42238080,
		2.41936028, 2.41633976, 2.41331924, 2.41029873, 2.40727821, 2.40425769, 2.40123717, 2.39821666, 2.39519614, 2.39217562, 2.38915510, 2.38613459, 2.38311407, 2.38009355, 2.37707303, 2.37405252,
		2.37103200, 2.36801148, 2.36499096, 2.36197045, 2.35894993, 2.35592941, 2.35290889, 2.34988838, 2.34686786, 2.34384734, 2.34082682, 2.33780631, 2.33478579, 2.33176527, 2.32874475, 2.32572424,
		2.32270372, 2.31968320, 2.31666268, 2.31364217, 2.31062165, 2.30760113, 2.30458061, 2.30156010, 2.29853958, 2.29551906, 2.29249854, 2.28947803, 2.28645751, 2.28343699, 2.28041647, 2.27739596,
		2.27437544, 2.27135492, 2.26833440, 2.26531389, 2.26229337, 2.25927285, 2.25625234, 2.25323182, 2.25021130, 2.24719078, 2.24417027, 2.24114975, 2.23812923, 2.23510871, 2.23208820, 2.22906768,
		2.22604716, 2.22302664, 2.22000613, 2.21698561, 2.21396509, 2.21094457, 2.20792406, 2.20490354, 2.20188302, 2.19886250, 2.19584199, 2.19282147, 2.18980095, 2.18678043, 2.18375992, 2.18073940,
		2.17771888, 2.17469836, 2.17167785, 2.16865733, 2.16563681, 2.16261629, 2.15959578, 2.15657526, 2.15355474, 2.15053422, 2.14751371, 2.14449319, 2.14147267, 2.13845215, 2.13543164, 2.13241112,
		2.12939060, 2.12637009, 2.12334957, 2.12032905, 2.11730853, 2.11428802, 2.11126750, 2.10824698, 2.10522646, 2.10220595, 2.09918543, 2.09616491, 2.09314439, 2.09012388, 2.08710336, 2.08408284,
		2.08106232, 2.07804181, 2.07502129, 2.07200077, 2.06898025, 2.06595974, 2.06293922, 2.05991870, 2.05689818, 2.05387767, 2.05085715, 2.04783663, 2.04481611, 2.04179560, 2.03877508, 2.03575456,
		2.03273405, 2.02971353, 2.02669301, 2.02367249, 2.02065198, 2.01763146, 2.01461094, 2.01159042, 2.00856991, 2.00554939, 2.00252887, 1.99950835, 1.99648784, 1.99346732, 1.99044680, 1.98742628,
		1.98440577, 1.98138525, 1.97836473, 1.97534421, 1.97232370, 1.96930318, 1.96628266, 1.96326214, 1.96024163, 1.95722111, 1.95420059, 1.95118008, 1.94815956, 1.94513904, 1.94211852, 1.93909801,
		1.93607749, 1.93305697, 1.93003645, 1.92701594, 1.92399542, 1.92097490, 1.91795438, 1.91493387, 1.91191335, 1.90889283, 1.90587231, 1.90285180, 1.89983128, 1.89681076, 1.89379025, 1.89076973,
		1.88774921, 1.88472869, 1.88170818, 1.87868766, 1.87566714, 1.87264662, 1.86962611, 1.86660559, 1.86358507, 1.86056455, 1.85754404, 1.85452352, 1.85150300, 1.84848248, 1.84546197, 1.84244145,
		1.83942093, 1.83640041, 1.83337990, 1.83035938, 1.82733886, 1.82431835, 1.82129783, 1.81827731, 1.81525679, 1.81223628, 1.80921576, 1.80619524, 1.80317472, 1.80015421, 1.79713369, 1.79411317,
		1.79109265, 1.78807214, 1.78505162, 1.78203110, 1.77901058, 1.77599007, 1.77296955, 1.76994903, 1.76692852, 1.76390800, 1.76088748, 1.75786696, 1.75484645, 1.75182593, 1.74880541, 1.74578489,
		1.74276438, 1.73974386, 1.73672334, 1.73370282, 1.73068231, 1.72766179, 1.72464127, 1.72162075, 1.71860024, 1.71557972, 1.71255920, 1.70953869, 1.70651817, 1.70349765, 1.70047713, 1.69745662,
		1.69443610, 1.69141558, 1.68839506, 1.68537455, 1.68235403, 1.67933351, 1.67631299, 1.67329248, 1.67027196, 1.66725144, 1.66423093, 1.66121041, 1.65818989, 1.65516937, 1.65214886, 1.64912834,
		1.64610782, 1.64308730, 1.64006679, 1.63704627, 1.63402575, 1.63100523, 1.62798472, 1.62496420, 1.62194368, 1.61892317, 1.61590265, 1.61288213, 1.60986161, 1.60684110, 1.60382058, 1.60080006,
		1.59777954, 1.59475903, 1.59173851, 1.58871799, 1.58569747, 1.58267696, 1.57965644, 1.57663592, 1.57361541, 1.57059489, 1.56757437, 1.56455385, 1.56153334, 1.55851282, 1.55549230, 1.55247178,
		1.54945127, 1.54643075, 1.54341023, 1.54038971, 1.53736920, 1.53434868, 1.53132816, 1.52830764, 1.52528713, 1.52226661, 1.51924609, 1.51622558, 1.51320506, 1.51018454, 1.50716402, 1.50414351,
		1.50112299, 1.49810247, 1.49508195, 1.49206144, 1.48904092, 1.48602040, 1.48299988, 1.47997937, 1.47695885, 1.47393833, 1.47091782, 1.46789730, 1.46487678, 1.46185626, 1.45883575, 1.45581523,
		1.45279471, 1.44977419, 1.44675368, 1.44373316, 1.44071264, 1.43769213, 1.43467161, 1.43165109, 1.42863057, 1.42561006, 1.42258954, 1.41956902, 1.41654850, 1.41352799, 1.41050747, 1.40748695,
		1.40446643, 1.40144592, 1.39842540, 1.39540488, 1.39238437, 1.38936385, 1.38634333, 1.38332281, 1.38030230, 1.37728178, 1.37426126, 1.37124074, 1.36822023, 1.36519971, 1.36217919, 1.35915867,
		1.35613816, 1.35311764, 1.35009712, 1.34707661, 1.34405609, 1.34103557, 1.33801505, 1.33499454, 1.33197402, 1.32895350, 1.32593298, 1.32291247, 1.31989195, 1.31687143, 1.31385092, 1.31083040,
		1.30780988, 1.30478936, 1.30176885, 1.29874833, 1.29572781, 1.29270729, 1.28968678, 1.28666626, 1.28364574, 1.28062522, 1.27760471, 1.27458419, 1.27156367, 1.26854316, 1.26552264, 1.26250212,
		1.25948160, 1.25646109, 1.25344057, 1.25042005, 1.24739953, 1.24437902, 1.24135850, 1.23833798, 1.23531746, 1.23229695, 1.22927643, 1.22625591, 1.22323540, 1.22021488, 1.21719436, 1.21417384,
		1.21115333, 1.20813281, 1.20511229, 1.20209177, 1.19907126, 1.19605074, 1.19303022, 1.19000971, 1.18698919, 1.18396867, 1.18094815, 1.17792764, 1.17490712, 1.17188660, 1.16886608, 1.16584557,
		1.16282505, 1.15980453, 1.15678401, 1.15376350, 1.15074298, 1.14772246, 1.14470195, 1.14168143, 1.13866091, 1.13564039, 1.13261988, 1.12959936, 1.12657884, 1.12355832, 1.12053781, 1.11751729,
		1.11449677, 1.11147626, 1.10845574, 1.10543522, 1.10241470, 1.09939419, 1.09637367, 1.09335315, 1.09033263, 1.08731212, 1.08429160, 1.08127108, 1.07825057, 1.07523005, 1.07220953, 1.06918901,
		1.06616850, 1.06314798, 1.06012746, 1.05710694, 1.05408643, 1.05106591, 1.04804539, 1.04502487, 1.04200436, 1.03898384, 1.03596332, 1.03294281, 1.02992229, 1.02690177, 1.02388125, 1.02086074,
		1.01784022, 1.01481970, 1.01179918, 1.00877867, 1.00575815, 1.00273763, 0.99971712, 0.99669660, 0.99367608, 0.99065556, 0.98763505, 0.98461453, 0.98159401, 0.97857349, 0.97555298, 0.97253246,
		0.96951194, 0.96649143, 0.96347091, 0.96045039, 0.95742987, 0.95440936, 0.95138884, 0.94836832, 0.94534780, 0.94232729, 0.93930677, 0.93628625, 0.93326574, 0.93024522, 0.92722470, 0.92420418,
		0.92118367, 0.91816315, 0.91514263, 0.91212211, 0.90910160, 0.90608108, 0.90306056, 0.90004004, 0.89701953, 0.89399901, 0.89097849, 0.88795798, 0.88493746, 0.88191694, 0.87889642, 0.87587591,
		0.87285539, 0.86983487, 0.86681435, 0.86379384, 0.86077332, 0.85775280, 0.85473229, 0.85171177, 0.84869125, 0.84567073, 0.84265022, 0.83962970, 0.83660918, 0.83358866, 0.83056815, 0.82754763,
		0.82452711, 0.82150660, 0.81848608, 0.81546556, 0.81244504, 0.80942453, 0.80640401, 0.80338349, 0.80036297, 0.79734246, 0.79432194, 0.79130142, 0.78828091, 0.78526039, 0.78223987, 0.77921935,
		0.77619884, 0.77317832, 0.77015780, 0.76713728, 0.76411677, 0.76109625, 0.75807573, 0.75505522, 0.75203470, 0.74901418, 0.74599366, 0.74297315, 0.73995263, 0.73693211, 0.73391159, 0.73089108,
		0.72787056, 0.72485004, 0.72182952, 0.71880901, 0.71578849, 0.71276797, 0.70974746, 0.70672694, 0.70370642, 0.70068590, 0.69766539, 0.69464487, 0.69162435, 0.68860383, 0.68558332, 0.68256280,
		0.67954228, 0.67652177, 0.67350125, 0.67048073, 0.66746021, 0.66443970, 0.66141918, 0.65839866, 0.65537814, 0.65235763, 0.64933711, 0.64631659, 0.64329608, 0.64027556, 0.63725504, 0.63423452,
		0.63121401, 0.62819349, 0.62517297, 0.62215245, 0.61913194, 0.61611142, 0.61309090, 0.61007039, 0.60704987, 0.60402935, 0.60100883, 0.59798832, 0.59496780, 0.59194728, 0.58892676, 0.58590625,
		0.58288573, 0.57986521, 0.57684470, 0.57382418, 0.57080366, 0.56778314, 0.56476263, 0.56174211, 0.55872159, 0.55570107, 0.55268056, 0.54966004, 0.54663952, 0.54361901, 0.54059849, 0.53757797,
		0.53455745, 0.53153694, 0.52851642, 0.52549590, 0.52247538, 0.51945487, 0.51643435, 0.51341383, 0.51039332, 0.50737280, 0.50435228, 0.50133176, 0.49831125, 0.49529073, 0.49227021, 0.48924969,
		0.48622918, 0.48320866, 0.48018814, 0.47716762, 0.47414711, 0.47112659, 0.46810607, 0.46508556, 0.46206504, 0.45904452, 0.45602400, 0.45300349, 0.44998297, 0.44696245, 0.44394193, 0.44092142,
		0.43790090, 0.43488038, 0.43185987, 0.42883935, 0.42581883, 0.42279831, 0.41977780, 0.41675728, 0.41373676, 0.41071624, 0.40769573, 0.40467521, 0.40165469, 0.39863418, 0.39561366, 0.39259314,
		0.38957262, 0.38655211, 0.38353159, 0.38051107, 0.37749055, 0.37447004, 0.37144952, 0.36842900, 0.36540849, 0.36238797, 0.35936745, 0.35634693, 0.35332642, 0.35030590, 0.34728538, 0.34426486,
		0.34124435, 0.33822383, 0.33520331, 0.33218280, 0.32916228, 0.32614176, 0.32312124, 0.32010073, 0.31708021, 0.31405969, 0.31103917, 0.30801866, 0.30499814, 0.30197762, 0.29895711, 0.29593659,
		0.29291607, 0.28989555, 0.28687504, 0.28385452, 0.28083400, 0.27781348, 0.27479297, 0.27177245, 0.26875193, 0.26573142, 0.26271090, 0.25969038, 0.25666986, 0.25364935, 0.25062883, 0.24760831,
		0.24458779, 0.24156728, 0.23854676, 0.23552624, 0.23250573, 0.22948521, 0.22646469, 0.22344417, 0.22042366, 0.21740314, 0.21438262, 0.21136210, 0.20834159, 0.20532107, 0.20230055, 0.19928004,
		0.19625952, 0.19323900, 0.19021848, 0.18719797, 0.18417745, 0.18115693, 0.17813641, 0.17511590, 0.17209538, 0.16907486, 0.16605435, 0.16303383, 0.16001331, 0.15699279, 0.15397228, 0.15095176,
		0.14793124, 0.14491072, 0.14189021, 0.13886969, 0.13584917, 0.13282866, 0.12980814, 0.12678762, 0.12376710, 0.12074659, 0.11772607, 0.11470555, 0.11168503, 0.10866452, 0.10564400, 0.10262348,
		0.09960296, 0.09658245, 0.09356193, 0.09054141, 0.08752090, 0.08450038, 0.08147986, 0.07845934, 0.07543883, 0.07241831, 0.06939779, 0.06637727, 0.06335676, 0.06033624, 0.05731572, 0.05429521,
		0.05127469, 0.04825417, 0.04523365, 0.04221314, 0.03919262, 0.03617210, 0.03315158, 0.03013107, 0.02711055, 0.02409003, 0.02106952, 0.01804900, 0.01502848, 0.01200796, 0.00898745, 0.00596693,
		0.00294641, -0.00007411, -0.00309462, -0.00611514, -0.00913566, -0.01215617, -0.01517669, -0.01819721, -0.02121773, -0.02423824, -0.02725876, -0.03027928, -0.03329980, -0.03632031, -0.03934083, -
		0.04236135, -0.04538186, -0.04840238, -0.05142290, -0.05444342, -0.05746393, -0.06048445, -0.06350497, -0.06652549, -0.06954600, -0.07256652, -0.07558704, -0.07860755, -0.08162807, -0.08464859, -
		0.08766911, -0.09068962, -0.09371014, -0.09673066, -0.09975118, -0.10277169, -0.10579221, -0.10881273, -0.11183324, -0.11485376, -0.11787428, -0.12089480, -0.12391531, -0.12693583, -0.12995635, -
		0.13297687, -0.13599738, -0.13901790, -0.14203842, -0.14505893, -0.14807945, -0.15109997, -0.15412049, -0.15714100, -0.16016152, -0.16318204, -0.16620256, -0.16922307, -0.17224359, -0.17526411, -
		0.17828462, -0.18130514, -0.18432566, -0.18734618, -0.19036669, -0.19338721, -0.19640773, -0.19942825, -0.20244876, -0.20546928, -0.20848980, -0.21151031, -0.21453083, -0.21755135, -0.22057187, -
		0.22359238, -0.22661290, -0.22963342, -0.23265394, -0.23567445, -0.23869497, -0.24171549, -0.24473600, -0.24775652, -0.25077704, -0.25379756, -0.25681807, -0.25983859, -0.26285911, -0.26587963, -
		0.26890014, -0.27192066, -0.27494118, -0.27796169, -0.28098221, -0.28400273, -0.28702325, -0.29004376, -0.29306428, -0.29608480, -0.29910532, -0.30212583, -0.30514635, -0.30816687, -0.31118739, -
		0.31420790, -0.31722842, -0.32024894, -0.32326945, -0.32628997, -0.32931049, -0.33233101, -0.33535152, -0.33837204, -0.34139256, -0.34441308, -0.34743359, -0.35045411, -0.35347463, -0.35649514, -
		0.35951566, -0.36253618, -0.36555670, -0.36857721, -0.37159773, -0.37461825, -0.37763877, -0.38065928, -0.38367980, -0.38670032, -0.38972083, -0.39274135, -0.39576187, -0.39878239, -0.40180290, -
		0.40482342, -0.40784394, -0.41086446, -0.41388497, -0.41690549, -0.41992601, -0.42294652, -0.42596704, -0.42898756, -0.43200808, -0.43502859, -0.43804911, -0.44106963, -0.44409015, -0.44711066, -
		0.45013118, -0.45315170, -0.45617221, -0.45919273, -0.46221325, -0.46523377, -0.46825428, -0.47127480, -0.47429532, -0.47731584, -0.48033635, -0.48335687, -0.48637739, -0.48939790, -0.49241842, -
		0.49543894, -0.49845946, -0.50147997, -0.50450049, -0.50752101, -0.51054153, -0.51356204, -0.51658256, -0.51960308, -0.52262359, -0.52564411, -0.52866463, -0.53168515, -0.53470566, -0.53772618, -
		0.54074670, -0.54376722, -0.54678773, -0.54980825, -0.55282877, -0.55584928, -0.55886980, -0.56189032, -0.56491084, -0.56793135, -0.57095187, -0.57397239, -0.57699291, -0.58001342, -0.58303394, -
		0.58605446, -0.58907498, -0.59209549, -0.59511601, -0.59813653, -0.60115704, -0.60417756, -0.60719808, -0.61021860, -0.61323911, -0.61625963, -0.61928015, -0.62230067, -0.62532118, -0.62834170, -
		0.63136222, -0.63438273, -0.63740325, -0.64042377, -0.64344429, -0.64646480, -0.64948532, -0.65250584, -0.65552636, -0.65854687, -0.66156739, -0.66458791, -0.66760842, -0.67062894, -0.67364946, -
		0.67666998, -0.67969049, -0.68271101, -0.68573153, -0.68875205, -0.69177256, -0.69479308, -0.69781360, -0.70083411, -0.70385463, -0.70687515, -0.70989567, -0.71291618, -0.71593670, -0.71895722, -
		0.72197774, -0.72499825, -0.72801877, -0.73103929, -0.73405980, -0.73708032, -0.74010084, -0.74312136, -0.74614187, -0.74916239, -0.75218291, -0.75520343, -0.75822394, -0.76124446, -0.76426498, -
		0.76728549, -0.77030601, -0.77332653, -0.77634705, -0.77936756, -0.78238808, -0.78540860, -0.78842912, -0.79144963, -0.79447015, -0.79749067, -0.80051118, -0.80353170, -0.80655222, -0.80957274, -
		0.81259325, -0.81561377, -0.81863429, -0.82165481, -0.82467532, -0.82769584, -0.83071636, -0.83373688, -0.83675739, -0.83977791, -0.84279843, -0.84581894, -0.84883946, -0.85185998, -0.85488050, -
		0.85790101, -0.86092153, -0.86394205, -0.86696257, -0.86998308, -0.87300360, -0.87602412, -0.87904463, -0.88206515, -0.88508567, -0.88810619, -0.89112670, -0.89414722, -0.89716774, -0.90018826, -
		0.90320877, -0.90622929, -0.90924981, -0.91227032, -0.91529084, -0.91831136, -0.92133188, -0.92435239, -0.92737291, -0.93039343, -0.93341395, -0.93643446, -0.93945498, -0.94247550, -0.94549601, -
		0.94851653, -0.95153705, -0.95455757, -0.95757808, -0.96059860, -0.96361912, -0.96663964, -0.96966015, -0.97268067, -0.97570119, -0.97872170, -0.98174222, -0.98476274, -0.98778326, -0.99080377, -
		0.99382429, -0.99684481, -0.99986533, -1.00288584, -1.00590636, -1.00892688, -1.01194740, -1.01496791, -1.01798843, -1.02100895, -1.02402946, -1.02704998, -1.03007050, -1.03309102, -1.03611153, -
		1.03913205, -1.04215257, -1.04517309, -1.04819360, -1.05121412, -1.05423464, -1.05725515, -1.06027567, -1.06329619, -1.06631671, -1.06933722, -1.07235774, -1.07537826, -1.07839878, -1.08141929, -
		1.08443981, -1.08746033, -1.09048084, -1.09350136, -1.09652188, -1.09954240, -1.10256291, -1.10558343, -1.10860395, -1.11162447, -1.11464498, -1.11766550, -1.12068602, -1.12370653, -1.12672705, -
		1.12974757, -1.13276809, -1.13578860, -1.13880912, -1.14182964, -1.14485016, -1.14787067, -1.15089119, -1.15391171, -1.15693222, -1.15995274, -1.16297326, -1.16599378, -1.16901429, -1.17203481, -
		1.17505533, -1.17807585, -1.18109636, -1.18411688, -1.18713740, -1.19015792, -1.19317843, -1.19619895, -1.19921947, -1.20223998, -1.20526050, -1.20828102, -1.21130154, -1.21432205, -1.21734257, -
		1.22036309, -1.22338361, -1.22640412, -1.22942464, -1.23244516, -1.23546567, -1.23848619, -1.24150671, -1.24452723, -1.24754774, -1.25056826, -1.25358878, -1.25660930, -1.25962981, -1.26265033, -
		1.26567085, -1.26869136, -1.27171188, -1.27473240, -1.27775292, -1.28077343, -1.28379395, -1.28681447, -1.28983499, -1.29285550, -1.29587602, -1.29889654, -1.30191705, -1.30493757, -1.30795809, -
		1.31097861, -1.31399912, -1.31701964, -1.32004016, -1.32306068, -1.32608119, -1.32910171, -1.33212223, -1.33514275, -1.33816326, -1.34118378, -1.34420430, -1.34722481, -1.35024533, -1.35326585, -
		1.35628637, -1.35930688, -1.36232740, -1.36534792, -1.36836844, -1.37138895, -1.37440947, -1.37742999, -1.38045050, -1.38347102, -1.38649154, -1.38951206, -1.39253257, -1.39555309, -1.39857361, -
		1.40159413, -1.40461464, -1.40763516, -1.41065568, -1.41367619, -1.41669671, -1.41971723, -1.42273775, -1.42575826, -1.42877878, -1.43179930, -1.43481982, -1.43784033, -1.44086085, -1.44388137, -
		1.44690188, -1.44992240, -1.45294292, -1.45596344, -1.45898395, -1.46200447, -1.46502499, -1.46804551, -1.47106602, -1.47408654, -1.47710706, -1.48012758, -1.48314809, -1.48616861, -1.48918913, -
		1.49220964, -1.49523016, -1.49825068, -1.50127120, -1.50429171, -1.50731223, -1.51033275, -1.51335327, -1.51637378, -1.51939430, -1.52241482, -1.52543533, -1.52845585, -1.53147637, -1.53449689, -
		1.53751740, -1.54053792, -1.54355844, -1.54657896, -1.54959947, -1.55261999, -1.55564051, -1.55866102, -1.56168154, -1.56470206, -1.56772258, -1.57074309, -1.57376361, -1.57678413, -1.57980465, -
		1.58282516, -1.58584568, -1.58886620, -1.59188672, -1.59490723, -1.59792775, -1.60094827, -1.60396878, -1.60698930, -1.61000982, -1.61303034, -1.61605085, -1.61907137, -1.62209189, -1.62511241, -
		1.62813292, -1.63115344, -1.63417396, -1.63719447, -1.64021499, -1.64323551, -1.64625603, -1.64927654, -1.65229706, -1.65531758, -1.65833810, -1.66135861, -1.66437913, -1.66739965, -1.67042016, -
		1.67344068, -1.67646120, -1.67948172, -1.68250223, -1.68552275, -1.68854327, -1.69156379, -1.69458430, -1.69760482, -1.70062534, -1.70364585, -1.70666637, -1.70968689, -1.71270741, -1.71572792, -
		1.71874844, -1.72176896, -1.72478948, -1.72780999, -1.73083051, -1.73385103, -1.73687155, -1.73989206, -1.74291258, -1.74593310, -1.74895361, -1.75197413, -1.75499465, -1.75801517, -1.76103568, -
		1.76405620, -1.76707672, -1.77009724, -1.77311775, -1.77613827, -1.77915879, -1.78217930, -1.78519982, -1.78822034, -1.79124086, -1.79426137, -1.79728189, -1.80030241, -1.80332293, -1.80634344, -
		1.80936396, -1.81238448, -1.81540499, -1.81842551, -1.82144603, -1.82446655, -1.82748706, -1.83050758, -1.83352810, -1.83654862, -1.83956913, -1.84258965, -1.84561017, -1.84863069, -1.85165120, -
		1.85467172, -1.85769224, -1.86071275, -1.86373327, -1.86675379, -1.86977431, -1.87279482, -1.87581534, -1.87883586, -1.88185638, -1.88487689, -1.88789741, -1.89091793, -1.89393844, -1.89695896, -
		1.89997948, -1.90300000, -1.90602051, -1.90904103, -1.91206155, -1.91508207, -1.91810258, -1.92112310, -1.92414362, -1.92716413, -1.93018465, -1.93320517, -1.93622569, -1.93924620, -1.94226672, -
		1.94528724, -1.94830776, -1.95132827, -1.95434879, -1.95736931, -1.96038983, -1.96341034, -1.96643086, -1.96945138, -1.97247189, -1.97549241, -1.97851293, -1.98153345, -1.98455396, -1.98757448, -
		1.99059500, -1.99361552, -1.99663603, -1.99965655, -2.00267707, -2.00569758, -2.00871810, -2.01173862, -2.01475914, -2.01777965, -2.02080017, -2.02382069, -2.02684121, -2.02986172, -2.03288224, -
		2.03590276, -2.03892327, -2.04194379, -2.04496431, -2.04798483, -2.05100534, -2.05402586, -2.05704638, -2.06006690, -2.06308741, -2.06610793, -2.06912845, -2.07214897, -2.07516948, -2.07819000, -
		2.08121052, -2.08423103, -2.08725155, -2.09027207, -2.09329259, -2.09631310, -2.09933362, -2.10235414, -2.10537466, -2.10839517, -2.11141569, -2.11443621, -2.11745672, -2.12047724, -2.12349776, -
		2.12651828, -2.12953879, -2.13255931, -2.13557983, -2.13860035, -2.14162086, -2.14464138, -2.14766190, -2.15068241, -2.15370293, -2.15672345, -2.15974397, -2.16276448, -2.16578500, -2.16880552, -
		2.17182604, -2.17484655, -2.17786707, -2.18088759, -2.18390811, -2.18692862, -2.18994914, -2.19296966, -2.19599017, -2.19901069, -2.20203121, -2.20505173, -2.20807224, -2.21109276, -2.21411328, -
		2.21713380, -2.22015431, -2.22317483, -2.22619535, -2.22921586, -2.23223638, -2.23525690, -2.23827742, -2.24129793, -2.24431845, -2.24733897, -2.25035949, -2.25338000, -2.25640052, -2.25942104, -
		2.26244156, -2.26546207, -2.26848259, -2.27150311, -2.27452362, -2.27754414, -2.28056466, -2.28358518, -2.28660569, -2.28962621, -2.29264673, -2.29566725, -2.29868776, -2.30170828, -2.30472880, -
		2.30774931, -2.31076983, -2.31379035, -2.31681087, -2.31983138, -2.32285190, -2.32587242, -2.32889294, -2.33191345, -2.33493397, -2.33795449, -2.34097500, -2.34399552, -2.34701604, -2.35003656, -
		2.35305707, -2.35607759, -2.35909811, -2.36211863, -2.36513914, -2.36815966, -2.37118018, -2.37420070, -2.37722121, -2.38024173, -2.38326225, -2.38628276, -2.38930328, -2.39232380, -2.39534432, -
		2.39836483, -2.40138535, -2.40440587, -2.40742639, -2.41044690, -2.41346742, -2.41648794, -2.41950845, -2.42252897, -2.42554949, -2.42857001, -2.43159052, -2.43461104, -2.43763156, -2.44065208, -
		2.44367259, -2.44669311, -2.44971363, -2.45273415, -2.45575466, -2.45877518, -2.46179570, -2.46481621, -2.46783673, -2.47085725, -2.47387777, -2.47689828, -2.47991880, -2.48293932, -2.48595984, -
		2.48898035, -2.49200087, -2.49502139, -2.49804190, -2.50106242, -2.50408294, -2.50710346, -2.51012397, -2.51314449, -2.51616501, -2.51918553, -2.52220604, -2.52522656, -2.52824708, -2.53126759, -
		2.53428811, -2.53730863, -2.54032915, -2.54334966, -2.54637018, -2.54939070, -2.55241122, -2.55543173, -2.55845225, -2.56147277, -2.56449329, -2.56751380, -2.57053432, -2.57355484, -2.57657535, -
		2.57959587, -2.58261639, -2.58563691, -2.58865742, -2.59167794, -2.59469846, -2.59771898, -2.60073949, -2.60376001, -2.60678053, -2.60980104, -2.61282156, -2.61584208, -2.61886260, -2.62188311, -
		2.62490363, -2.62792415, -2.63094467, -2.63396518, -2.63698570, -2.64000622, -2.64302674, -2.64604725, -2.64906777, -2.65208829, -2.65510880, -2.65812932, -2.66114984, -2.66417036, -2.66719087, -
		2.67021139, -2.67323191, -2.67625243, -2.67927294, -2.68229346, -2.68531398, -2.68833449, -2.69135501, -2.69437553, -2.69739605, -2.70041656, -2.70343708, -2.70645760, -2.70947812, -2.71249863, -
		2.71551915, -2.71853967, -2.72156018, -2.72458070, -2.72760122, -2.73062174, -2.73364225, -2.73666277, -2.73968329, -2.74270381, -2.74572432, -2.74874484, -2.75176536, -2.75478588, -2.75780639, -
		2.76082691, -2.76384743, -2.76686794, -2.76988846, -2.77290898, -2.77592950, -2.77895001, -2.78197053, -2.78499105, -2.78801157, -2.79103208, -2.79405260, -2.79707312, -2.80009363, -2.80311415, -
		2.80613467, -2.80915519, -2.81217570, -2.81519622, -2.81821674, -2.82123726, -2.82425777, -2.82727829, -2.83029881, -2.83331933, -2.83633984, -2.83936036, -2.84238088, -2.84540139, -2.84842191, -
		2.85144243, -2.85446295, -2.85748346, -2.86050398, -2.86352450, -2.86654502, -2.86956553, -2.87258605, -2.87560657, -2.87862708, -2.88164760, -2.88466812, -2.88768864, -2.89070915, -2.89372967, -
		2.89675019, -2.89977071, -2.90279122, -2.90581174, -2.90883226, -2.91185278, -2.91487329, -2.91789381, -2.92091433, -2.92393484, -2.92695536, -2.92997588, -2.93299640, -2.93601691, -2.93903743, -
		2.94205795, -2.94507847, -2.94809898, -2.95111950, -2.95414002, -2.95716053, -2.96018105, -2.96320157, -2.96622209, -2.96924260, -2.97226312, -2.97528364, -2.97830416, -2.98132467, -2.98434519, -
		2.98736571, -2.99038622, -2.99340674, -2.99642726, -2.99944778, -3.00246829, -3.00548881, -3.00850933, -3.01152985, -3.01455036, -3.01757088, -3.02059140, -3.02361192, -3.02663243, -3.02965295, -
		3.03267347, -3.03569398, -3.03871450, -3.04173502, -3.04475554, -3.04777605, -3.05079657, -3.05381709, -3.05683761, -3.05985812, -3.06287864, -3.06589916, -3.06891967, -3.07194019, -3.07496071, -
		3.07798123, -3.08100174, -3.08402226, -3.08704278, -3.09006330, -3.09308381, -3.09610433, -3.09912485, -3.10214537, -3.10516588, -3.10818640, -3.11120692, -3.11422743, -3.11724795, -3.12026847, -
		3.12328899, -3.12630950, -3.12933002, -3.13235054, -3.13537106, -3.13839157, -3.14141209, -3.14443261, -3.14745312, -3.15047364, -3.15349416, -3.15651468, -3.15953519, -3.16255571, -3.16557623, -
		3.16859675, -3.17161726, -3.17463778, -3.17765830, -3.18067882, -3.18369933, -3.18671985, -3.18974037, -3.19276088, -3.19578140, -3.19880192, -3.20182244, -3.20484295, -3.20786347, -3.21088399, -
		3.21390451, -3.21692502, -3.21994554, -3.22296606, -3.22598657, -3.22900709, -3.23202761, -3.23504813, -3.23806864, -3.24108916, -3.24410968, -3.24713020, -3.25015071, -3.25317123, -3.25619175, -
		3.25921227, -3.26223278, -3.26525330, -3.26827382, -3.27129433, -3.27431485, -3.27733537, -3.28035589, -3.28337640, -3.28639692, -3.28941744, -3.29243796, -3.29545847, -3.29847899, -3.30149951, -
		3.30452002, -3.30754054, -3.31056106, -3.31358158, -3.31660209, -3.31962261, -3.32264313, -3.32566365, -3.32868416, -3.33170468, -3.33472520, -3.33774572, -3.34076623, -3.34378675, -3.34680727, -
		3.34982778, -3.35284830, -3.35586882, -3.35888934, -3.36190985, -3.36493037, -3.36795089, -3.37097141, -3.37399192, -3.37701244, -3.38003296, -3.38305347, -3.38607399, -3.38909451, -3.39211503, -
		3.39513554, -3.39815606, -3.40117658, -3.40419710, -3.40721761, -3.41023813, -3.41325865, -3.41627917, -3.41929968, -3.42232020, -3.42534072, -3.42836123, -3.43138175, -3.43440227, -3.43742279, -
		3.44044330, -3.44346382, -3.44648434, -3.44950486, -3.45252537, -3.45554589, -3.45856641, -3.46158692, -3.46460744, -3.46762796, -3.47064848, -3.47366899, -3.47668951, -3.47971003, -3.48273055, -
		3.48575106, -3.48877158, -3.49179210, -3.49481261, -3.49783313, -3.50085365, -3.50387417, -3.50689468, -3.50991520, -3.51293572, -3.51595624, -3.51897675, -3.52199727, -3.52501779, -3.52803831, -
		3.53105882, -3.53407934, -3.53709986, -3.54012037, -3.54314089, -3.54616141, -3.54918193, -3.55220244, 3.51844081, 3.51536657, 3.51229233, 3.50921810, 3.50614386, 3.50306962, 3.49999538, 3.49692114,
		3.49384691, 3.49077267, 3.48769843, 3.48462419, 3.48154996, 3.47847572, 3.47540148, 3.47232724, 3.46925300, 3.46617877, 3.46310453, 3.46003029, 3.45695605, 3.45388182, 3.45080758, 3.44773334,
		3.44465910, 3.44158487, 3.43851063, 3.43543639, 3.43236215, 3.42928791, 3.42621368, 3.42313944, 3.42006520, 3.41699096, 3.41391673, 3.41084249, 3.40776825, 3.40469401, 3.40161977, 3.39854554,
		3.39547130, 3.39239706, 3.38932282, 3.38624859, 3.38317435, 3.38010011, 3.37702587, 3.37395164, 3.37087740, 3.36780316, 3.36472892, 3.36165468, 3.35858045, 3.35550621, 3.35243197, 3.34935773,
		3.34628350, 3.34320926, 3.34013502, 3.33706078, 3.33398654, 3.33091231, 3.32783807, 3.32476383, 3.32168959, 3.31861536, 3.31554112, 3.31246688, 3.30939264, 3.30631840, 3.30324417, 3.30016993,
		3.29709569, 3.29402145, 3.29094722, 3.28787298, 3.28479874, 3.28172450, 3.27865026, 3.27557603, 3.27250179, 3.26942755, 3.26635331, 3.26327908, 3.26020484, 3.25713060, 3.25405636, 3.25098213,
		3.24790789, 3.24483365, 3.24175941, 3.23868517, 3.23561094, 3.23253670, 3.22946246, 3.22638822, 3.22331399, 3.22023975, 3.21716551, 3.21409127, 3.21101703, 3.20794280, 3.20486856, 3.20179432,
		3.19872008, 3.19564585, 3.19257161, 3.18949737, 3.18642313, 3.18334889, 3.18027466, 3.17720042, 3.17412618, 3.17105194, 3.16797771, 3.16490347, 3.16182923, 3.15875499, 3.15568075, 3.15260652,
		3.14953228, 3.14645804, 3.14338380, 3.14030957, 3.13723533, 3.13416109, 3.13108685, 3.12801261, 3.12493838, 3.12186414, 3.11878990, 3.11571566, 3.11264143, 3.10956719, 3.10649295, 3.10341871,
		3.10034447, 3.09727024, 3.09419600, 3.09112176, 3.08804752, 3.08497329, 3.08189905, 3.07882481, 3.07575057, 3.07267633, 3.06960210, 3.06652786, 3.06345362, 3.06037938, 3.05730515, 3.05423091,
		3.05115667, 3.04808243, 3.04500819, 3.04193396, 3.03885972, 3.03578548, 3.03271124, 3.02963701, 3.02656277, 3.02348853, 3.02041429, 3.01734005, 3.01426582, 3.01119158, 3.00811734, 3.00504310,
		3.00196887, 2.99889463, 2.99582039, 2.99274615, 2.98967191, 2.98659768, 2.98352344, 2.98044920, 2.97737496, 2.97430073, 2.97122649, 2.96815225, 2.96507801, 2.96200377, 2.95892954, 2.95585530,
		2.95278106, 2.94970682, 2.94663259, 2.94355835, 2.94048411, 2.93740987, 2.93433563, 2.93126140, 2.92818716, 2.92511292, 2.92203868, 2.91896445, 2.91589021, 2.91281597, 2.90974173, 2.90666749,
		2.90359326, 2.90051902, 2.89744478, 2.89437054, 2.89129631, 2.88822207, 2.88514783, 2.88207359, 2.87899935, 2.87592512, 2.87285088, 2.86977664, 2.86670240, 2.86362817, 2.86055393, 2.85747969,
		2.85440545, 2.85133121, 2.84825698, 2.84518274, 2.84210850, 2.83903426, 2.83596003, 2.83288579, 2.82981155, 2.82673731, 2.82366307, 2.82058884, 2.81751460, 2.81444036, 2.81136612, 2.80829189,
		2.80521765, 2.80214341, 2.79906917, 2.79599493, 2.79292070, 2.78984646, 2.78677222, 2.78369798, 2.78062375, 2.77754951, 2.77447527, 2.77140103, 2.76832679, 2.76525256, 2.76217832, 2.75910408,
		2.75602984, 2.75295560, 2.74988137, 2.74680713, 2.74373289, 2.74065865, 2.73758442, 2.73451018, 2.73143594, 2.72836170, 2.72528746, 2.72221323, 2.71913899, 2.71606475, 2.71299051, 2.70991628,
		2.70684204, 2.70376780, 2.70069356, 2.69761932, 2.69454509, 2.69147085, 2.68839661, 2.68532237, 2.68224814, 2.67917390, 2.67609966, 2.67302542, 2.66995118, 2.66687695, 2.66380271, 2.66072847,
		2.65765423, 2.65458000, 2.65150576, 2.64843152, 2.64535728, 2.64228304, 2.63920881, 2.63613457, 2.63306033, 2.62998609, 2.62691186, 2.62383762, 2.62076338, 2.61768914, 2.61461490, 2.61154067,
		2.60846643, 2.60539219, 2.60231795, 2.59924372, 2.59616948, 2.59309524, 2.59002100, 2.58694676, 2.58387253, 2.58079829, 2.57772405, 2.57464981, 2.57157557, 2.56850134, 2.56542710, 2.56235286,
		2.55927862, 2.55620439, 2.55313015, 2.55005591, 2.54698167, 2.54390743, 2.54083320, 2.53775896, 2.53468472, 2.53161048, 2.52853625, 2.52546201, 2.52238777, 2.51931353, 2.51623929, 2.51316506,
		2.51009082, 2.50701658, 2.50394234, 2.50086811, 2.49779387, 2.49471963, 2.49164539, 2.48857115, 2.48549692, 2.48242268, 2.47934844, 2.47627420, 2.47319997, 2.47012573, 2.46705149, 2.46397725,
		2.46090301, 2.45782878, 2.45475454, 2.45168030, 2.44860606, 2.44553182, 2.44245759, 2.43938335, 2.43630911, 2.43323487, 2.43016064, 2.42708640, 2.42401216, 2.42093792, 2.41786368, 2.41478945,
		2.41171521, 2.40864097, 2.40556673, 2.40249250, 2.39941826, 2.39634402, 2.39326978, 2.39019554, 2.38712131, 2.38404707, 2.38097283, 2.37789859, 2.37482436, 2.37175012, 2.36867588, 2.36560164,
		2.36252740, 2.35945317, 2.35637893, 2.35330469, 2.35023045, 2.34715621, 2.34408198, 2.34100774, 2.33793350, 2.33485926, 2.33178503, 2.32871079, 2.32563655, 2.32256231, 2.31948807, 2.31641384,
		2.31333960, 2.31026536, 2.30719112, 2.30411689, 2.30104265, 2.29796841, 2.29489417, 2.29181993, 2.28874570, 2.28567146, 2.28259722, 2.27952298, 2.27644875, 2.27337451, 2.27030027, 2.26722603,
		2.26415179, 2.26107756, 2.25800332, 2.25492908, 2.25185484, 2.24878060, 2.24570637, 2.24263213, 2.23955789, 2.23648365, 2.23340942, 2.23033518, 2.22726094, 2.22418670, 2.22111246, 2.21803823,
		2.21496399, 2.21188975, 2.20881551, 2.20574128, 2.20266704, 2.19959280, 2.19651856, 2.19344432, 2.19037009, 2.18729585, 2.18422161, 2.18114737, 2.17807313, 2.17499890, 2.17192466, 2.16885042,
		2.16577618, 2.16270195, 2.15962771, 2.15655347, 2.15347923, 2.15040499, 2.14733076, 2.14425652, 2.14118228, 2.13810804, 2.13503381, 2.13195957, 2.12888533, 2.12581109, 2.12273685, 2.11966262,
		2.11658838, 2.11351414, 2.11043990, 2.10736567, 2.10429143, 2.10121719, 2.09814295, 2.09506871, 2.09199448, 2.08892024, 2.08584600, 2.08277176, 2.07969752, 2.07662329, 2.07354905, 2.07047481,
		2.06740057, 2.06432634, 2.06125210, 2.05817786, 2.05510362, 2.05202938, 2.04895515, 2.04588091, 2.04280667, 2.03973243, 2.03665820, 2.03358396, 2.03050972, 2.02743548, 2.02436124, 2.02128701,
		2.01821277, 2.01513853, 2.01206429, 2.00899005, 2.00591582, 2.00284158, 1.99976734, 1.99669310, 1.99361887, 1.99054463, 1.98747039, 1.98439615, 1.98132191, 1.97824768, 1.97517344, 1.97209920,
		1.96902496, 1.96595073, 1.96287649, 1.95980225, 1.95672801, 1.95365377, 1.95057954, 1.94750530, 1.94443106, 1.94135682, 1.93828258, 1.93520835, 1.93213411, 1.92905987, 1.92598563, 1.92291140,
		1.91983716, 1.91676292, 1.91368868, 1.91061444, 1.90754021, 1.90446597, 1.90139173, 1.89831749, 1.89524326, 1.89216902, 1.88909478, 1.88602054, 1.88294630, 1.87987207, 1.87679783, 1.87372359,
		1.87064935, 1.86757511, 1.86450088, 1.86142664, 1.85835240, 1.85527816, 1.85220393, 1.84912969, 1.84605545, 1.84298121, 1.83990697, 1.83683274, 1.83375850, 1.83068426, 1.82761002, 1.82453579,
		1.82146155, 1.81838731, 1.81531307, 1.81223883, 1.80916460, 1.80609036, 1.80301612, 1.79994188, 1.79686764, 1.79379341, 1.79071917, 1.78764493, 1.78457069, 1.78149646, 1.77842222, 1.77534798,
		1.77227374, 1.76919950, 1.76612527, 1.76305103, 1.75997679, 1.75690255, 1.75382832, 1.75075408, 1.74767984, 1.74460560, 1.74153136, 1.73845713, 1.73538289, 1.73230865, 1.72923441, 1.72616017,
		1.72308594, 1.72001170, 1.71693746, 1.71386322, 1.71078899, 1.70771475, 1.70464051, 1.70156627, 1.69849203, 1.69541780, 1.69234356, 1.68926932, 1.68619508, 1.68312085, 1.68004661, 1.67697237,
		1.67389813, 1.67082389, 1.66774966, 1.66467542, 1.66160118, 1.65852694, 1.65545270, 1.65237847, 1.64930423, 1.64622999, 1.64315575, 1.64008152, 1.63700728, 1.63393304, 1.63085880, 1.62778456,
		1.62471033, 1.62163609, 1.61856185, 1.61548761, 1.61241338, 1.60933914, 1.60626490, 1.60319066, 1.60011642, 1.59704219, 1.59396795, 1.59089371, 1.58781947, 1.58474523, 1.58167100, 1.57859676,
		1.57552252, 1.57244828, 1.56937405, 1.56629981, 1.56322557, 1.56015133, 1.55707709, 1.55400286, 1.55092862, 1.54785438, 1.54478014, 1.54170591, 1.53863167, 1.53555743, 1.53248319, 1.52940895,
		1.52633472, 1.52326048, 1.52018624, 1.51711200, 1.51403776, 1.51096353, 1.50788929, 1.50481505, 1.50174081, 1.49866658, 1.49559234, 1.49251810, 1.48944386, 1.48636962, 1.48329539, 1.48022115,
		1.47714691, 1.47407267, 1.47099844, 1.46792420, 1.46484996, 1.46177572, 1.45870148, 1.45562725, 1.45255301, 1.44947877, 1.44640453, 1.44333029, 1.44025606, 1.43718182, 1.43410758, 1.43103334,
		1.42795911, 1.42488487, 1.42181063, 1.41873639, 1.41566215, 1.41258792, 1.40951368, 1.40643944, 1.40336520, 1.40029097, 1.39721673, 1.39414249, 1.39106825, 1.38799401, 1.38491978, 1.38184554,
		1.37877130, 1.37569706, 1.37262282, 1.36954859, 1.36647435, 1.36340011, 1.36032587, 1.35725164, 1.35417740, 1.35110316, 1.34802892, 1.34495468, 1.34188045, 1.33880621, 1.33573197, 1.33265773,
		1.32958349, 1.32650926, 1.32343502, 1.32036078, 1.31728654, 1.31421231, 1.31113807, 1.30806383, 1.30498959, 1.30191535, 1.29884112, 1.29576688, 1.29269264, 1.28961840, 1.28654417, 1.28346993,
		1.28039569, 1.27732145, 1.27424721, 1.27117298, 1.26809874, 1.26502450, 1.26195026, 1.25887602, 1.25580179, 1.25272755, 1.24965331, 1.24657907, 1.24350484, 1.24043060, 1.23735636, 1.23428212,
		1.23120788, 1.22813365, 1.22505941, 1.22198517, 1.21891093, 1.21583670, 1.21276246, 1.20968822, 1.20661398, 1.20353974, 1.20046551, 1.19739127, 1.19431703, 1.19124279, 1.18816855, 1.18509432,
		1.18202008, 1.17894584, 1.17587160, 1.17279737, 1.16972313, 1.16664889, 1.16357465, 1.16050041, 1.15742618, 1.15435194, 1.15127770, 1.14820346, 1.14512923, 1.14205499, 1.13898075, 1.13590651,
		1.13283227, 1.12975804, 1.12668380, 1.12360956, 1.12053532, 1.11746108, 1.11438685, 1.11131261, 1.10823837, 1.10516413, 1.10208990, 1.09901566, 1.09594142, 1.09286718, 1.08979294, 1.08671871,
		1.08364447, 1.08057023, 1.07749599, 1.07442175, 1.07134752, 1.06827328, 1.06519904, 1.06212480, 1.05905057, 1.05597633, 1.05290209, 1.04982785, 1.04675361, 1.04367938, 1.04060514, 1.03753090,
		1.03445666, 1.03138243, 1.02830819, 1.02523395, 1.02215971, 1.01908547, 1.01601124, 1.01293700, 1.00986276, 1.00678852, 1.00371428, 1.00064005, 0.99756581, 0.99449157, 0.99141733, 0.98834310,
		0.98526886, 0.98219462, 0.97912038, 0.97604614, 0.97297191, 0.96989767, 0.96682343, 0.96374919, 0.96067496, 0.95760072, 0.95452648, 0.95145224, 0.94837800, 0.94530377, 0.94222953, 0.93915529,
		0.93608105, 0.93300681, 0.92993258, 0.92685834, 0.92378410, 0.92070986, 0.91763563, 0.91456139, 0.91148715, 0.90841291, 0.90533867, 0.90226444, 0.89919020, 0.89611596, 0.89304172, 0.88996748,
		0.88689325, 0.88381901, 0.88074477, 0.87767053, 0.87459630, 0.87152206, 0.86844782, 0.86537358, 0.86229934, 0.85922511, 0.85615087, 0.85307663, 0.85000239, 0.84692816, 0.84385392, 0.84077968,
		0.83770544, 0.83463120, 0.83155697, 0.82848273, 0.82540849, 0.82233425, 0.81926001, 0.81618578, 0.81311154, 0.81003730, 0.80696306, 0.80388883, 0.80081459, 0.79774035, 0.79466611, 0.79159187,
		0.78851764, 0.78544340, 0.78236916, 0.77929492, 0.77622068, 0.77314645, 0.77007221, 0.76699797, 0.76392373, 0.76084950, 0.75777526, 0.75470102, 0.75162678, 0.74855254, 0.74547831, 0.74240407,
		0.73932983, 0.73625559, 0.73318136, 0.73010712, 0.72703288, 0.72395864, 0.72088440, 0.71781017, 0.71473593, 0.71166169, 0.70858745, 0.70551321, 0.70243898, 0.69936474, 0.69629050, 0.69321626,
		0.69014203, 0.68706779, 0.68399355, 0.68091931, 0.67784507, 0.67477084, 0.67169660, 0.66862236, 0.66554812, 0.66247389, 0.65939965, 0.65632541, 0.65325117, 0.65017693, 0.64710270, 0.64402846,
		0.64095422, 0.63787998, 0.63480574, 0.63173151, 0.62865727, 0.62558303, 0.62250879, 0.61943456, 0.61636032, 0.61328608, 0.61021184, 0.60713760, 0.60406337, 0.60098913, 0.59791489, 0.59484065,
		0.59176641, 0.58869218, 0.58561794, 0.58254370, 0.57946946, 0.57639523, 0.57332099, 0.57024675, 0.56717251, 0.56409827, 0.56102404, 0.55794980, 0.55487556, 0.55180132, 0.54872709, 0.54565285,
		0.54257861, 0.53950437, 0.53643013, 0.53335590, 0.53028166, 0.52720742, 0.52413318, 0.52105894, 0.51798471, 0.51491047, 0.51183623, 0.50876199, 0.50568776, 0.50261352, 0.49953928, 0.49646504,
		0.49339080, 0.49031657, 0.48724233, 0.48416809, 0.48109385, 0.47801962, 0.47494538, 0.47187114, 0.46879690, 0.46572266, 0.46264843, 0.45957419, 0.45649995, 0.45342571, 0.45035147, 0.44727724,
		0.44420300, 0.44112876, 0.43805452, 0.43498029, 0.43190605, 0.42883181, 0.42575757, 0.42268333, 0.41960910, 0.41653486, 0.41346062, 0.41038638, 0.40731214, 0.40423791, 0.40116367, 0.39808943,
		0.39501519, 0.39194096, 0.38886672, 0.38579248, 0.38271824, 0.37964400, 0.37656977, 0.37349553, 0.37042129, 0.36734705, 0.36427282, 0.36119858, 0.35812434, 0.35505010, 0.35197586, 0.34890163,
		0.34582739, 0.34275315, 0.33967891, 0.33660467, 0.33353044, 0.33045620, 0.32738196, 0.32430772, 0.32123349, 0.31815925, 0.31508501, 0.31201077, 0.30893653, 0.30586230, 0.30278806, 0.29971382,
		0.29663958, 0.29356534, 0.29049111, 0.28741687, 0.28434263, 0.28126839, 0.27819416, 0.27511992, 0.27204568, 0.26897144, 0.26589720, 0.26282297, 0.25974873, 0.25667449, 0.25360025, 0.25052602,
		0.24745178, 0.24437754, 0.24130330, 0.23822906, 0.23515483, 0.23208059, 0.22900635, 0.22593211, 0.22285787, 0.21978364, 0.21670940, 0.21363516, 0.21056092, 0.20748669, 0.20441245, 0.20133821,
		0.19826397, 0.19518973, 0.19211550, 0.18904126, 0.18596702, 0.18289278, 0.17981855, 0.17674431, 0.17367007, 0.17059583, 0.16752159, 0.16444736, 0.16137312, 0.15829888, 0.15522464, 0.15215040,
		0.14907617, 0.14600193, 0.14292769, 0.13985345, 0.13677922, 0.13370498, 0.13063074, 0.12755650, 0.12448226, 0.12140803, 0.11833379, 0.11525955, 0.11218531, 0.10911107, 0.10603684, 0.10296260,
		0.09988836, 0.09681412, 0.09373989, 0.09066565, 0.08759141, 0.08451717, 0.08144293, 0.07836870, 0.07529446, 0.07222022, 0.06914598, 0.06607175, 0.06299751, 0.05992327, 0.05684903, 0.05377479,
		0.05070056, 0.04762632, 0.04455208, 0.04147784, 0.03840360, 0.03532937, 0.03225513, 0.02918089, 0.02610665, 0.02303242, 0.01995818, 0.01688394, 0.01380970, 0.01073546, 0.00766123, 0.00458699,
		0.00151275, -0.00156149, -0.00463573, -0.00770996, -0.01078420, -0.01385844, -0.01693268, -0.02000691, -0.02308115, -0.02615539, -0.02922963, -0.03230387, -0.03537810, -0.03845234, -0.04152658, -
		0.04460082, -0.04767505, -0.05074929, -0.05382353, -0.05689777, -0.05997201, -0.06304624, -0.06612048, -0.06919472, -0.07226896, -0.07534320, -0.07841743, -0.08149167, -0.08456591, -0.08764015, -
		0.09071438, -0.09378862, -0.09686286, -0.09993710, -0.10301134, -0.10608557, -0.10915981, -0.11223405, -0.11530829, -0.11838252, -0.12145676, -0.12453100, -0.12760524, -0.13067948, -0.13375371, -
		0.13682795, -0.13990219, -0.14297643, -0.14605067, -0.14912490, -0.15219914, -0.15527338, -0.15834762, -0.16142185, -0.16449609, -0.16757033, -0.17064457, -0.17371881, -0.17679304, -0.17986728, -
		0.18294152, -0.18601576, -0.18909000, -0.19216423, -0.19523847, -0.19831271, -0.20138695, -0.20446118, -0.20753542, -0.21060966, -0.21368390, -0.21675814, -0.21983237, -0.22290661, -0.22598085, -
		0.22905509, -0.23212932, -0.23520356, -0.23827780, -0.24135204, -0.24442628, -0.24750051, -0.25057475, -0.25364899, -0.25672323, -0.25979747, -0.26287170, -0.26594594, -0.26902018, -0.27209442, -
		0.27516865, -0.27824289, -0.28131713, -0.28439137, -0.28746561, -0.29053984, -0.29361408, -0.29668832, -0.29976256, -0.30283680, -0.30591103, -0.30898527, -0.31205951, -0.31513375, -0.31820798, -
		0.32128222, -0.32435646, -0.32743070, -0.33050494, -0.33357917, -0.33665341, -0.33972765, -0.34280189, -0.34587612, -0.34895036, -0.35202460, -0.35509884, -0.35817308, -0.36124731, -0.36432155, -
		0.36739579, -0.37047003, -0.37354427, -0.37661850, -0.37969274, -0.38276698, -0.38584122, -0.38891545, -0.39198969, -0.39506393, -0.39813817, -0.40121241, -0.40428664, -0.40736088, -0.41043512, -
		0.41350936, -0.41658359, -0.41965783, -0.42273207, -0.42580631, -0.42888055, -0.43195478, -0.43502902, -0.43810326, -0.44117750, -0.44425174, -0.44732597, -0.45040021, -0.45347445, -0.45654869, -
		0.45962292, -0.46269716, -0.46577140, -0.46884564, -0.47191988, -0.47499411, -0.47806835, -0.48114259, -0.48421683, -0.48729107, -0.49036530, -0.49343954, -0.49651378, -0.49958802, -0.50266225, -
		0.50573649, -0.50881073, -0.51188497, -0.51495921, -0.51803344, -0.52110768, -0.52418192, -0.52725616, -0.53033039, -0.53340463, -0.53647887, -0.53955311, -0.54262735, -0.54570158, -0.54877582, -
		0.55185006, -0.55492430, -0.55799854, -0.56107277, -0.56414701, -0.56722125, -0.57029549, -0.57336972, -0.57644396, -0.57951820, -0.58259244, -0.58566668, -0.58874091, -0.59181515, -0.59488939, -
		0.59796363, -0.60103786, -0.60411210, -0.60718634, -0.61026058, -0.61333482, -0.61640905, -0.61948329, -0.62255753, -0.62563177, -0.62870601, -0.63178024, -0.63485448, -0.63792872, -0.64100296, -
		0.64407719, -0.64715143, -0.65022567, -0.65329991, -0.65637415, -0.65944838, -0.66252262, -0.66559686, -0.66867110, -0.67174534, -0.67481957, -0.67789381, -0.68096805, -0.68404229, -0.68711652, -
		0.69019076, -0.69326500, -0.69633924, -0.69941348, -0.70248771, -0.70556195, -0.70863619, -0.71171043, -0.71478466, -0.71785890, -0.72093314, -0.72400738, -0.72708162, -0.73015585, -0.73323009, -
		0.73630433, -0.73937857, -0.74245281, -0.74552704, -0.74860128, -0.75167552, -0.75474976, -0.75782399, -0.76089823, -0.76397247, -0.76704671, -0.77012095, -0.77319518, -0.77626942, -0.77934366, -
		0.78241790, -0.78549214, -0.78856637, -0.79164061, -0.79471485, -0.79778909, -0.80086332, -0.80393756, -0.80701180, -0.81008604, -0.81316028, -0.81623451, -0.81930875, -0.82238299, -0.82545723, -
		0.82853146, -0.83160570, -0.83467994, -0.83775418, -0.84082842, -0.84390265, -0.84697689, -0.85005113, -0.85312537, -0.85619961, -0.85927384, -0.86234808, -0.86542232, -0.86849656, -0.87157079, -
		0.87464503, -0.87771927, -0.88079351, -0.88386775, -0.88694198, -0.89001622, -0.89309046, -0.89616470, -0.89923893, -0.90231317, -0.90538741, -0.90846165, -0.91153589, -0.91461012, -0.91768436, -
		0.92075860, -0.92383284, -0.92690708, -0.92998131, -0.93305555};
	for (int i = 0; i < Nx; ++i)
	{
		g[i] = G[i];
	};
}