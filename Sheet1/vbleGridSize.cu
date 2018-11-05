#include <chrono>
#include <iostream>

//Testing Structs, easiest way to define datatypes of size

typedef struct 
{//just define an amount of chars in the struct that is equal to the desired size
	char p1;
}testingStruct_1;

typedef struct 
{//just define an amount of chars in the struct that is equal to the desired size
	char p1;
	char p2;
}testingStruct_2;

typedef struct 
{//just define an amount of chars in the struct that is equal to the desired size
	char p1;
	char p2;
	char p3;
	char p4;
}testingStruct_4;

typedef struct 
{//just define an amount of chars in the struct that is equal to the desired size
	char p1;
	char p2;
	char p3;
	char p4;
	char p5;
	char p6;
	char p7;
	char p8;
}testingStruct_8;

typedef struct 
{//just define an amount of chars in the struct that is equal to the desired size
	char p1;
	char p2;
	char p3;
	char p4;
	char p5;
	char p6;
	char p7;
	char p8;
	char p9;
	char p10;
	char p11;
	char p12;
	char p13;
	char p14;
	char p15;
	char p16;
}testingStruct_16;


//Kernel definition
template<typename T>
__global__
void copyKernel(T* out, T* in) {
	unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

	out[id] = in[id];
}


int main () {
	using namespace std::chrono;

	//Call emptyKernel to get the starting cost out of the measurement 
	//copyKernel<<<1,1>>>();
	
	/*
	for (int n = 0; n <= 6; n++){
		//Time Measururement Point 1
		high_resolution_clock::time_point timeBefore = high_resolution_clock::now();
		
		unsigned I = pow(10,  n);
		
		for (unsigned i = 0; i < I; i++) {
			emptyKernel<<<1,1>>>();
		}	
		cudaDeviceSynchronize();

		//Time Measururement Point 1
		high_resolution_clock::time_point timeAfter = high_resolution_clock::now();

		//Output Time Measurement Result
		duration<double> time_span = duration_cast<duration<double>>(timeAfter - timeBefore);
		
		std::cout << "Time for 10^" << n <<  " kernels started: " << time_span.count() << " seconds";
		std::cout << std:: endl;
	}

	*/
	for (int n = 4; n <= 14; n++) {//we are looping the numbers of blocks now
		int numBlocks = 2 << n; 
		int block_len;
			
		block_len = 1024;

		//different access sizes go here; its easiest to just change the code here before the measurement
		testingStruct_16* out;
		testingStruct_16* in;
		int accessSize = sizeof (testingStruct_16);
		
		cudaMalloc(&out , numBlocks * block_len * accessSize);
		cudaMalloc(&in , numBlocks * block_len * accessSize);	
			

		// Make a warmup
		copyKernel<<<numBlocks, block_len>>>(out, in);
		cudaDeviceSynchronize();
		
		//Time Measururement Point 1
		high_resolution_clock::time_point timeBefore = high_resolution_clock::now();

		for (int i = 1; i <= 10; i++){
			copyKernel<<<numBlocks, block_len>>>(out, in);
			cudaDeviceSynchronize();
		}

		//Time Measurement Point 2
		high_resolution_clock::time_point timeAfter = high_resolution_clock::now();

		
		//Output Time Measurement Result
		duration<double> time_span = duration_cast<duration<double>>(timeAfter - timeBefore);
			
		std::cout << "Time for 10x numBLocks = " << numBlocks <<  " block_len = " << block_len << " accessSize = " << accessSize << " is "  << time_span.count() << " seconds";
		std::cout << std:: endl;
	}

	return 0;

}

