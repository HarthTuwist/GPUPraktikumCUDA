#include <chrono>
#include <iostream>
#include <string>

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
void copyKernel(T* out, T* in, int stride) {
	unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
//	if (id*stride < sizeof(out)) {
		out[id*stride] = in[id*stride];
//	}
}


int main () {
	using namespace std::chrono;


	//output strings so that they may be pasted directly into python
	std::string keyString;
	std::string valueString;
	for (int i  = 0; i <= 6; i++) {//we are looping the numbers of blocks now
		
		int stride = 1 << i;

		
		testingStruct_4* out;
		testingStruct_4* in;
		int accessSize = sizeof (testingStruct_4);
	
		// TODO check total memory consumtpion and cout total size in GB	
 		cudaMalloc(&out , 16384 * 1024 *   accessSize * stride );
		auto err = cudaMalloc(&in , 16384 * 1024  *  accessSize *stride ) ;	
			
		if (err != CUDA_SUCCESS) {
			std::cout << "ERROR: could not alloc!" << std::endl;
}

	
		copyKernel<<<16384, 1024>>>(out, in, stride);
		cudaDeviceSynchronize();
		
		//Time Measururement Point 1
		high_resolution_clock::time_point timeBefore = high_resolution_clock::now();

		for (int i = 1; i <= 10; i++){
			copyKernel<<<16384, 1024>>>(out, in, stride);
			cudaDeviceSynchronize();
		}

		//Time Measurement Point 2
		high_resolution_clock::time_point timeAfter = high_resolution_clock::now();

		
		//Output Time Measurement Result
		duration<double> time_span = duration_cast<duration<double>>(timeAfter - timeBefore);
			
		std::cout << "Time for 10x stride = " << stride << " is "  << time_span.count() << " seconds";
		std::cout << std:: endl;

		cudaFree(out);
		cudaFree(in);
	}
	//std::cout << "))" << std::endl;
	return 0;

}

