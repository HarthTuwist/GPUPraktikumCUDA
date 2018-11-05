#include <chrono>
#include <iostream>

//Kernel Definition

__global__ void emptyKernel() {


}


int main () {
	using namespace std::chrono;

	//Call emptyKernel to get the starting cost out of the measurement 
	emptyKernel<<<1,1>>>();

	for (int n = 0; n <= 6; n++){
		//Time Measururement Point 1
		high_resolution_clock::time_point timeBefore = high_resolution_clock::now();
		
		unsigned I = pow(10,  n);
		
		for (unsigned i = 0; i < I; i++) {
			emptyKernel<<<1,1>>>();
			cudaDeviceSynchronize();
		}	
		
		//Time Measururement Point 1
		high_resolution_clock::time_point timeAfter = high_resolution_clock::now();

		//Output Time Measurement Result
		duration<double> time_span = duration_cast<duration<double>>(timeAfter - timeBefore);
		
		std::cout << "Time for 10^" << n <<  " kernels started: " << time_span.count() << " seconds";
		std::cout << std:: endl;
	}


	return 0;
}

