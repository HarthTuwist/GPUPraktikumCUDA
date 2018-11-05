#include <chrono>
#include <iostream>
#include <stdlib.h>

//Kernel Definition

__global__ void emptyKernel() {


}


int main () {
	using namespace std::chrono;
/*
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
*/
	for(int x = 1; x <= 2; x++) {// Handles whetherDeviceSynchronize is included or not
		for (int copyvalue = 1; copyvalue <= 4; copyvalue++) { //this loop handles the 4 different ways of copying with cudaMemcpy()
			
			int j = 5;
			size_t size = j * (1 << 27); // = 512 Mebibyte
			
			//Allocate arrays on the device and on the host
			void* src_h;
			void* dst_h;
			void* src_d;
			void* dst_d;

			cudaMalloc(&src_d, size);
			cudaMalloc(&dst_d, size);
			src_h = malloc(size);
			dst_h = malloc(size);

			//Time Measurement Point 1
			high_resolution_clock::time_point timeBefore = high_resolution_clock::now();	
			
			for (int i = 1; i <= 10; i++){//run 10 times to get better accuracy
				if (copyvalue == 1) cudaMemcpy(dst_d, src_h, size, cudaMemcpyHostToDevice); //host to device
				if (copyvalue == 2) cudaMemcpy(dst_h, src_d, size, cudaMemcpyDeviceToHost); //device to host
				if (copyvalue == 3) cudaMemcpy(dst_h, src_h, size, cudaMemcpyHostToHost); //host to host
				if (copyvalue == 4) cudaMemcpy(dst_d, src_d, size, cudaMemcpyDeviceToDevice); //device to device
				
				if (x==1) cudaDeviceSynchronize();
			}
			
			//Time Measurement Point 2
			high_resolution_clock::time_point timeAfter = high_resolution_clock::now();
			
			//Output Time Measurement Result
			duration<double> time_span = duration_cast<duration<double>>(timeAfter - timeBefore);
			
			std::cout << "Time for 10x copy with j = " << j <<  " and ";
			if (copyvalue == 1) {std::cout << "Host to Device: ";}
			if (copyvalue == 2) {std::cout << "Device to Host: ";}
			if (copyvalue == 3) {std::cout << "Host to Host: ";}
			if (copyvalue == 4) {std::cout << "Device to Device: ";}
			std::cout << time_span.count() << " seconds";
			if (x == 2) {std::cout << " w/o DeviceSynchronize";}
			
			std::cout << std:: endl;


			cudaFree(src_d);
			cudaFree(dst_d);
			free(src_h);
			free(dst_h);
		}
	}

	return 0;
}
