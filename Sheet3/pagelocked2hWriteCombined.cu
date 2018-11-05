#include <chrono>
#include <iostream>



//Kernel definition
template<typename T>
__global__
void copyKernel (T* out,
		T* in,
		const unsigned int N)
{
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	for (unsigned int i= id; i < N; i = i + blockDim.x * gridDim.x)
	{
		const unsigned el_id = i;
		((T*) out)[el_id] = ((T*) in)[el_id];
	
//		((T*) out)[(1<<29) + 100] = ((T*) in)[0];

	}
}

int main () {
	using namespace std::chrono;

	unsigned int N = 1<<29; //N is the Number of elements in the Array
	double lastMeasurementTimeSpan = 100.0f;//we are not expecting measurements greater 100 s
	bool stopMeasurement = false;
	std::cout << "np.array("; //output the results so that they can be read easily by python
	
	std::cout << "(";
	for (int M = 1; M <= 4; M++)
	{			
		std::cout << "(";

		for(int i = 1; i <= 32; i++)
		{
	
			if(!stopMeasurement) 
			{
				unsigned int m = 32 * i; 
			//	int* carray;
				void* out;
				void* in;
			//	malloc(carray);
				
			//	auto err1 = cudaMallocHost(&out, N*4);
				auto err1 = cudaHostAlloc(&out, N*4, cudaHostAllocWriteCombined);
				auto err2 = cudaMalloc(&in, N*4);	
	  
				if (err1 != cudaSuccess)
				{	
					std::cout << "Allocation ERROR: " << cudaGetErrorString(err1) << std::endl;
				}
		
				if (err2 != cudaSuccess)
				{	
					std::cout << "Allocation ERROR2: " << cudaGetErrorString(err2) << std::endl;
				}	

				//make a warmup 
				copyKernel<<<M, m>>> (static_cast<int*> (out), static_cast<int*> (in), N);
				cudaDeviceSynchronize();

				//Time Measururement Point 1
				high_resolution_clock::time_point timeBefore = high_resolution_clock::now();

				for(int x = 1; x <= 10; x++)//run 10 times for better measurement accuracy
				{
					//run kernel here
					copyKernel<<<M, m>>> (static_cast<int*> (out), static_cast<int*> (in), N);
					cudaDeviceSynchronize();
				
					auto lstErr = cudaGetLastError();
					if ( cudaSuccess != lstErr )
					{
						std::cout << lstErr  << ": " << cudaGetErrorString(lstErr)  << std::endl;
					}
				}
			
				//Time Measurement Point 2
				high_resolution_clock::time_point timeAfter = high_resolution_clock::now();			
			
				//Output Time Measurement Result
				duration<double> time_span = duration_cast<duration<double>>(timeAfter - timeBefore);
				

		
				cudaFreeHost(out);
				cudaFree(in);
				
				//it seems we cant use automatic measurement stops
				if(false)// (lastMeasurementTimeSpan- time_span.count()  < 0.01 && i=1)
				{
					stopMeasurement = true;

				}
				else
				{
					lastMeasurementTimeSpan = time_span.count();
					std::cout << time_span.count();
				
				}

			}
		
			else
			{
				std::cout << 0.0;
			}

			if( i != 32) {std::cout << ",";} //output a , if we aren't the last element of the for loop	
		}
		
		std::cout << ")";

				
		if( M != 15) {std::cout << ",";} //output a , if we aren't the last element of the for loop	
	}
	
	std::cout << ")";
	
	std::cout << ")" << std::endl;


			
	return 0;
}


