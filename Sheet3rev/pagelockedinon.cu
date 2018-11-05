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

//template<typename T>
//__global__
//void initKernel (T* out)
//{
//	((T*) out)[threadIdx.x] = threadIdx.x;
//}


//Kernel definition
template<typename T>
__global__
void plusKernel (T* out,
		T* in,
		const unsigned int N)
{
	const unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
	for (unsigned int i= id; i < N; i = i + blockDim.x * gridDim.x)
	{
		const unsigned el_id = i;
		((T*) out)[el_id] += ((T*) in)[el_id];


	}
}


int main (int argc, char * argv[]) {

	std::cout<<"Arguments:";
	for (int i = 0; i <argc; i++)
	{
		std::cout<<argv[i];
	}	
	std::cout<<std::endl;
	if(
		argc != 4
	)
	{
		std::cout<<"Error: InvalidArguments" << std::endl;
	}
	
	
	using namespace std::chrono;

	unsigned int N = 1<<29; //N is the Number of elements in the Array
	double lastMeasurementTimeSpan = 100.0f;//we are not expecting measurements greater 100 s
	bool stopMeasurement = false;
	std::cout << "np.array("; //output the results so that they can be read easily by python
	bool usePlusKernel = false;
		
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
				void* deviceArray;
				void* hostArray;
			//	malloc(carray);
					
				cudaError_t err1 = cudaSuccess;
				cudaError_t err2 = cudaSuccess;	  
				
				
				//standard allocation
				if(strcmp(argv[2],"-standard") == 0)
				{
					err1 = cudaMallocHost(&hostArray, N*4);
					err2 = cudaMalloc(&deviceArray, N*4);	  
				}
				
				//writeCombined
				if(strcmp(argv[2],"-writecombined") == 0)
				{
					err1 = cudaHostAlloc(&hostArray, N*4, cudaHostAllocWriteCombined);
					err2 = cudaMalloc(&deviceArray, N*4);	  
				}
	
				//unifiedMemorz
				if(strcmp(argv[2],"-unified") == 0)
				{
					err1 = cudaMallocManaged(&hostArray, N*4);
					err2 = cudaMallocManaged(&deviceArray, N*4);	  
				}

				if (err1 != cudaSuccess)
				{	
					std::cout << "Allocation ERROR: " << cudaGetErrorString(err1) << std::endl;
				}
		
				if (err2 != cudaSuccess)
				{	
					std::cout << "Allocation ERROR2: " << cudaGetErrorString(err2) << std::endl;
				}	

				
				if(strcmp(argv[1],"-h2d") == 0)
				{
					in = hostArray;
					out = deviceArray;	  
				}
				
				else
				{
					in = deviceArray;
					out = hostArray;
				
				}
		
		
				if(strcmp(argv[3],"-plus") == 0)
				{
					usePlusKernel = true;
				}
	
				if(strcmp(argv[3],"-copy") == 0)
				{
					usePlusKernel = false;
				}

				//std::cout << "in:" << in << "out:" << out << "hostArray:" << hostArray << "deviceArray:" << deviceArray;	
	

				//make a warmup 
			//	copyKernel<<<M, m>>> (static_cast<int*> (out), static_cast<int*> (in), N);
			//	cudaDeviceSynchronize();

				double currentTimeSum = 0;
				
				for(int x = 1; x <= 5; x++)//run 10 times for better measurement accuracy
				{
				//	if(strcmp(argv[2],"-unified") == 0)
				//	{
						if(strcmp(argv[1],"-h2d") == 0)
						{
							cudaMemset(out, 4, N*4);
							cudaDeviceSynchronize();
							memset(in, 111, N*4);
							cudaDeviceSynchronize();	
						}
					
						if(strcmp(argv[1],"-d2h") == 0)
						{
							cudaMemset(in, 111, N*4);
							cudaDeviceSynchronize();
							memset(out, 4, N*4);
							cudaDeviceSynchronize();
						}
				//	}

					//Time Measururement Point 1
					high_resolution_clock::time_point timeBefore = high_resolution_clock::now();

					//run kernel here
					if (usePlusKernel)
					{
						plusKernel<<<M, m>>> (static_cast<int*> (out), static_cast<int*> (in), N);
					}
					else
					{
						copyKernel<<<M, m>>> (static_cast<int*> (out), static_cast<int*> (in), N);
					}

					cudaDeviceSynchronize();
				
					//Time Measurement Point 2
					high_resolution_clock::time_point timeAfter = high_resolution_clock::now();			
			
					auto lstErr = cudaGetLastError();
					if ( cudaSuccess != lstErr )
					{
						std::cout << lstErr  << ": " << cudaGetErrorString(lstErr)  << std::endl;
					}
						
					//Output Time Measurement Result
					duration<double> time_span = duration_cast<duration<double>>(timeAfter - timeBefore);
					currentTimeSum += time_span.count();

					cudaDeviceSynchronize();
				
				}
				
				if(false)
				{	
				//perform error checking
					void* checkAry = out;
					if(strcmp(argv[1],"-h2d") == 0)
					{
					//copy Out to In so we can read it on the host
				//	
						cudaMemcpy(in, out, N*4, cudaMemcpyDeviceToHost); 
						cudaDeviceSynchronize();
						checkAry = in;//as we copied to in, we need to check this now
						
					}

					char targetChar = 111;
					if(strcmp(argv[3],"-plus") == 0)
					{
						targetChar = 111 + 4;	
					}

					for (int pos = 0; pos < N*4; pos++)
					{

						if (static_cast<char*> (checkAry)[pos] != targetChar)
						{
							std::cout << "Copy Misstake at:" << pos <<"with:"<< static_cast<int>(static_cast<char*> (checkAry)[pos])
							<< "instead of:" <<static_cast<int>( targetChar) << std::endl;
						}
						
					}	
				}

				if(strcmp(argv[2],"-unified") == 0)
				{
					cudaFree(deviceArray);
					cudaFree(hostArray);
				
				}
				else
				{
					cudaFreeHost(hostArray);
					cudaFree(deviceArray);
				}
					
				//it seems we cant use automatic measurement stops
				if(false)// (lastMeasurementTimeSpan- time_span.count()  < 0.01 && i=1)
				{
					stopMeasurement = true;

				}
				else
				{
				//	lastMeasurementTimeSpan = time_span.count();
					std::cout << currentTimeSum;
				
				}

			}
		
			else
			{
				std::cout << 0.0;
			}

			if( i != 32) {std::cout << ",";} //output a , if we aren't the last element of the for loop	
		}
		
		std::cout << ")";

				
		if( M != 4) {std::cout << ",";} //output a , if we aren't the last element of the for loop	
	}
	
	std::cout << ")";
	
	std::cout << ")" << std::endl;


			
	return 0;
}


