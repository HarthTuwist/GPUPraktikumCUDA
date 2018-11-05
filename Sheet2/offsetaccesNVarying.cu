#include <chrono>
#include <iostream>



//Kernel definition
template<typename T>
__global__
void oftKernel (T* out,
		T* in,
		const unsigned int sd_size,
		const unsigned int block_size,
		const unsigned int I,
		const unsigned int L)
{
	const unsigned int sd_id = static_cast<int> (threadIdx.x / L); //automatically rounded down in int arithmetics
	const unsigned int id = threadIdx.x - sd_id * L;
	const unsigned int sd_start = blockIdx.x * blockDim.x * I + sd_id * L * I;
	
	for (unsigned int i = 0; i < I; i++)
	{
		const unsigned el_id = sd_start + i * L + id;
		((T*) out)[el_id] = ((T*) in)[el_id];
	//	out[el_id] = in[el_id]; 
//		((T*) out)[0] = ((T*) in)[0];
	}
}

int main () {
	using namespace std::chrono;

	std::cout << "np.array("; //output the results so that they can be read easily by python
	
	std::cout << "(";
	for (int n = 0; n <= 5; n++)
	{			
		std::cout << "(";
		for(int j = 0; j <= 10; j++)
		{
			unsigned int I = 1 << j;
			unsigned int size = 1 << 30;
			unsigned int L = 4;
			unsigned int N = 8 * (1<<n);
			unsigned int sd_size;
			int Tindx = 5;
			switch(Tindx)
			{
				case 1 :
					sd_size = I * L * sizeof(char);
					break;
				case 2 :
					sd_size = I * L * sizeof(short);
					break;
				case 3 :
					sd_size = I * L * sizeof(int);
					break;
				case 4 :
					sd_size = I * L * sizeof(int2);
					break;
				case 5 :
					sd_size = I * L * sizeof(int4);
					break;
			}
			
			unsigned int block_size = sd_size * N;
			unsigned int block_amount = size / block_size; 
		
			void* out;
			void* in;
			
			auto err1 = cudaMalloc(&out, block_size * block_amount);
			auto err2 = cudaMalloc(&in, block_size * block_amount);	
  
			//initArrays
			cudaMemset(in, 111, size);
			cudaMemset(out, 4, size);			
	
	//		size_t free;
	//		size_t total;
	//		auto err3 = cudaMemGetInfo(&free, &total);

                  	if (err2 != cudaSuccess)
			{	
                        	std::cout << "ERROR: " << cudaGetErrorString(err2) << std::endl;
  			}
	//		for (int x = 1; x <= 10; x++) {
	//			oftKernel<<<block_amount, L * N >>> (out, in, sd_size, block_size, I, L);	
	//			cudaDeviceSynchronize();
	//		}
	//		std::cout<<"free:" <<free << " total:" << total << " savedArrays: " << (total - free)/ (block_size * block_amount) << " j:" << j << " Tindx:" << Tindx << std::endl;		
		
	//		cudaFree(out);
	//		cudaFree(in);
			
			//make a warmup 
			switch(Tindx)
			{
				case 1 :
					oftKernel<<<block_amount, L * N >>> (static_cast<char*> (out), static_cast<char*> (in), sd_size, block_size, I, L);
					break;
				case 2 :
					oftKernel<<<block_amount, L * N >>> (static_cast<short*> (out), static_cast<short*> (in), sd_size, block_size, I, L);
					break;
				case 3 :
					oftKernel<<<block_amount, L * N >>> (static_cast<int*> (out), static_cast<int*> (in), sd_size, block_size, I, L);
					break;
				case 4 :
					oftKernel<<<block_amount, L * N >>> (static_cast<int2*> (out), static_cast<int2*> (in), sd_size, block_size, I, L);
					break;
				case 5 :
					oftKernel<<<block_amount, L * N >>> (static_cast<int4*> (out), static_cast<int4*> (in), sd_size, block_size, I, L);
					break;
			}
	
			cudaDeviceSynchronize();

			//Time Measururement Point 1
			high_resolution_clock::time_point timeBefore = high_resolution_clock::now();

			for(int x = 1; x <= 100; x++)//run 100 times for better measurement accuracy
			{
				switch(Tindx)
				{
					case 1 :
						oftKernel<<<block_amount, L * N >>> (static_cast<char*> (out), static_cast<char*> (in), sd_size, block_size, I, L);
						break;
					case 2 :
						oftKernel<<<block_amount, L * N >>> (static_cast<short*> (out), static_cast<short*> (in), sd_size, block_size, I, L);
						break;
					case 3 :
						oftKernel<<<block_amount, L * N >>> (static_cast<int*> (out), static_cast<int*> (in), sd_size, block_size, I, L);
						break;
					case 4 :
						oftKernel<<<block_amount, L * N >>> (static_cast<int2*> (out), static_cast<int2*> (in), sd_size, block_size, I, L);
						break;
					case 5 :
						oftKernel<<<block_amount, L * N >>> (static_cast<int4*> (out), static_cast<int4*> (in), sd_size, block_size, I, L);
						break;
				
				}
				
				cudaDeviceSynchronize();
			
				auto lstErr = cudaGetLastError();
				if ( cudaSuccess != lstErr )
				{
					std::cout <<"runningError:"<< lstErr  << ": " << cudaGetErrorString(lstErr)  << std::endl;
				}
			}
	//			oftKernel<<<block_amount, L * N >>> (out, in, sd_size, block_size, I, L);
				
		//		std::cout<< "size of out:" << sizeof(out)  <<  "tindx:" << Tindx << " block_amount:" << block_amount << " L:" << L << " N:" << N << " block_size: " << block_size  <<  std::endl;	

			//	cudaDeviceSynchronize();	
			//	oftKernel<<<block_amount, L * N >>> (static_cast<int4*> (out), static_cast<int4*> (in), sd_size, block_size, I, L);
		//		cudaDeviceSynchronize();
						
			
			//Time Measurement Point 2
			high_resolution_clock::time_point timeAfter = high_resolution_clock::now();			
		
			//Output Time Measurement Result
			duration<double> time_span = duration_cast<duration<double>>(timeAfter - timeBefore);
			
			std::cout << time_span.count();
	
			//Check for copy errors
			void* checkAry = malloc(size);
			cudaMemcpy(checkAry, out, size , cudaMemcpyDeviceToHost);

			for (int pos = 0; pos < size; pos++)
			{
                        	if (static_cast<char*> (checkAry)[pos] != 111)
			{
                                	std::cout << "Copy Misstake at:" << pos <<"with:"<< static_cast<int>(static_cast<char*> (checkAry)[pos])
                                	 << "instead of:" <<static_cast<int>( 111) << std::endl;
                                }
 			 }

	
			cudaFree(out);
			cudaFree(in);
			free(checkAry);
		
			if( j != 10) {std::cout << ",";} //output a , if we aren't the last element of the for loop	
		}
		
		std::cout << ")";

				
		if( n != 5) {std::cout << ",";} //output a , if we aren't the last element of the for loop	
	}
	
	std::cout << ")";
	
	std::cout << ")" << std::endl;


			
	return 0;
}


