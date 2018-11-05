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

template<typename T>
__global__
void initKernel (T* out)
{
	((T*) out)[threadIdx.x] = threadIdx.x;
}


int main () {
	using namespace std::chrono;
	unsigned int N = 1<<29;
	void* out;
	void* in;
	auto err1 = cudaMallocHost(&out, N*4);
	auto err2 = cudaMalloc(&in, N*4);

	initKernel<<<1, N>>> (static_cast<int*> (in));
	cudaDeviceSynchronize();
	
	cudaMemcpy(in, out, N*4, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++) {
		std::cout<< i << "," << static_cast<int*>(out)[i]   << std::endl;
	}

	return 0;
}


