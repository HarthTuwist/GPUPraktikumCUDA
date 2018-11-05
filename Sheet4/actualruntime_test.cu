// compile with -std=c++11 -O3 -lcurand
#include <iostream>
#include <cstdio>
#include <curand.h>

using std::cout;
using std::endl;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
	__device__ inline operator T*() {
		extern __shared__ int __smem[];
		return (T*) __smem;
	}

	__device__ inline operator const T*() const {
		extern __shared__ int __smem[];
		return (T*) __smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
	__device__ inline operator double*() {
		extern __shared__ double __smem_d[];
		return (double*) __smem_d;
	}

	__device__ inline operator const double*() const {
		extern __shared__ double __smem_d[];
		return (double*) __smem_d;
	}
};

template <class T>
__device__ void reduce(T* g_idata, T* sdata) {

	// load shared mem
	unsigned int tid = threadIdx.x;

	sdata[tid] = g_idata[tid];

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}

		__syncthreads();
	}
	// result is now in sdata[0]
}

// This kernel requires blockDim * sizeof(T) Bytes of shared memory.
// Each block processes `c * blockDim` elements.
// The buffers, which should be affected by the call to `__threadfence_system`
// must be volatile, as it is described in the CUDA C programming guide.
template <class T, bool COMMUNICATION_ON>
__global__ void producer_kernel(T* data, volatile T* partial_data, volatile unsigned* counter,
                                const unsigned c) {

	const unsigned global_start = blockIdx.x * blockDim.x * c;
	T* sdata = SharedMemory<T>();
	for (unsigned i = 0; i < c; ++i) {
		const unsigned offset = i * blockDim.x;
		const auto curr_start = data + global_start + offset;
		reduce(curr_start, sdata);
		// now we have the sum of blockDim elements in sdata[0]
		if (threadIdx.x == 0) {
			// save the mean of recently processed elements
			partial_data[blockIdx.x * c + i] = sdata[0] / (T) blockDim.x;
			if (COMMUNICATION_ON) {
				__threadfence_system();
				++counter[blockIdx.x]; // mark this block as processed
			}
		}
	}
}

template<class T, bool COMMUNICATION_ON>
__global__ void consumer_kernel(T* data, const volatile T* partial_data,
                                const volatile unsigned* counter,
                                const unsigned c) {
	__shared__ T mean;
	const unsigned global_start = blockIdx.x * blockDim.x * c;
	for (unsigned i = 0; i < c; ++i) {
		const unsigned offset = i * blockDim.x;
		if (COMMUNICATION_ON) {
			if (threadIdx.x == 0) {
				while (counter[blockIdx.x] < c) {}
				mean = partial_data[blockIdx.x * c + i];
			}
			__syncthreads();
			data[offset + global_start + threadIdx.x] = mean;
		}
		else { // no communication
			data[offset + global_start + threadIdx.x] = threadIdx.x;
		}
	}

}

int runMeasurement(
	const unsigned num_threads,
	const unsigned num_blocks,
	const unsigned c)
{
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	using DataT = float;

	//const unsigned num_threads = 32;
	//const unsigned num_blocks = 2;
	const size_t sh_mem_size = num_threads * sizeof(DataT);
	//const unsigned c = 2; // number of results per block
	//const size_t N = num_threads * num_blocks * c; // total number of elements
	const size_t size = c * num_threads * num_blocks * sizeof(DataT); // total size in Bytes

	//cout << "  - size = " << size / 1e9 << " GB" << endl;
	//cout << "  - num_threads = " << num_threads << endl;
	//cout << "  - num_blocks = " << num_blocks << endl;
	//cout << "  - sh_mem_size = " << sh_mem_size << endl;
	//cout << "  - N = " << N << endl;
	//cout << "  - c = " << c << endl;

	DataT* in_data;
	DataT* out_data;
	DataT* partial_data;
	unsigned* counter;

	cudaSetDevice(0);
	cudaMalloc(&in_data, size);
	cudaMallocManaged(&partial_data, num_blocks * c * sizeof(DataT));
	curandGenerateUniform(gen, (float*) in_data, size / sizeof(float)); // fill with random bits
	cudaDeviceSynchronize();
	cudaSetDevice(1);
	cudaMallocManaged(&counter, num_blocks * sizeof(unsigned));
	cudaMemAdvise(counter, num_blocks * sizeof(unsigned),
	              cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
	cudaMalloc(&out_data, size);
	cudaMemset(out_data, 0, size);
	cudaSetDevice(0);

//	cout << "  - Going to start the kernel" << endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	producer_kernel<DataT, true><<<num_blocks, num_threads, sh_mem_size>>>(in_data, partial_data, counter, c);
	cudaSetDevice(1);
	consumer_kernel<DataT, true><<<num_blocks, num_threads>>>(out_data, partial_data, counter, c);
	cudaSetDevice(0);
	gpuErrchk(cudaEventRecord(stop, 0));
	gpuErrchk(cudaEventSynchronize(stop));
	float time_in_ms;
	gpuErrchk(cudaEventElapsedTime(&time_in_ms, start, stop));
	gpuErrchk(cudaSetDevice(0));
	gpuErrchk(cudaDeviceSynchronize());

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(in_data);
	cudaFree(partial_data);
	cudaFree(counter);
	curandDestroyGenerator(gen);
	
	//cout << "time:" << time_in_ms << endl;
	cout << time_in_ms;
	return 0;
}

int main()
{	cout << "np.array(";
	int tArray[] = {512,1024};
	int tLength = 2;
	
	int bArray[] = {1,2,4,8};
	int bLength = 4;

	int cArray[] = {256, 1024, 4096, 16384, 65536};
	int cLength = 5;

	cout << "(";
	for (int t = 0; t < tLength; t++)
	{
	
		cout << "(";
		for (int b = 0; b < bLength; b++)
		{	
			cout << "(";
			for (int c = 0; c < cLength; c++)
			{
				//run kernel
				runMeasurement(tArray[t], bArray[b], cArray[c]);	
				if(c < cLength - 1)
				{
					cout << ","<<endl;
				}
			}
			cout << ")";
	
			if(b < bLength - 1)
			{
				cout << ",";
			}
		}
		cout << ")" ;
		

		if(t < tLength - 1)
		{
			cout << ",";
		}
	}
	cout << ")";

//	runMeasurement(512,4,4096); //b, t, c
	cout << ")" << std::endl;
}
