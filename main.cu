#include <iostream>
#include <random>
#include <time.h>
#include <math.h>
#include <limits>

struct RandomSpecs
{
  static constexpr int multiplier = 8253729;
  static constexpr int addend = 2396403;
  static constexpr int mod = 32767;
};


__host__ __device__
int random_int(int seed) {
  seed = (RandomSpecs::multiplier * seed + RandomSpecs::addend);
  return abs(seed % RandomSpecs::mod);
}

int log2_ceil(int n){
  float power2 = log2(n);
  float floor_power2 = floor(power2);
  if (abs(power2 - floor_power2) > std::numeric_limits<float>::epsilon()) {
    floor_power2 += 1;
  }

  return floor_power2;
}

__global__
void assign_random(int* x, int seed)
{
  int index = threadIdx.x;
  int rand_num = random_int(seed + index);
  x[index] = rand_num;
}

__global__
void assign_constant_from(int* x, int constant, int from)
{
  int index = threadIdx.x;
  x[index+from] = constant;
}

__global__
void bitonic_stage(int* x, int step, int stage)
{
  int thread_idx = threadIdx.x;
  int distance = int(pow(2, stage-1));
  int repeativness = int(pow(2, step-1));
  int first_idx = thread_idx + (thread_idx / distance) * distance;
  int second_idx = first_idx + distance;

  int first, second;
  first = min(x[first_idx], x[second_idx]);
  second = max(x[first_idx], x[second_idx]);
  int mod2 = (thread_idx / repeativness) % 2;
  if (mod2 % 2 == 1){
    int temp = first;
    first = second;
    second = temp;
  }

  x[first_idx] = first;
  x[second_idx] = second;
}

void print_array(int* x, int N)
{
  for (int i = 0; i < N; i++){
      std::cout << x[i] << ", ";
  }
  std::cout << std::endl;
}

void bitonic_sort(int* x, int N) // N should be a power of 2
{
  int power2 = int(log2(N));
  int workers = N/2;
  for (int step = 1; step <= power2; step++){
    for (int stage = step; stage > 0; stage--){
      bitonic_stage<<<1, workers>>>(x, step, stage);
      cudaDeviceSynchronize();
    }
  }
}

int main(void)
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int max_threads_num = prop.maxThreadsPerBlock;
  while(true){
    int N = 0;
    int real_size = 0;
    std::cout << "Enter the number of elements in the array [2-" << max_threads_num << "]: ";
    std::string input;
    std::cin >> N;

    if (N < 2 || N > max_threads_num){
      std::cout << "Invalid input." << std::endl;
      continue;
    }

    int real_size_power = log2_ceil(N);
    real_size = 1 << real_size_power;

    int* x = nullptr;
    cudaMallocManaged(&x, real_size*sizeof(int));

    assign_random<<<1, N>>>(x, time(NULL));
    assign_constant_from<<<1, real_size-N>>>(x, std::numeric_limits<int>::max(), N);
    cudaDeviceSynchronize();
    std::cout<<"Before sorting: ";
    print_array(x, N);

    bitonic_sort(x, real_size);
    std::cout<<"After sorting: ";
    print_array(x, N);
    cudaFree(x);
  }
}