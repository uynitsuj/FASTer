import cupy as cp
import cupyx.scipy.fft as cufft
import scipy.fft    
import time
import numpy as np

def benchmark_dct_implementations(size=1024, num_trials=100):
    """
    Benchmark DCT/IDCT implementations between scipy and cuFFT
    Args:
        size: Size of square matrix to transform
        num_trials: Number of trials to average over
    """
    # Generate random test data
    data = np.random.random((size, size)).astype(np.float32)
    data_gpu = cp.asarray(data)

    # Benchmark scipy DCT
    start = time.time()
    for _ in range(num_trials):
        scipy_dct = scipy.fft.dct(data)
        scipy_idct = scipy.fft.idct(scipy_dct)
    scipy_time = (time.time() - start) / num_trials

    # Benchmark cuFFT DCT 
    start = time.time()
    for _ in range(num_trials):
        with scipy.fft.set_backend(cufft):
            gpu_dct = scipy.fft.dct(data_gpu) 
            gpu_idct = scipy.fft.idct(gpu_dct)
    gpu_time = (time.time() - start) / num_trials
    
    # Print results
    print(f"\nBenchmark results for {size}x{size} matrix averaged over {num_trials} trials:")
    print(f"SciPy DCT+IDCT time: {scipy_time*1000:.2f} ms")
    print(f"cuFFT DCT+IDCT time: {gpu_time*1000:.2f} ms")
    print(f"Speedup: {scipy_time/gpu_time:.2f}x")
    
    # Verify results match
    scipy_result = scipy.fft.dct(data)
    with scipy.fft.set_backend(cufft):
        gpu_result = cp.asnumpy(scipy.fft.dct(data))
    max_diff = np.max(np.abs(scipy_result - gpu_result))
    print(f"\nMax difference between implementations: {max_diff:.2e}")

if __name__ == "__main__":
    # Test different matrix sizes
    for size in [128, 256, 512, 1024, 2048]:
        benchmark_dct_implementations(size=size)
