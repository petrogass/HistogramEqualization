import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import cv2
from time import perf_counter

# CUDA kernel for histogram equalization
kernel = """
__global__ void histogram_calc(unsigned char *input_image, unsigned char *output_image, int *hist_gpu, int width, int height)
{ 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;    
     
    if (x < width && y < height)
    {
        int R = input_image[3*index];
        int G = input_image[3*index + 1];
        int B = input_image[3*index + 2];
        
        int Y =  0.299 * R + 0.587 * G + 0.114 * B;        
        int U = -0.169 * R - 0.331 * G + 0.499 * B + 128;        
        int V =  0.499 * R - 0.418 * G - 0.0813 * B + 128;
        
        output_image[3*index] = Y;
        output_image[3*index + 1] = U;
        output_image[3*index + 2] = V;                    
    
        atomicAdd(&(hist_gpu[Y]), 1);        
        }
        
    // Wait for all threads to finish updating the histogram
    __syncthreads();
}
    
    
__global__ void histogram_equalize(unsigned char *input_image, unsigned char *output_image, int *cdf, int cdfmin, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    
    if (x < width && y < height){
        int Y = int (output_image[3 * index]);
        int equalized_Y = (int)(255 * (float)(cdf[Y] - cdfmin) / (float)(width * height - cdfmin));
        output_image[3 * index] = equalized_Y;
    }
}    
__global__ void YUV_to_RGB(unsigned char *output_image, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    
    if (x < width && y < height){
        int Y = int(output_image[3*index]);
        int U = int(output_image[3*index + 1]);
        int V = int(output_image[3*index + 2]);
        
        int R = max(0, min(255, (int)(Y + 1.402*(V-128))));       
        int G = max(0, min(255, (int)(Y - 0.344*(U-128) - 0.714*(V-128))));        
        int B = max(0, min(255, (int)(Y + 1.772*(U- 128))));
        
        output_image[3*index] = (unsigned char)R;
        output_image[3*index + 1] = (unsigned char)G;
        output_image[3*index + 2] = (unsigned char)B;
    }
}

"""
if __name__ == "__main__":
   
    times = np.array([])
    
    for _ in range(10):
        # Load the image
        input_image_path = "m.jpg"
        input_image = cv2.imread(input_image_path)
        input_image = cv2.resize(input_image, (8000,8000), interpolation = cv2.INTER_NEAREST)
        #print(input_image)
        start_time = perf_counter()
        height, width = input_image.shape[:2]
        #print(height)
        #print(width)

        # Define the array for the histogram and the cdf
        hist = np.zeros(256, dtype=np.int32)
        cdf = np.zeros(256, dtype=np.int32)

        # Allocate memory on the GPU
        input_image_gpu = cuda.mem_alloc(input_image.nbytes)
        output_image_gpu = cuda.mem_alloc(input_image.nbytes)
        hist_gpu = cuda.mem_alloc(hist.nbytes)
        cdf_gpu = cuda.mem_alloc(cdf.nbytes)

        # Copy input image and histogram to the GPU 
        cuda.memcpy_htod(input_image_gpu, input_image)

        
        # Compile the CUDA kernel and get a reference to the functions
        mod = SourceModule(kernel)
        histogram_calc_kernel = mod.get_function("histogram_calc")
        histogram_equalize_kernel = mod.get_function("histogram_equalize")
        YUV_to_RGB_kernel = mod.get_function("YUV_to_RGB")
        
        # Set block and grid dimensions
        block_dim = (16, 16, 1)
        grid_dim = (int(np.ceil(width / block_dim[0])), int(np.ceil(height / block_dim[1])))
        #print(hist)
        
        # Make the histogram and copy it back to Host
        histogram_calc_kernel(input_image_gpu, output_image_gpu, hist_gpu, np.int32(width), np.int32(height), block=block_dim, grid=grid_dim)    
        cuda.memcpy_dtoh(hist, hist_gpu)
        output_image = np.empty_like(input_image)
        cuda.memcpy_dtoh(input_image, input_image_gpu)
        #print(input_image)

        # Calculate the cumulative distribution function (CDF) on CPU and copy it to Device
        cdf[0] = hist[0];                                                                  
        for i in range(256):                                                     
            cdf[i] = hist[i] + cdf[i-1]
        #print(cdf)

        cuda.memcpy_htod(cdf_gpu, cdf)
        for i in range(256):
            if (cdf[i] != 0):
                cdfmin = i
                break
    
        
        #print(cdfmin)
        

        # Equalize the image
        histogram_equalize_kernel(input_image_gpu, output_image_gpu, cdf_gpu, np.int32(cdfmin), np.int32(width), np.int32(height), block=block_dim, grid=grid_dim)
        
        # Convert the image back to RGB
        YUV_to_RGB_kernel(output_image_gpu, np.int32(width), np.int32(height), block=block_dim, grid=grid_dim)
        
        # Save the output image on host
        output_image = np.empty_like(input_image)
        cuda.memcpy_dtoh(output_image, output_image_gpu)
        
        # Free the memory
        
        cdf_gpu.free()
        hist_gpu.free()
        input_image_gpu.free()
        output_image_gpu.free()
        
        # Stop the timer
        end_time = perf_counter()
        #print(end_time - start_time)
        times = np.append(times, end_time - start_time)
        
        # Save the output image
        
        output_image_path = "output_image.jpg"
        cv2.imwrite(output_image_path, output_image)

    print("Average execution time: " + str(times.mean()*10**3) + ' milliseconds')
