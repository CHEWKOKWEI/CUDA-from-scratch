#include <iostream>
#include <chrono>
#include "ImageUtils.cpp"
#include "ArrayAdd.cuh"
#include "CheckIndex.cuh"
#include "Flip.cuh"
#include "Transpose.cuh"
#include "Convolution.cuh"
#include "Sort.cuh"

std::string RESOURCE_MAINDIR("C:\\Users\\kok-wei.chew\\Documents\\Visual Studio 2015\\Projects\\CudaExample\\resources");

void example_arrayAdd()
{
	std::cout << "CUDA Example - Array Addition" << std::endl;
	int cuda_dev_id = 0;
	int n_data = 2098;
	float* arr1 = new float[n_data];
	float* arr2 = new float[n_data];
	float* result_cpu = new float[n_data];
	float* result_gpu = new float[n_data];
	for (int i = 0; i < n_data; i++)
	{
		arr1[i] = (float)i;
		arr2[i] = (float)(2*(n_data-i));
	}
	arrayAdd(result_cpu, arr1, arr2, n_data);
	arrayAddCuda(result_gpu, arr1, arr2, n_data, cuda_dev_id);
	std::cout << "    CPU \t\t\tGPU" << std::endl;
	for (int i = 0; i < n_data; i+=(n_data/20))
	{
		std::cout << " " << arr1[i] << "+" << arr2[i] << "=" << result_cpu[i] << "\t\t";
		std::cout << " " << arr1[i] << "+" << arr2[i] << "=" << result_gpu[i] << std::endl;
	}
	delete[] arr1, arr2, result_cpu, result_gpu;
	std::cout << std::endl;
}

void example_flip()
{
	std::cout << "CUDA Example - Image Flipping" << std::endl;
	std::string img_filepath = RESOURCE_MAINDIR + std::string("\\flip\\before.png");
	std::string img_savepath1 = RESOURCE_MAINDIR + std::string("\\flip\\after(cpu).png");
	std::string img_savepath2 = RESOURCE_MAINDIR + std::string("\\flip\\after(gpu).png");
	int const num_byte = sizeof(float);
	float* img0 = nullptr;
	int nx, ny;
	if (!readImage(img_filepath.c_str(), img0, nx, ny))
	{
		std::cout << "Can't read image file '" << img_filepath << "'." << std::endl;
		return;
	}
	std::cout << "Image size: (nx,ny)=(" << nx << "," << ny << ")" << std::endl;
	float* img1 = new float[nx*ny];
	std::memcpy(&img1[0], &img0[0], num_byte*nx*ny);
	std::cout << "Flip image using CPU." << std::endl;
	//flipVertical(img1, nx, ny);
	flipHorizontal(img1, nx, ny);
	if (!saveImage(img_savepath1.c_str(), img1, nx, ny))
	{
		std::cout << "Can't save image to '" << img_savepath1 << "'." << std::endl;
		return;
	}
	std::cout << "Image saved to '" << img_savepath1 << "'." << std::endl;

	float* img2 = new float[nx*ny];
	std::memcpy(&img2[0], &img0[0], num_byte*nx*ny);
	std::cout << "Flip image using GPU." << std::endl;
	//flipVerticalCuda(img2, nx, ny);
	flipHorizontalCuda(img2, nx, ny);
	if (!saveImage(img_savepath2.c_str(), img2, nx, ny))
	{
		std::cout << "Can't save image to '" << img_savepath2 << "'." << std::endl;
		return;
	}
	std::cout << "Image saved to '" << img_savepath2 << "'." << std::endl;
	delete[] img0, img1, img2;
	std::cout << std::endl;
}

void example_transpose()
{
	std::cout << "CUDA Example - Image Transpose" << std::endl;
	std::string img_filepath = RESOURCE_MAINDIR + std::string("\\transpose\\before.png");
	std::string img_savepath1 = RESOURCE_MAINDIR + std::string("\\transpose\\after(cpu).png");
	std::string img_savepath2 = RESOURCE_MAINDIR + std::string("\\transpose\\after(gpu).png");
	int const num_byte = sizeof(float);
	float* img0 = nullptr;
	int nx, ny;
	if (!readImage(img_filepath.c_str(), img0, nx, ny)) 
	{
		std::cout << "Can't read image file '"<< img_filepath <<"'." << std::endl;
		return;
	}
	std::cout << "Image size: (nx,ny)=(" << nx << "," << ny << ")" << std::endl;
	float* img1 = new float[nx*ny];
	std::memcpy(&img1[0], &img0[0], num_byte*nx*ny);
	std::cout << "Transpose image using CPU." << std::endl;
	transpose(img1, nx, ny);
	if (!saveImage(img_savepath1.c_str(), img1, ny, nx))
	{
		std::cout << "Can't save image to '" << img_savepath1 << "'." << std::endl;
		return;
	}
	std::cout << "Image saved to '" << img_savepath1 << "'." << std::endl;

	float* img2 = new float[nx*ny];
	std::memcpy(&img2[0], &img0[0], num_byte*nx*ny);
	std::cout << "Transpose image using GPU." << std::endl;
	transposeCuda(img2, nx, ny);
	if (!saveImage(img_savepath2.c_str(), img2, ny, nx))
	{
		std::cout << "Can't save image to '" << img_savepath2 << "'." << std::endl;
		return;
	}
	std::cout << "Image saved to '" << img_savepath2 << "'." << std::endl;
	delete[] img0, img1, img2;
	std::cout << std::endl;
}

void example_convolution2d()
{
	std::cout << "CUDA Example - Image Convolve" << std::endl;
	std::string img_filepath = RESOURCE_MAINDIR + std::string("\\convolution\\before.png");
	std::string img_savepath1 = RESOURCE_MAINDIR + std::string("\\convolution\\after(cpu).png");
	std::string img_savepath2 = RESOURCE_MAINDIR + std::string("\\convolution\\after(gpu1).png");
	std::string img_savepath3 = RESOURCE_MAINDIR + std::string("\\convolution\\after(gpu2).png");
	std::chrono::steady_clock::time_point time_start, time_end;
	float duration_cpu = 0;
	float duration_gpu1 = 0;
	float duration_gpu2 = 0;
	int kern_nx = 11, kern_ny = 11;
	float* kernel = new float[kern_nx*kern_ny];
	for (int i = 0; i < kern_nx*kern_ny; i++) { kernel[i] = 1.0 / (float)kern_nx / (float)kern_ny; }
	int nx, ny;
	float* img0 = nullptr;
	if (!readImage(img_filepath.c_str(), img0, nx, ny))
	{
		std::cout << "Can't read image file '" << img_filepath << "'." << std::endl;
		return;
	}
	std::cout << "Image size:  (nx,ny)=(" << nx << "," << ny << ")" << std::endl;
	std::cout << "Kernel size: (nx,ny)=(" << kern_nx << "," << kern_ny << ")" << std::endl;
	int conv_nx = nx - kern_nx + 1;
	int conv_ny = ny - kern_ny + 1;
	float* img1 = new float[conv_nx*conv_ny];
	std::cout << "Convlove image using CPU." << std::endl;
	time_start = std::chrono::steady_clock::now();
	convolve2d(img1, img0, kernel, nx, ny, kern_nx, kern_ny);
	time_end = std::chrono::steady_clock::now();
	duration_cpu = (float)((time_end - time_start).count()) / 1000000.0;
	if (!saveImage(img_savepath1.c_str(), img1, conv_nx, conv_ny))
	{
		std::cout << "Can't save image to '" << img_savepath1 << "'." << std::endl;
		return;
	}
	std::cout << "Image saved to '" << img_savepath1 << "'." << std::endl;

	float* img2 = new float[conv_nx*conv_ny];
	std::cout << "Convlove image using GPU." << std::endl;
	time_start = std::chrono::steady_clock::now();
	convolve2dCuda(img2, img0, kernel, nx, ny, kern_nx, kern_ny);
	time_end = std::chrono::steady_clock::now();
	duration_gpu1 = (float)((time_end - time_start).count()) / 1000000.0;
	if (!saveImage(img_savepath2.c_str(), img2, conv_nx, conv_ny))
	{
		std::cout << "Can't save image to '" << img_savepath2 << "'." << std::endl;
		return;
	}
	std::cout << "Image saved to '" << img_savepath2 << "'." << std::endl;

	float* img3 = new float[conv_nx*conv_ny];
	std::cout << "Convlove image using GPU." << std::endl;
	time_start = std::chrono::steady_clock::now();
	convolve2dCuda2(img3, img0, kernel, nx, ny, kern_nx, kern_ny);
	time_end = std::chrono::steady_clock::now();
	duration_gpu2 = (float)((time_end - time_start).count()) / 1000000.0;
	if (!saveImage(img_savepath3.c_str(), img2, conv_nx, conv_ny))
	{
		std::cout << "Can't save image to '" << img_savepath3 << "'." << std::endl;
		return;
	}
	std::cout << "Image saved to '" << img_savepath3 << "'." << std::endl;

	std::cout << "Complete convolution using CPU in " << duration_cpu << "ms." << std::endl;
	std::cout << "Complete convolution using GPU(configuration1) in " << duration_gpu1 << "ms." << std::endl;
	std::cout << "Complete convolution using GPU(configuration2) in " << duration_gpu2 << "ms." << std::endl;
	delete[] img0, img1, img2, img3;
	std::cout << std::endl;
}


void example_sort_cpu()
{
	std::cout << "CUDA Example - Array Sorting (CPU)" << std::endl;
	std::string imgdirpath = RESOURCE_MAINDIR + std::string("\\sort\\tmp");
	std::string savepath_x = RESOURCE_MAINDIR + std::string("\\sort\\after_x");
	std::string savepath_y = RESOURCE_MAINDIR + std::string("\\sort\\after_y");
	std::string savepath_z = RESOURCE_MAINDIR + std::string("\\sort\\after_z");
	std::chrono::steady_clock::time_point time_start, time_end;
	float duration = 0;
	int algo = 4;
	int const num_byte = sizeof(float);
	float* arr0 = nullptr;
	int nx, ny, nz;
	if (!readStack(imgdirpath.c_str(), arr0, nx, ny, nz))
	{
		std::cout << "Can't read image stack from '" << imgdirpath << "'." << std::endl;
		return;
	}
	std::cout << "Array size: (nx,ny,nz)=(" << nx << "," << ny << "," << nz << ")" << std::endl;

	float* arr_x = new float[nx*ny*nz];
	std::memcpy(&arr_x[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sort(arr_x, 'x', nx, ny, nz, algo)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along x-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_x.c_str(), arr_x, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_x << "'." << std::endl; }

	float* arr_y = new float[nx*ny*nz];
	std::memcpy(&arr_y[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sort(arr_y, 'y', nx, ny, nz, algo)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along y-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_y.c_str(), arr_y, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_y << "'." << std::endl; }

	float* arr_z = new float[nx*ny*nz];
	std::memcpy(&arr_z[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sort(arr_z, 'z', nx, ny, nz, algo)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along z-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_z.c_str(), arr_z, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_z << "'." << std::endl; }

	delete[] arr0, arr_x, arr_y, arr_z;
	std::cout << std::endl;
}

void example_sort_gpu()
{
	std::cout << "CUDA Example - Array Sorting (GPU)" << std::endl;
	std::string imgdirpath = RESOURCE_MAINDIR + std::string("\\sort\\tmp");
	std::string savepath_x = RESOURCE_MAINDIR + std::string("\\sort\\after_x");
	std::string savepath_y = RESOURCE_MAINDIR + std::string("\\sort\\after_y");
	std::string savepath_z = RESOURCE_MAINDIR + std::string("\\sort\\after_z");
	std::chrono::steady_clock::time_point time_start, time_end;
	float duration = 0;
	int algo = 0;
	int const num_byte = sizeof(float);
	float* arr0 = nullptr;
	int nx, ny, nz;
	if (!readStack(imgdirpath.c_str(), arr0, nx, ny, nz))
	{
		std::cout << "Can't read image stack from '" << imgdirpath << "'." << std::endl;
		return;
	}
	std::cout << "Array size: (nx,ny,nz)=(" << nx << "," << ny << "," << nz << ")" << std::endl;

	float* arr_x = new float[nx*ny*nz];
	std::memcpy(&arr_x[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortSimpleCuda(arr_x, 'x', nx, ny, nz)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along x-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_x.c_str(), arr_x, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_x << "'." << std::endl; }

	float* arr_y = new float[nx*ny*nz];
	std::memcpy(&arr_y[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortSimpleCuda(arr_y, 'y', nx, ny, nz)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along y-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_y.c_str(), arr_y, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_y << "'." << std::endl; }

	float* arr_z = new float[nx*ny*nz];
	std::memcpy(&arr_z[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortSimpleCuda(arr_z, 'z', nx, ny, nz)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along z-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_z.c_str(), arr_z, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_z << "'." << std::endl; }

	delete[] arr0, arr_x, arr_y, arr_z;
	std::cout << std::endl;
}

void example_sorting_network()
{
	std::cout << "CUDA Example - Sorting Network" << std::endl;
	std::string imgdirpath = RESOURCE_MAINDIR + std::string("\\sort\\tmp");
	std::string savepath_x = RESOURCE_MAINDIR + std::string("\\sort\\after_x1");
	std::string savepath_y = RESOURCE_MAINDIR + std::string("\\sort\\after_y1");
	std::string savepath_z = RESOURCE_MAINDIR + std::string("\\sort\\after_z1");
	std::chrono::steady_clock::time_point time_start, time_end;
	float duration = 0;
	int const num_byte = sizeof(float);
	float* arr0 = nullptr;
	int nx, ny, nz;
	if (!readStack(imgdirpath.c_str(), arr0, nx, ny, nz))
	{
		std::cout << "Can't read image stack from '" << imgdirpath << "'." << std::endl;
		return;
	}
	std::cout << "Array size: (nx,ny,nz)=(" << nx << "," << ny << "," << nz << ")" << std::endl;

	float* arr_x = new float[nx*ny*nz];
	std::memcpy(&arr_x[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortingNetworkCuda(arr_x, 'x', nx, ny, nz)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along x-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_x.c_str(), arr_x, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_x << "'." << std::endl; }
	//return;
	float* arr_y = new float[nx*ny*nz];
	std::memcpy(&arr_y[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortingNetworkCuda(arr_y, 'y', nx, ny, nz)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along y-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_y.c_str(), arr_y, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_y << "'." << std::endl; }

	float* arr_z = new float[nx*ny*nz];
	std::memcpy(&arr_z[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortingNetworkCuda(arr_z, 'z', nx, ny, nz)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along z-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_z.c_str(), arr_z, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_z << "'." << std::endl; }

	delete[] arr0, arr_x, arr_y, arr_z;
	std::cout << std::endl;
}


void example_blocksize()
{
	std::cout << "CUDA Optimization Example - Block Size" << std::endl;
	std::string imgdirpath = RESOURCE_MAINDIR + std::string("\\sort\\before");
	std::string savepath_1 = RESOURCE_MAINDIR + std::string("\\sort\\after_1");
	std::string savepath_2 = RESOURCE_MAINDIR + std::string("\\sort\\after_2");
	std::string savepath_3 = RESOURCE_MAINDIR + std::string("\\sort\\after_3");
	std::chrono::steady_clock::time_point time_start, time_end;
	float duration = 0;
	int algo = 0;
	int const num_byte = sizeof(float);
	float* arr0 = nullptr;
	int nx, ny, nz;
	if (!readStack(imgdirpath.c_str(), arr0, nx, ny, nz))
	{
		std::cout << "Can't read image stack from '" << imgdirpath << "'." << std::endl;
		return;
	}
	std::cout << "Array size: (nx,ny,nz)=(" << nx << "," << ny << "," << nz << ")" << std::endl;

	int block_size1 = 512;
	float* arr_1 = new float[nx*ny*nz];
	std::memcpy(&arr_1[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortCuda(arr_1, 'y', nx, ny, nz, algo, block_size1)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting using block size=" << block_size1 << " in " << duration << "s." << std::endl;
	if (!saveStack(savepath_1.c_str(), arr_1, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_1 << "'." << std::endl; }

	int block_size2 = 256;
	float* arr_2 = new float[nx*ny*nz];
	std::memcpy(&arr_2[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortCuda(arr_2, 'y', nx, ny, nz, algo, block_size2)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting using block size=" << block_size2 << " in " << duration << "s." << std::endl;
	if (!saveStack(savepath_2.c_str(), arr_2, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_2 << "'." << std::endl; }

	int block_size3 = 128;
	float* arr_3 = new float[nx*ny*nz];
	std::memcpy(&arr_3[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortCuda(arr_3, 'y', nx, ny, nz, algo, block_size3)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting using block size=" << block_size3 << " in " << duration << "s." << std::endl;
	if (!saveStack(savepath_3.c_str(), arr_3, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_3 << "'." << std::endl; }

	delete[] arr0, arr_1, arr_2, arr_3;
	std::cout << std::endl;
}

void example_thread_diverge()
{
	std::cout << "CUDA Optimization Example - Thread Divergence" << std::endl;
	std::string imgdirpath = RESOURCE_MAINDIR + std::string("\\sort\\before");
	std::string savepath_1 = RESOURCE_MAINDIR + std::string("\\sort\\after_1");
	std::string savepath_2 = RESOURCE_MAINDIR + std::string("\\sort\\after_2");
	std::string savepath_3 = RESOURCE_MAINDIR + std::string("\\sort\\after_3");
	std::string savepath_4 = RESOURCE_MAINDIR + std::string("\\sort\\after_4");
	std::string savepath_5 = RESOURCE_MAINDIR + std::string("\\sort\\after_5");
	std::chrono::steady_clock::time_point time_start, time_end;
	float duration = 0;
	int algo = 0;
	int block_size = 512;
	int const num_byte = sizeof(float);
	float* arr0 = nullptr;
	int nx, ny, nz;
	if (!readStack(imgdirpath.c_str(), arr0, nx, ny, nz))
	{
		std::cout << "Can't read image stack from '" << imgdirpath << "'." << std::endl;
		return;
	}
	std::cout << "Array size: (nx,ny,nz)=(" << nx << "," << ny << "," << nz << ")" << std::endl;

	float* arr_1 = new float[nx*ny*nz];
	algo = 0;
	std::memcpy(&arr_1[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sort(arr_1, 'y', nx, ny, nz, algo)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "(CPU) Complete sorting using selection sort in " << duration << "s." << std::endl;
	std::memcpy(&arr_1[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortCuda(arr_1, 'y', nx, ny, nz, algo, block_size)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "(GPU) Complete sorting using selection sort in " << duration << "s." << std::endl;
	if (!saveStack(savepath_1.c_str(), arr_1, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_1 << "'." << std::endl; }

	float* arr_2 = new float[nx*ny*nz];
	algo = 1;
	std::memcpy(&arr_2[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sort(arr_2, 'y', nx, ny, nz, algo)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "(CPU) Complete sorting using insertion sort in " << duration << "s." << std::endl;
	std::memcpy(&arr_2[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortCuda(arr_2, 'y', nx, ny, nz, algo, block_size)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "(GPU) Complete sorting using insertion sort in " << duration << "s." << std::endl;
	if (!saveStack(savepath_2.c_str(), arr_2, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_2 << "'." << std::endl; }

	float* arr_3 = new float[nx*ny*nz];
	algo = 2;
	std::memcpy(&arr_3[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sort(arr_3, 'y', nx, ny, nz, algo)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "(CPU) Complete sorting using heap sort in " << duration << "s." << std::endl;
	std::memcpy(&arr_3[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortCuda(arr_3, 'y', nx, ny, nz, algo, block_size)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "(GPU) Complete sorting using heap sort in " << duration << "s." << std::endl;
	if (!saveStack(savepath_3.c_str(), arr_3, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_3 << "'." << std::endl; }

	float* arr_4 = new float[nx*ny*nz];
	algo = 3;
	std::memcpy(&arr_4[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sort(arr_4, 'y', nx, ny, nz, algo)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "(CPU) Complete sorting using quick sort in " << duration << "s." << std::endl;
	std::memcpy(&arr_4[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortCuda(arr_4, 'y', nx, ny, nz, algo, block_size)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "(GPU) Complete sorting using quick sort in " << duration << "s." << std::endl;
	if (!saveStack(savepath_4.c_str(), arr_4, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_4 << "'." << std::endl; }

	float* arr_5 = new float[nx*ny*nz];
	algo = 4; 
	std::memcpy(&arr_5[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sort(arr_5, 'y', nx, ny, nz, algo)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "(CPU) Complete sorting using bitonic sort in " << duration << "s." << std::endl;
	std::memcpy(&arr_5[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortCuda(arr_5, 'y', nx, ny, nz, algo, block_size)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "(GPU) Complete sorting using bitonic sort in " << duration << "s." << std::endl;
	if (!saveStack(savepath_5.c_str(), arr_5, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_5 << "'." << std::endl; }

	delete[] arr0, arr_1, arr_2, arr_3, arr_4, arr_5;
	std::cout << std::endl;
}

void example_memory_coalesce()
{
	std::cout << "CUDA Optimization Example - Memory Coalescing" << std::endl;
	std::string imgdirpath = RESOURCE_MAINDIR + std::string("\\sort\\before");
	std::string savepath_x = RESOURCE_MAINDIR + std::string("\\sort\\after_x");
	std::string savepath_y = RESOURCE_MAINDIR + std::string("\\sort\\after_y");
	std::string savepath_z = RESOURCE_MAINDIR + std::string("\\sort\\after_z");
	std::chrono::steady_clock::time_point time_start, time_end;
	float duration = 0;
	int algo = 4;
	int block_size = 512;
	int const num_byte = sizeof(float);
	float* arr0 = nullptr;
	int nx, ny, nz;
	if (!readStack(imgdirpath.c_str(), arr0, nx, ny, nz))
	{
		std::cout << "Can't read image stack from '" << imgdirpath << "'." << std::endl;
		return;
	}
	std::cout << "Array size: (nx,ny,nz)=(" << nx << "," << ny << "," << nz << ")" << std::endl;

	float* arr_x = new float[nx*ny*nz];
	std::memcpy(&arr_x[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortCuda(arr_x, 'x', nx, ny, nz, algo, block_size)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along x-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_x.c_str(), arr_x, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_x << "'." << std::endl; }

	float* arr_y = new float[nx*ny*nz];
	std::memcpy(&arr_y[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortCuda(arr_y, 'y', nx, ny, nz, algo, block_size)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along y-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_y.c_str(), arr_y, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_y << "'." << std::endl; }

	float* arr_z = new float[nx*ny*nz];
	std::memcpy(&arr_z[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortCuda(arr_z, 'z', nx, ny, nz, algo, block_size)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along z-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_z.c_str(), arr_z, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_z << "'." << std::endl; }

	delete[] arr0, arr_x, arr_y, arr_z;
	std::cout << std::endl;
}

void example_shared_memory()
{
	std::cout << "CUDA Optimization Example - Shared Memory" << std::endl;
	std::string imgdirpath = RESOURCE_MAINDIR + std::string("\\sort\\before");
	std::string savepath_x = RESOURCE_MAINDIR + std::string("\\sort\\after_x1");
	std::string savepath_y = RESOURCE_MAINDIR + std::string("\\sort\\after_y1");
	std::string savepath_z = RESOURCE_MAINDIR + std::string("\\sort\\after_z1");
	std::chrono::steady_clock::time_point time_start, time_end;
	float duration = 0;
	int const num_byte = sizeof(float);
	float* arr0 = nullptr;
	int nx, ny, nz;
	if (!readStack(imgdirpath.c_str(), arr0, nx, ny, nz))
	{
		std::cout << "Can't read image stack from '" << imgdirpath << "'." << std::endl;
		return;
	}
	std::cout << "Array size: (nx,ny,nz)=(" << nx << "," << ny << "," << nz << ")" << std::endl;

	float* arr_x = new float[nx*ny*nz];
	std::memcpy(&arr_x[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortingNetworkCuda(arr_x, 'x', nx, ny, nz)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along x-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_x.c_str(), arr_x, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_x << "'." << std::endl; }
	//return;
	float* arr_y = new float[nx*ny*nz];
	std::memcpy(&arr_y[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortingNetworkCuda(arr_y, 'y', nx, ny, nz)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along y-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_y.c_str(), arr_y, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_y << "'." << std::endl; }

	float* arr_z = new float[nx*ny*nz];
	std::memcpy(&arr_z[0], &arr0[0], num_byte*nx*ny*nz);
	time_start = std::chrono::steady_clock::now();
	if (!sortingNetworkCuda(arr_z, 'z', nx, ny, nz)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting along z-direction in " << duration << "s" << std::endl;
	if (!saveStack(savepath_z.c_str(), arr_z, nx, ny, nz)) { std::cout << "Can't save image stack to '" << savepath_z << "'." << std::endl; }

	delete[] arr0, arr_x, arr_y, arr_z;
	std::cout << std::endl;
}

void example_cudastream()
{
	std::cout << "CUDA Optimization Example - Stream" << std::endl;
	std::string imgdirpath = RESOURCE_MAINDIR + std::string("\\cuda_stream\\before.tiff");
	std::string savepath_1 = RESOURCE_MAINDIR + std::string("\\cuda_stream\\after(cpu).tiff");
	std::string savepath_2 = RESOURCE_MAINDIR + std::string("\\cuda_stream\\after(gpu_nostream).tiff");
	std::string savepath_3 = RESOURCE_MAINDIR + std::string("\\cuda_stream\\after(gpu_stream).tiff");
	std::chrono::steady_clock::time_point time_start, time_end;
	float duration = 0;
	int const num_byte = sizeof(float);
	float* arr0 = nullptr;
	int nx, ny;
	if (!readImage(imgdirpath.c_str(), arr0, nx, ny))
	{
		std::cout << "Can't read image stack from '" << imgdirpath << "'." << std::endl;
		return;
	}
	std::cout << "Array size: (nx,ny)=(" << nx << "," << ny << ")" << std::endl;

	float* arr_1 = new float[nx*ny];
	std::memcpy(&arr_1[0], &arr0[0], num_byte*nx*ny);
	time_start = std::chrono::steady_clock::now();
	int algo = 3;
	if (!sort(arr_1, 'x', nx, ny, 1, algo)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting using CPU (quick sort) in " << duration << "s." << std::endl;
	if (!saveImage(savepath_1.c_str(), arr_1, nx, ny, false)) { std::cout << "Can't save image stack to '" << savepath_1 << "'." << std::endl; }

	float* arr_2 = new float[nx*ny];
	std::memcpy(&arr_2[0], &arr0[0], num_byte*nx*ny);
	time_start = std::chrono::steady_clock::now();
	if (!sortingNetworkCuda(arr_2, 'x', nx, ny, 1)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting using GPU (bitonic sort) without stream in " << duration << "s." << std::endl;
	if (!saveImage(savepath_2.c_str(), arr_2, nx, ny, false)) { std::cout << "Can't save image stack to '" << savepath_2 << "'." << std::endl; }

	float* arr_3 = new float[nx*ny];
	std::memcpy(&arr_3[0], &arr0[0], num_byte*nx*ny);
	time_start = std::chrono::steady_clock::now();
	if (!sort2DWithStream(arr_3, nx, ny)) { return; }
	time_end = std::chrono::steady_clock::now();
	duration = (float)((time_end - time_start).count()) / 1000000000.0;
	std::cout << "Complete sorting using GPU (bitonic sort) with stream in " << duration << "s." << std::endl;
	if (!saveImage(savepath_3.c_str(), arr_3, nx, ny, false)) { std::cout << "Can't save image stack to '" << savepath_3 << "'." << std::endl; }

	delete[] arr0, arr_1, arr_2, arr_3;
	std::cout << std::endl;
}


int main()
{
	//checkIndex1Dgrid_1Dblock();
	//checkIndex2Dgrid_2Dblock();

	example_arrayAdd();
	example_flip();
	example_transpose();
	example_convolution2d();

	//example_sort_cpu();
	//example_sort_gpu();
	//example_sorting_network();

	//example_blocksize();
	//example_thread_diverge();
	//example_memory_coalesce();
	//example_shared_memory();
	//example_cudastream();

	return 0;
}

