#include <iostream>
#include <vector>
#include <experimental/filesystem>
#include "opencv2/opencv.hpp"

namespace fs = std::experimental::filesystem;

inline bool readImage(const char* image_filepath, float* &dst, int& nx, int& ny)
{
	bool imgfile_exist = fs::exists(image_filepath);
	if (imgfile_exist == false) { return false; }
	cv::Mat cv_arr = cv::imread(image_filepath, cv::IMREAD_ANYDEPTH);
	if (cv_arr.empty()) { return false; }
	cv::Mat cv_arr_f32;
	int mat_dtype = CV_32F;
	cv_arr.convertTo(cv_arr_f32, mat_dtype);
	nx = cv_arr_f32.size().width;
	ny = cv_arr_f32.size().height;
	int n_pixel = nx * ny;
	if (n_pixel == 0) { return false; }
	if (dst != nullptr) { delete[] dst; }
	dst = new float[n_pixel];
	int num_byte = sizeof(float);
	int num_byte_per_row = num_byte * nx;
	for (int j = 0; j < ny; j++)
	{
		int loc = j * nx;
		std::memcpy(&dst[loc], cv_arr_f32.ptr(j), num_byte_per_row);
	}
	//for (int j = 0; j < nx*ny; j++) { dst[j] /= 255.0; }
	return true;
}

inline bool saveImage(const char* image_filepath, float* src, int nx, int ny)
{
	int num_byte = sizeof(float);
	int num_byte_per_row = num_byte * nx;
	cv::Mat cv_arr(ny, nx, CV_32F, src);
	cv::Mat cv_arr_8u;
	cv_arr.convertTo(cv_arr_8u, CV_8U);
	bool save_success = cv::imwrite(image_filepath, cv_arr_8u);
	return save_success;
}

inline bool readStack(const char* image_dirpath, float* &dst, int& nx, int& ny, int& nz)
{
	bool dir_exist = fs::exists(image_dirpath);
	std::string img_prefix = "slice_";
	int width = 0;
	int height = 0;
	int n_slice = 0;
	int i = 0;
	std::vector<cv::Mat> imgstack;
	for (auto& dir_item : fs::directory_iterator(image_dirpath))
	{
		//std::cout<<i << ", " << n_slice<<std::endl;
		std::string imgpath = std::string(image_dirpath) + "\\" + img_prefix + std::to_string(i) + std::string(".") + std::string("tiff");
		bool imgfile_exist = fs::exists(imgpath);
		if (imgfile_exist)
		{
			cv::Mat img_slice = cv::imread(imgpath.c_str(), cv::IMREAD_ANYDEPTH);
			if (img_slice.empty()) { continue; }
			int slice_width = img_slice.size().width;
			int slice_height = img_slice.size().height;
			if (n_slice == 0)
			{
				width = slice_width;
				height = slice_height;
			}
			else
			{
				if ((width != slice_width) || (height != slice_height))
				{
					return false;
				}
			}
			imgstack.push_back(img_slice);
			n_slice++;
		}
		i++;
	}
	if (imgstack.size() == 0)
	{
		std::cout << "No supported filenames/filetypes " << std::endl;
		return false;
	}
	nx = width;
	ny = height;
	nz = n_slice;
	int n_voxel = nx * ny * nz;
	if (dst != nullptr) { delete[] dst; }
	dst = new float[n_voxel];
	int num_byte = sizeof(float);
	int num_byte_per_row = num_byte * nx;
	for (int i = 0; i < nz; i++)
	{
		cv::Mat cv_slice;
		imgstack[i].convertTo(cv_slice, CV_32F);
		for (int j = 0; j < ny; j++)
		{
			int loc = i * nx*ny + j * nx;
			std::memcpy(&dst[loc], cv_slice.ptr(j), num_byte_per_row);
		}
	}
	return true;
}

inline bool saveStack(const char* image_dirpath, float* src, int nx, int ny, int nz)
{
	bool dir_exist = fs::exists(image_dirpath);
	if (dir_exist == false) { return false; }
	int area = nx * ny;
	int num_byte = sizeof(float);
	int byte_per_row = num_byte * nx;
	std::vector<cv::Mat> cv_stack;
	for (int i = 0; i < nz; i++)
	{
		float* tmp;
		tmp = &src[i * area];
		cv::Mat cv_slice(ny, nx, CV_32F, tmp);
		//dst = cv::Mat(srcHeight, srcWidth, CV_32FC1, const_cast<T*>(src), srcPitch).clone();
		cv_stack.push_back(cv_slice);
	}
	if (cv_stack.size() < 1) { return false; }
	if (dir_exist == false) { return false; }
	std::string img_prefix = "slice_";
	bool save_success = false;
	for (int i = 0; i < cv_stack.size(); ++i)
	{
		std::string imgpath = std::string(image_dirpath) + "\\" + img_prefix + std::to_string(i) + std::string(".") + std::string("tiff");
		cv::Mat img_slice = cv_stack[i];
		save_success = cv::imwrite(imgpath.c_str(), img_slice);
		if (save_success == false) { return false; }
	}
	return true;
}



