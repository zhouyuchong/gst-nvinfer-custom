/*
 * @Author: zhouyuchong
 * @Date: 2024-03-01 13:09:36
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2024-03-04 15:13:16
 */
#include <npp.h>
#include <opencv2/opencv.hpp>

void nppiAffineTransformExample(NVBufferSurface *surface, const cv::Mat& dst, cv::Mat& transformed) {
    // 获取NVBufferSurface的尺寸
    NVBufferCreateParams params;
    NVBufferGetParams(surface, &params);
    int srcRows = params.height;
    int srcCols = params.width;

    // 创建NPP缓冲区
    Npp8u* npSrc = nppiMalloc_8u_C1R(srcRows, srcCols);

    // 将NVBufferSurface的数据复制到NPP缓冲区
    NVBufferSurface *npDstSurface = nppiNewBufferSurface(srcRows, srcCols, NPP_8U, NPP_C1);
    NppiCopy_8u_C1R(surface->surfaceList[0].ptr, srcRows, npSrc, srcCols);

    // 创建变换矩阵
    cv::Mat M = cv::getAffineTransform(cv::Mat::zeros(2, 3, CV_32F), cv::Mat::zeros(2, 1, CV_32F));

    // 将变换矩阵转换为NPP格式
    Npp32f affineMatrix[3];
    for (int i = 0; i < 3; i++) {
        affineMatrix[i] = (Npp32f)M.at<float>(i, 0);
    }

    // 执行仿射变换
    nppiAffineTransform_8u_C1R(npSrc, srcRows, srcCols, affineMatrix, npDstSurface->surfaceList[0].ptr, srcRows, srcCols, NPPI_INTER_LINEAR);

    // 将NPP缓冲区数据复制回OpenCV Mat
    transformed.create(srcRows, srcCols, CV_8UC1);
    NppiCopy_8u_C1R(npDstSurface->surfaceList[0].ptr, srcRows, transformed.data, transformed.step);

    // 释放NPP缓冲区
    nppiFree(npSrc);
    nppiDeleteBufferSurface(npDstSurface);
}

int main() {
    // 加载图像
    cv::Mat src = cv::imread("path/to/source/image.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat dst = cv::imread("path/to/destination/image.jpg", cv::IMREAD_GRAYSCALE);

    // 执行仿射变换
    cv::Mat transformed;
    nppiAffineTransformExample(surface, dst, transformed);

    // 显示结果
    cv::imshow("Transformed Image", transformed);
    cv::waitKey(0);

    return 0;
}


#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>
#include <opencv2/opencv.hpp>

#define CUDA_FREE(ptr) { if (ptr != nullptr) { cudaFree(ptr); ptr = nullptr; } }

int main() {
  std::string directory = "../";
  cv::Mat image_dog = cv::imread(directory + "dog.png");
  int image_width = image_dog.cols;
  int image_height = image_dog.rows;
  int image_size = image_width * image_height;

  // =============== device memory ===============
  // input
  uint8_t *in_image;
  cudaMalloc((void**)&in_image, image_size * 3 * sizeof(uint8_t));
  cudaMemcpy(in_image, image_dog.data, image_size * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // output
  uint8_t *out_ptr1, *out_ptr2;
  cudaMalloc((void**)&out_ptr1, image_size * 3 * sizeof(uint8_t));  // 三通道
  cudaMalloc((void**)&out_ptr2, image_size * 3 * sizeof(uint8_t));  // 三通道

  double angle = 30.0;
  double scale = 0.6;
  cv::Point center = cv::Point(image_width / 2, image_height / 2);
  cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
  double coeffs[2][3] = { rot_mat.at<double>(0, 0),
                          rot_mat.at<double>(0, 1),
                          rot_mat.at<double>(0, 2),
                          rot_mat.at<double>(1, 0),
                          rot_mat.at<double>(1, 1),
                          rot_mat.at<double>(1, 2)};

  NppiSize in_size;
  in_size.width = image_width;
  in_size.height = image_height;
  NppiRect rc;
  rc.x = 0;
  rc.y = 0;
  rc.width = image_width;
  rc.height = image_height;

  cv::Mat out_image = cv::Mat::zeros(image_height, image_width, CV_8UC3);
  NppStatus status;
  // =============== nppiWarpAffine_8u_C3R ===============
  status = nppiWarpAffine_8u_C3R(in_image, in_size, image_width * 3, rc, out_ptr1, image_width * 3, 
                                 rc, coeffs, NPPI_INTER_LINEAR);
  if (status != NPP_SUCCESS) {
    std::cout << "[GPU] ERROR nppiWarpAffine_8u_C3R failed, status = " << status << std::endl;
    return false;
  }
  cudaMemcpy(out_image.data, out_ptr1, image_size * 3, cudaMemcpyDeviceToHost);
  cv::imwrite(directory + "affine.jpg", out_image);

   // =============== nppiWarpAffineBack_8u_C3R ===============
  status = nppiWarpAffineBack_8u_C3R(out_ptr1, in_size, image_width * 3, rc, out_ptr2, image_width * 3, 
                                     rc, coeffs, NPPI_INTER_LINEAR);
  if (status != NPP_SUCCESS) {
    std::cout << "[GPU] ERROR nppiWarpAffineBack_8u_C3R failed, status = " << status << std::endl;
    return false;
  }
  cudaMemcpy(out_image.data, out_ptr2, image_size * 3, cudaMemcpyDeviceToHost);
  cv::imwrite(directory + "affine_back.jpg", out_image);

  // free
  CUDA_FREE(in_image)
  CUDA_FREE(out_ptr1)
  CUDA_FREE(out_ptr2)
}
