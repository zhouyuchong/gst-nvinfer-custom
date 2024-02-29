/ sample compile command line: nvcc -o rs rs.cu -lnppicc -lnppig -DUSE_DEBUG -DUNIT_TEST
#include <nppi.h>
#include <iostream>

template <typename T>
__global__ void pack_uv(T * __restrict__ u, T * __restrict__ v, T * __restrict__ uv, const int w, const int h, const int pitch_uv, const int pitch_u, const int pitch_v){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  int idy = threadIdx.y+blockDim.y*blockIdx.y;
  if ((idx < w) && (idy < h)){
    T *o  = (T *)(((char *)uv) + idy*pitch_uv);
    T *iu = (T *)(((char *)u)  + idy*pitch_u);
    T *iv = (T *)(((char *)v)  + idy*pitch_v);
    int idx2 = idx >> 1;
    o[idx] = (idx&1)?iv[idx2]:iu[idx2];}
}

int rs(const int ish, const int isw, const int ipitch, const int osh, const int osw, const int opitch, const unsigned char *iy, const unsigned char *iuv, unsigned char *oy, unsigned char *ouv, unsigned char *tempbuff, int method = 0,  int eInterpolation = NPPI_INTER_LANCZOS){

#ifdef USE_DEBUG
  if ((iy != NULL) && (tempbuff == NULL)) std::cout << "error: tempbuff is NULL" << std::endl;
  if ((iy != NULL) && (iuv == NULL)) std::cout << "error: iuv is NULL" << std::endl;
  if ((iy != NULL) && (oy == NULL)) std::cout << "error: oy is NULL" << std::endl;
  if ((iy != NULL) && (ouv == NULL)) std::cout << "error: ouv is NULL" << std::endl;
  if (isw < 2) std::cout << "error on input width: " << isw << std::endl;
  if (ish < 2) std::cout << "error on input height: " << ish << std::endl;
  if (ipitch < isw) std::cout << "error on input pitch: " << ipitch << std::endl;
  if (osw < 1) std::cout << "error on output width: " << osw << std::endl;
  if (osh < 1) std::cout << "error on output height: " << osh << std::endl;
  if (opitch < osw) std::cout << "error on output pitch: " << opitch << std::endl;
#endif
  cudaError_t err;
  NppStatus stat;

// convert NV12 input to RGB

  if (iy == NULL){ // temp buffer sizing
     // for method 1
     NppiSize oSrcROI;
     oSrcROI.width  = isw;
     oSrcROI.height = ish;
     NppiSize oDstROI;
     oDstROI.width  = osw;
     oDstROI.height = osh;
     int bufferSize;
     stat = nppiResizeAdvancedGetBufferHostSize_8u_C1R(oSrcROI, oDstROI, &bufferSize, NPPI_INTER_LANCZOS3_ADVANCED);
     return ((ish*isw + osh*osw)*3*sizeof(unsigned char))+bufferSize;  // temp buffer sizing
     }
  if (method == 0){

    const Npp8u *pSrc[2] = {iy, iuv};
    NppiSize oSizeROI;
    oSizeROI.width  = isw;
    oSizeROI.height = ish;
#ifdef USE_709
    stat = nppiNV12ToRGB_709HDTV_8u_P2C3R(pSrc, ipitch, tempbuff, isw*3*sizeof(Npp8u), oSizeROI);
#else
    stat = nppiNV12ToRGB_8u_P2C3R(pSrc, ipitch, tempbuff, isw*3*sizeof(Npp8u), oSizeROI);
#endif
#ifdef USE_DEBUG
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cout << "NV12 to RGB CUDA error: " << cudaGetErrorString(err) << std::endl;
    if (stat != NPP_SUCCESS) std::cout << "NV12 to RGB NPP error: " << (int)stat << std::endl;
#endif
    if (stat != NPP_SUCCESS) return -1;

// perform resize

    NppiSize oSrcSize;
    oSrcSize.width = isw;
    oSrcSize.height = ish;
    NppiRect oSrcROI;
    oSrcROI.x = 0;
    oSrcROI.y = 0;
    oSrcROI.width = isw;
    oSrcROI.height = ish;
    NppiRect oDstROI;
    oDstROI.x = 0;
    oDstROI.y = 0;
    oDstROI.width = osw;
    oDstROI.height = osh;
    double nXFactor = osw/(double)isw;
    double nYFactor = osh/(double)ish;
    double nXShift = 0;
    double nYShift = 0;
    stat = nppiResizeSqrPixel_8u_C3R(tempbuff, oSrcSize, isw*3*sizeof(Npp8u), oSrcROI, tempbuff+ish*isw*3, osw*3*sizeof(Npp8u), oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation);
#ifdef USE_DEBUG
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cout << "RGB LANCZOS RESIZE CUDA error: " << cudaGetErrorString(err) << std::endl;
    if (stat != NPP_SUCCESS) std::cout << "RGB LANCZOS RESIZE NPP error: " << (int)stat << std::endl;
#endif
    if (stat != NPP_SUCCESS) return -2;

// convert resized RGB to YUV420


    Npp8u *pDst[3] = { oy, ouv, ouv + osh*opitch/4 };
    int rDstStep[3] = { opitch, opitch/2, opitch/2 };
    oSizeROI.width = osw;
    oSizeROI.height = osh;
    stat = nppiRGBToYUV420_8u_C3P3R(tempbuff+ish*isw*3, osw*3*sizeof(Npp8u), pDst, rDstStep, oSizeROI);
#ifdef USE_DEBUG
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cout << "RGB TO YUV420 CUDA error: " << cudaGetErrorString(err) << std::endl;
    if (stat != NPP_SUCCESS) std::cout << "RGB TO YUV420 NPP error: " << (int)stat << std::endl;
#endif
    if (stat != NPP_SUCCESS) return -3;

// pack uv

    dim3 block(32, 8);
    dim3 grid((osw+block.x-1)/block.x, (osh+block.y-1)/block.y);
    pack_uv<<< grid, block >>>(ouv, ouv + osh*opitch/4, tempbuff, osw, osh/2, osw, osw/2, osw/2);
    err = cudaGetLastError();
#ifdef USE_DEBUG
    if (err != cudaSuccess) std::cout << "PACK UV LAUNCH CUDA error: " << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cout << "PACK UV EXEC CUDA error: " << cudaGetErrorString(err) << std::endl;
#endif
    if (err != cudaSuccess) return -4;

// move packed uv to output

    err = cudaMemcpy2D(ouv, opitch,  tempbuff, osw*sizeof(Npp8u), osw*sizeof(Npp8u), osh/2, cudaMemcpyDeviceToDevice);
#ifdef USE_DEBUG
    if (err != cudaSuccess) std::cout << "PACK UV COPY CUDA error: " << cudaGetErrorString(err) << std::endl;
#endif
    if (err != cudaSuccess) return -5;
    }
  else{  // method 1

// NV12 to YUV420 planar
    const Npp8u *const pSrc[2] = {iy, iuv};
    Npp8u *pDst[3] = {tempbuff, tempbuff+isw*ish, tempbuff+isw*ish+(isw*ish)/4};
    int aDstStep[3] = {isw, isw/2, isw/2};
    NppiSize oSizeROI;
    oSizeROI.width  = isw;
    oSizeROI.height = ish;
    stat = nppiNV12ToYUV420_8u_P2P3R(pSrc, ipitch, pDst, aDstStep, oSizeROI);
#ifdef USE_DEBUG
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cout << "NV12 TO YUV420 CUDA error: " << cudaGetErrorString(err) << std::endl;
    if (stat != NPP_SUCCESS) std::cout << "NV12 TO YUV420 NPP error: " << (int)stat << std::endl;
#endif
    if (stat != NPP_SUCCESS) return -6;
// resize each plane individually
    NppiSize oSrcSize = oSizeROI;
    NppiRect oSrcROI;
    oSrcROI.x = 0;
    oSrcROI.y = 0;
    oSrcROI.width = isw;
    oSrcROI.height = ish;
    NppiRect oDstROI;
    oDstROI.x = 0;
    oDstROI.y = 0;
    oDstROI.width = osw;
    oDstROI.height = osh;
    double nXFactor = osw/(double)isw;
    double nYFactor = osh/(double)ish;
// resize Y
    stat = nppiResizeSqrPixel_8u_C1R_Advanced(tempbuff, oSrcSize, isw, oSrcROI, oy, opitch, oDstROI, nXFactor, nYFactor, tempbuff+(ish*isw*3),NPPI_INTER_LANCZOS3_ADVANCED);
#ifdef USE_DEBUG
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cout << "Y RESIZE CUDA error: " << cudaGetErrorString(err) << std::endl;
    if (stat != NPP_SUCCESS) std::cout << "Y RESIZE NPP error: " << (int)stat << std::endl;
#endif
    if (stat != NPP_SUCCESS) return -7;
// resize U
    oSrcSize.width  /= 2;
    oSrcSize.height /= 2;
    oSrcROI.width   /= 2;
    oSrcROI.height  /= 2;
    oDstROI.width   /= 2;
    oDstROI.height  /= 2;
    stat = nppiResizeSqrPixel_8u_C1R_Advanced(tempbuff+ish*isw, oSrcSize, isw/2, oSrcROI, tempbuff+(ish*isw*3), osw/2, oDstROI, nXFactor, nYFactor, tempbuff+(ish*isw*3) + (osh*osw*3),NPPI_INTER_LANCZOS3_ADVANCED);
#ifdef USE_DEBUG
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cout << "U RESIZE CUDA error: " << cudaGetErrorString(err) << std::endl;
    if (stat != NPP_SUCCESS) std::cout << "U RESIZE NPP error: " << (int)stat << std::endl;
#endif
    if (stat != NPP_SUCCESS) return -8;
// resize V
    stat = nppiResizeSqrPixel_8u_C1R_Advanced(tempbuff+ish*isw+(ish*isw/4), oSrcSize, isw/2, oSrcROI, tempbuff+(ish*isw*3)+(osh*osw/4), osw/2, oDstROI, nXFactor, nYFactor, tempbuff+(ish*isw*3) + (osh*osw*3),NPPI_INTER_LANCZOS3_ADVANCED);
#ifdef USE_DEBUG
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cout << "V RESIZE CUDA error: " << cudaGetErrorString(err) << std::endl;
    if (stat != NPP_SUCCESS) std::cout << "V RESIZE NPP error: " << (int)stat << std::endl;
#endif
    if (stat != NPP_SUCCESS) return -9;

// pack_uv
    dim3 block(32, 8);
    dim3 grid((osw+block.x-1)/block.x, (osh+block.y-1)/block.y);
    pack_uv<<< grid, block >>>(tempbuff+(ish*isw*3), tempbuff+(ish*isw*3)+(osh*osw/4), ouv, osw, osh/2, opitch, osw/2, osw/2);
    err = cudaGetLastError();
#ifdef USE_DEBUG
    if (err != cudaSuccess) std::cout << "PACK UV LAUNCH CUDA error: " << cudaGetErrorString(err) << std::endl;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cout << "PACK UV EXEC CUDA error: " << cudaGetErrorString(err) << std::endl;
#endif
    if (err != cudaSuccess) return -10;
    }

  return 0;
}

#ifdef UNIT_TEST
// timing
#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL
unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

// bitmap file handling
struct Info{
    int width;
    int height;
    int offset;
    unsigned char * info;
    unsigned char * data;

    int size;
};
#include <fstream>
Info readBMP(const char* filename)
{
    int i;
    std::ifstream is(filename, std::ifstream::binary);
    is.seekg(0, is.end);
    i = is.tellg();
    is.seekg(0);
    unsigned char *info = new unsigned char[i];
    is.read((char *)info,i);

    int width = *(int*)&info[18];
    int height = *(int*)&info[22];
    int offset = *(int*)&info[10];
    Info dat;
    dat.width = width;
    dat.height = height;
    dat.offset = offset;
    dat.size = i;
    dat.info = new unsigned char[offset - 1];
    dat.data = new unsigned char[i - offset + 1];
    if ((i-offset+1) < (3*height*width)) std::cout << "size: " << i-offset+1 << " expected: " << height*width*3 << std::endl;
    std::copy(info,
              info + offset,
              dat.info);

    std::copy(info + offset,
              info + i,
              dat.data);
    delete[] info;
    return dat;

}
void writeBMP(const char *filename, Info dat){

    std::ofstream fout;
    fout.open(filename, std::ios::binary | std::ios::out);
    fout.write( reinterpret_cast<char *>(dat.info), dat.offset);

    fout.write( reinterpret_cast<char *>(dat.data), dat.size - dat.offset );
    fout.close();
}

int main(int argc, char *argv[]){
  int eInterpolation = NPPI_INTER_LANCZOS;
  if (argc > 1) eInterpolation = atoi(argv[1]);
  else{
    std::cout << "Must specify a valid interpolation mode:" << std::endl;
    std::cout << NPPI_INTER_NN << " :NPPI_INTER_NN" << std::endl;
    std::cout << NPPI_INTER_LINEAR << " :NPPI_INTER_LINEAR" << std::endl;
    std::cout << NPPI_INTER_CUBIC << " :NPPI_INTER_CUBIC" << std::endl;
    std::cout << NPPI_INTER_LANCZOS << " :NPPI_INTER_LANCZOS" << std::endl;
    return 0;}
  int method = 0;
  if (argc > 2) method = atoi(argv[2]);
  // input to NV12
  Info rfile = readBMP("input.bmp");
  const int H = rfile.height;
  const int W = rfile.width;
  std::cout << "Height = " << rfile.height << std::endl;
  std::cout << "Width  = " << rfile.width  << std::endl;
  Npp8u *rgbdata, *ty, *tu, *tv, *tuv;
  cudaMalloc(&rgbdata, H*W*3);
  cudaMalloc(&ty, H*W);
  cudaMalloc(&tu, H*W/4);
  cudaMalloc(&tv, H*W/4);
  cudaMalloc(&tuv, H*W/2);

  cudaMemcpy(rgbdata, rfile.data, H*W*3, cudaMemcpyHostToDevice);
  Npp8u *pDst[3] = { ty, tu, tv};
  int rDstStep[3] = { W, W/2, W/2 };
  NppiSize oSizeROI;
  oSizeROI.width = W;
  oSizeROI.height = H;
  NppStatus stat = nppiRGBToYUV420_8u_C3P3R(rgbdata, W*3*sizeof(Npp8u), pDst, rDstStep, oSizeROI);
  if (stat != NPP_SUCCESS) { std::cout << "Input NPP error"  << std::endl; return 0;}
  dim3 block(32, 8);
  dim3 grid((W+block.x-1)/block.x, (H+block.y-1)/block.y);
  pack_uv<<< grid, block >>>(tu, tv, tuv, W, H/2, W, W/2, W/2);

  // 1:1 test

  int buff_size = rs(H, W, W, H, W, W, NULL, NULL, NULL, NULL, NULL);
  unsigned char *tbuff;
  cudaError_t err = cudaMalloc(&tbuff, buff_size);
  if (err != cudaSuccess) {std::cout << "on temp buff allocation of size: " << buff_size << " error: " << (int)err << std::endl; return 0;}
  unsigned char *oy, *ouv;
  err = cudaMalloc(&oy, H*W*sizeof(unsigned char));
  if (err != cudaSuccess) {std::cout << "on oy allocation of size: " << H*W*sizeof(unsigned char) << " error: " << (int)err << std::endl; return 0;}
  err = cudaMalloc(&ouv, H*W*sizeof(unsigned char)/2);
  if (err != cudaSuccess) {std::cout << "on ouv allocation of size: " << H*W*sizeof(unsigned char) << " error: " << (int)err << std::endl; return 0;}


  int error = rs(H, W, W, H, W, W, ty, tuv, oy, ouv, tbuff, method, eInterpolation);
  if (error != 0) std::cout << "Function Failure: " << error << std::endl;
  // output to RGB

  const Npp8u *pSrc[2] = {ty, tuv};
  oSizeROI.width  = W;
  oSizeROI.height = H;
#ifdef USE_709
  stat = nppiNV12ToRGB_709HDTV_8u_P2C3R(pSrc, W, rgbdata, W*3*sizeof(Npp8u), oSizeROI);
#else
  stat = nppiNV12ToRGB_8u_P2C3R(pSrc, W, rgbdata, W*3*sizeof(Npp8u), oSizeROI);
#endif
  if (stat != NPP_SUCCESS) { std::cout << "Output NPP error"  << std::endl; return 0;}
  cudaMemcpy(rfile.data, rgbdata, H*W*3, cudaMemcpyDeviceToHost);

  writeBMP("output.bmp", rfile);
  // 2x upscale test

  cudaFree(tbuff);
  buff_size = rs(H, W, W, 2*H, 2*W, 2*W, NULL, NULL, NULL, NULL, NULL);
  err = cudaMalloc(&tbuff, buff_size);
  if (err != cudaSuccess) {std::cout << "on temp buff allocation of size: " << buff_size << " error: " << (int)err << std::endl; return 0;}
  cudaFree(oy);
  cudaFree(ouv);
  err = cudaMalloc(&oy, 4*H*W*sizeof(unsigned char));
  if (err != cudaSuccess) {std::cout << "on oy allocation of size: " << H*W*sizeof(unsigned char) << " error: " << (int)err << std::endl; return 0;}
  err = cudaMalloc(&ouv, 2*H*W*sizeof(unsigned char));
  if (err != cudaSuccess) {std::cout << "on ouv allocation of size: " << H*W*sizeof(unsigned char) << " error: " << (int)err << std::endl; return 0;}

  unsigned long long dt = dtime_usec(0);
  error = rs(H, W, W, 2*H, 2*W, 2*W, ty, tuv, oy, ouv, tbuff, method, eInterpolation);
  cudaDeviceSynchronize();
  dt = dtime_usec(dt);
  if (error != 0) std::cout << "Function Failure: " << error << std::endl;
  std::cout << "2x resize time: " << dt/(float)USECPSEC << "s" << std::endl;
  // output to RGB

  const Npp8u *pSrc2[2] = {oy, ouv};
  oSizeROI.width  = 2*W;
  oSizeROI.height = 2*H;
  cudaFree(rgbdata);
  cudaMalloc(&rgbdata, H*W*12);
#ifdef USE_709
  stat = nppiNV12ToRGB_709HDTV_8u_P2C3R(pSrc2, 2*W, rgbdata, W*6*sizeof(Npp8u), oSizeROI);
#else
  stat = nppiNV12ToRGB_8u_P2C3R(pSrc2, 2*W, rgbdata, W*6*sizeof(Npp8u), oSizeROI);
#endif
  if (stat != NPP_SUCCESS) { std::cout << "Output NPP error"  << std::endl; return 0;}
  delete[] rfile.data;
  rfile.data = new unsigned char[H*W*12];
  cudaMemcpy(rfile.data, rgbdata, H*W*12, cudaMemcpyDeviceToHost);
  int osize = rfile.size - rfile.offset;
  int nsizeinc = H*W*12 - osize;
  rfile.size += nsizeinc;
  *((int*)(rfile.info+18)) = 2*W;
  *((int*)(rfile.info+22)) = 2*H;
  writeBMP("output2.bmp", rfile);
  return 0;
}
#endif