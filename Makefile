################################################################################
# Copyright (c) 2017-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#################################################################################

CUDA_VER?=10.2
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX:= g++
SRCS:= gstnvinfer.cpp  gstnvinfer_allocator.cpp gstnvinfer_property_parser.cpp \
       gstnvinfer_meta_utils.cpp gstnvinfer_impl.cpp gstnvinfer_yaml_parser.cpp \
	   align_functions.cpp tensor_extractor.cpp
INCS:= $(wildcard *.h)
LIB:=libnvdsgst_infer.so

NVDS_VERSION:=6.0

CFLAGS+= -fPIC -std=c++11 -DDS_VERSION=\"6.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I /opt/nvidia/deepstream/deepstream-6.0/sources/includes \
	 -I /opt/nvidia/deepstream/deepstream-6.0/sources/gst-plugins/gst-nvdspreprocess/include \
	 -I /opt/nvidia/deepstream/deepstream-6.0/sources/libs/nvdsinfer -DNDEBUG

-D_GLIBCXX_USE_CXX11_ABI=0
CFLAGS+=-I /usr/include/opencv4 
		

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
       -lnvds_infer -lnvbufsurface -lnvbufsurftransform -ldl -lpthread -lyaml-cpp \
       -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)
	   

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 /usr/lib/aarch64-linux-gnu/pkgconfig/opencv.pc
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
