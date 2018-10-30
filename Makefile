NVCC=nvcc

OPENCV_LIBPATH = /usr/local/lib
OPENCV_INCLUDEPATH = /usr/local/include
CUDA_INCLUDEPATH = /usr/local/cuda-9.0/include

CUDA_HELPERS_INCLUDEPATH = inc/cu_inc
C_INCLUDEPATH = inc/c_inc

NVCC_OPTS = -Xcompiler -m64 -Wno-deprecated-gpu-targets `pkg-config --cflags --libs opencv`
GCC_OPTS = -m64 `pkg-config --cflags --libs opencv`

blur: main.o load_save.o blur_ops.o Makefile
	$(NVCC) -o blur main.o load_save.o blur_ops.o -L $(OPENCV_LIBPATH) $(NVCC_OPTS)

edged: edge_detection.o Makefile
	$(NVCC) -o edged edge_detection.o -L $(OPENCV_LIBPATH) $(NVCC_OPTS)

main.o: src/main.cpp
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(C_INCLUDEPATH)

load_save.o: src/utils/load_save.cpp
	g++ -c load_save.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

blur_ops.o: src/kernels/blur_ops.cu
	$(NVCC) -c blur_ops.cu $(NVCC_OPTS) -I $(C_INCLUDEPATH)

edge_detection.o: src/kernels/edge_detection.cu/
	$(NVCC) -c edge_detection.cu $(NVCC_OPTS) -I $(CUDA_HELPERS_INCLUDEPATH)

all: blur edged

clean:
	rm -f *.o photops
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f
