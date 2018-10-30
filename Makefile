NVCC=nvcc

####################################
# OpenCV default install locations #
# Check yours and replace.         #
####################################

OPENCV_LIBPATH = /usr/local/lib
OPENCV_INCLUDEPATH = /usr/local/include

CUDA_INCLUDEPATH = /usr/local/cuda-9.0/include

NVCC_OPTS = -Xcompiler -m64 -Wno-deprecated-gpu-targets `pkg-config --cflags --libs opencv`
GCC_OPTS = -m64 `pkg-config --cflags --libs opencv`

all: blur edged

blur: main.o load_save.o blur_ops.o Makefile
	$(NVCC) -o blur main.o load_save.o blur_ops.o -L $(OPENCV_LIBPATH) $(NVCC_OPTS)

edged: edge_detection.o Makefile
	$(NVCC) -o edged edge_detection.o -L $(OPENCV_LIBPATH) $(NVCC_OPTS)

main.o: main.cpp load_save.h blur_ops.h
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

load_save.o: load_save.cpp load_save.h
	g++ -c load_save.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

blur_ops.o: blur_ops.cu load_save.h blur_ops.h
	$(NVCC) -c blur_ops.cu $(NVCC_OPTS)

edge_detection.o: edge_detection.cu inc/helper_image.h
	$(NVCC) -c edge_detection.cu $(NVCC_OPTS)

clean:
	rm -f *.o photops
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f
