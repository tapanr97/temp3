NVCC=nvcc

OPENCV_LIBPATH = /usr/local/lib
OPENCV_INCLUDEPATH = /usr/local/include
CUDA_INCLUDEPATH = /usr/local/cuda-9.0/include

CUDA_HELPERS_INCLUDEPATH = inc/cu_inc
C_INCLUDEPATH = inc/c_inc

NVCC_OPTS = -Xcompiler -m64 -Wno-deprecated-gpu-targets `pkg-config --cflags --libs opencv`
GCC_OPTS = -m64 `pkg-config --cflags --libs opencv`

blur: obj/main.o obj/load_save.o obj/blur_ops.o obj/edge_detection.o
	$(NVCC) -o $@ $^ -L $(OPENCV_LIBPATH) $(NVCC_OPTS)

obj/main.o: src/main.cpp
	g++ -o $@ -c $^ $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(C_INCLUDEPATH)

obj/load_save.o: src/util/load_save.cpp
	g++ -o $@ -c $^ -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

obj/blur_ops.o: src/kernels/blur_ops.cu
	$(NVCC) -o $@ -c $^ $(NVCC_OPTS) -I $(C_INCLUDEPATH)

obj/edge_detection.o: src/kernels/edge_detection.cu
	$(NVCC) -o $@ -c $^ $(NVCC_OPTS) -I $(CUDA_HELPERS_INCLUDEPATH)

clean:
	rm -f *.o blur
	rm -rf obj
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f
