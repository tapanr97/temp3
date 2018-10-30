NVCC=nvcc

####################################
# OpenCV default install locations #
# Check yours and replace.         #
####################################

OPENCV_LIBPATH=/usr/local/lib
OPENCV_INCLUDEPATH=/usr/local/include

NVCC_OPTS=-Xcompiler -m64 -Wno-deprecated-gpu-targets `pkg-config --cflags --libs opencv`
GCC_OPTS=-m64 `pkg-config --cflags --libs opencv`

photops: main.o load_save.o blur_ops.o Makefile
	$(NVCC) -o photops main.o load_save.o blur_ops.o -L $(OPENCV_LIBPATH) -lboost_program_options $(NVCC_OPTS)

main.o: main.cpp include/load_save.h include/square_ops.h include/mirror_ops.h include/blur_ops.h include/filter_ops.h
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

load_save.o: load_save.cpp include/load_save.h
	g++ -c load_save.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

blur_ops.o: blur_ops.cu include/load_save.h include/blur_ops.h
	$(NVCC) -c blur_ops.cu $(NVCC_OPTS)

clean:
	rm -f *.o photops
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f
