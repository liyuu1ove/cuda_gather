.PHONY : build clean test

TYPE ?= Release
KERNEL ?= attention

build:
	mkdir -p build
	cmake -Bbuild -DCMAKE_BUILD_TYPE=$(TYPE) -DUSE_CUDA=ON
	make -j -C build

clean:
	rm -rf build
	rm -f info

test:
	python3 test/$(KERNEL).py --device cuda

info:
	@nvcc src/cuda_info/gpu/info.cu -o info
	@./info > info.txt
	@cat info.txt

