.PHONY : build clean test

TYPE ?= Release
KERNEL ?= matmul

build:
	mkdir -p build
	cmake -Bbuild -DCMAKE_BUILD_TYPE=$(TYPE) -DUSE_CUDA=ON
	make -j -C build

clean:
	rm -rf build
	rm -f info

test:
	@mkdir -p test_log
	@echo `date` >> test_log/`date`_$(KERNEL).log
	python3 test/$(KERNEL).py --device cuda >> test_log/`date`_$(KERNEL).log
info:
	@nvcc src/cuda_info/gpu/info.cu -o info
	@./info > info.txt
	@cat info.txt


