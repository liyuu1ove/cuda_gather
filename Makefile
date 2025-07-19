.PHONY : build clean test

TYPE ?= Release
KERNEL ?= attention

build:
	mkdir -p build
	cmake -Bbuild -DCMAKE_BUILD_TYPE=$(TYPE) -DUSE_CUDA=ON
	make -j -C build

clean:
	rm -rf build

test:
	python3 test/$(KERNEL).py --device cuda
