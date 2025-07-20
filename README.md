# CUDA 算子框架
本框架fork自肖老师的框架 https://github.com/xgqdut2016/hpc2torch

删除了算子的原有实现，修复了框架的一些问题，增加了几个profile函数，完善了性能测试的内容

将.cu文件编译为动态库，使用了ctypes进行python对cuda函数的调用

在该框架的基础上进行cuda开发，因为找不到好用（免费）的服务器，只能在本人的可怜的4060laptop上进行开发

## make命令
- 查看显卡基本参数

  ```shell
  make info
  ```

  result will both be saved in info.txt and show in terminal
- 编译项目

  ```shell
  make
  ```

- 清理项目

  ```shell
  make clean
  ```

- 测试算子

  ```shell
  make test KERNEL=<kernel_name>
  ```
  list of kernel names can be found below at `operaters` part
# operaters
## matmul
### defination
### test cases
### performance
## gather
### defination
参照 [ONNX 的 Gather 算子文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather)实现一个 CUDA kernel。该 kernel 需要接收三个输入参数：`data`、`indices` 和 `axis`，并输出 `output`。`axis` 的值缺省为 0。在开始实现之前，建议先访问 cuDNN 的官方网站，查询是否有现成的库函数可用于实现 Gather 操作；如果存在现成的库函数，可以同时添加库函数以及手写 cuda 算子的实现作为对比。
###




