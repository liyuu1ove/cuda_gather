import torch
import time
import logging
def CudaProfile(*function_with_args):
    times = 20
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    end_event.record()
    # 等待事件完成
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)  # 以毫秒为单位        
    return elapsed_time/times
def CpuProfile(*function_with_args):
    times = 20
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    start = time.time()
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    
    elapsed_time = time.time() - start  # 以毫秒为单位        
    return 1000 * elapsed_time/times
def logValidation(torch,kernel):
    atol = max(abs(torch - kernel))

    rtol = atol / max(abs(kernel) + 1e-8) * 100
    msg=""
    if atol>1e-03:
        msg+="\033[31m"+"validation failed!"+"\033[0m"
    else:
        msg+="\033[32m"+"validation passed!"+"\033[0m"
    atol_string="{:.2e}".format(atol)
    rtol_string="{:.2f}%".format(rtol)
    msg+="\n"+"absolute error:"+atol_string+"\n"
    msg+="relative error:"+rtol_string+"\n"

    print(msg)


def logBenchmark(baseline, time,total_GFLOPS):
    unitlist=[" GFOLPs "," TFLOPs "]
    msg = "\n"+"Pytorch: " + "{:.3f}".format(baseline) + " ms "

    GFLOPs_Pytorch=total_GFLOPS/baseline*1000
    if(GFLOPs_Pytorch>1000):
        GFLOPs_Pytorch/=1000
        msg+="{:.3f}".format(GFLOPs_Pytorch)+unitlist[1]
    else:
        msg+="{:.3f}".format(GFLOPs_Pytorch)+unitlist[0]
    msg+="\nkernel: " + "{:.3f}".format(time) + " ms "
    
    GFLOPs_kernel=total_GFLOPS/time*1000
    if(GFLOPs_kernel>1000):
        GFLOPs_kernel/=1000
        msg+="{:.3f}".format(GFLOPs_kernel)+unitlist[1]
    else:
        msg+="{:.3f}".format(GFLOPs_kernel)+unitlist[0]

    percentage = "{:.2f}%".format(abs(baseline - time)/baseline * 100)
    if baseline >= time:
        print(msg + "\033[32m" + "[-" + percentage + "]" +"\033[0m")
    else:
        print(msg + "\033[31m" + "[+" + percentage + "]" +"\033[0m")