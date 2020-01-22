#/usr/bin/python3

import os
import numpy as np
import time

import pynvml
pynvml.nvmlInit()
Number_of_gpus = pynvml.nvmlDeviceGetCount()

def check_available(gpu_id,min_memory):
    cuda_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(cuda_handle)
    mem = meminfo.free
    utilrate = pynvml.nvmlDeviceGetUtilizationRates(cuda_handle).gpu
    if utilrate<90 and mem>min_memory :
        return True
    else :
        return False

def check_waiting_list(fileno) :
    with open("./.bulletin.swp",'r') as fi:
        lines=fi.readlines()
    try :
        File_Numbers = np.array(lines[0].split(),dtype=np.int)
        last_run_info = np.array(lines[-1].split(),dtype=np.int)
    except ValueError :
        raise FileExistsError("Wrong Format .bullentin.swp")

    runNo = np.where(File_Numbers==fileno)[0]
    if len(runNo)==0 : 
        raise ValueError("unknown FileNo!")
    else :
        runNo = runNo[0]
    
    if runNo==0 :
        if len(lines)!=1 :
            raise FileExistsError("Wrong Format .bullentin.swp")
        else :
            return True
    elif len(lines)==(runNo+1) :
        if last_run_info[0]==File_Numbers[runNo-1] :
            if last_run_info[1]==0 :
                return True
            else :
                raise RuntimeError('Load taining Data and model failed, file numer={}'.format(File_Numbers[runNo-1]))
        else :
            raise FileExistsError("Wrong Format .bullentin.swp")
    return False

def QueueUp(fileno):
    lines=[""]
    L=[""]
    if os.path.exists(".bulletin.swp"):
        with open(".bulletin.swp",'r') as file:
            lines=file.readlines()
        L = lines[0].split()
        if str(fileno) in set(L) :
            return True #already in line (successful)
        L=np.append(L,str(fileno))
    else :
        L[0]=str(fileno)
    L = ' '.join(L)
    lines[0]=L+'\n'
    with open(".bulletin.swp",'w') as file:
        file.writelines(lines)
    return False #newly append, may not success. Need rerun QueueUp to check

def wait_in_line(fileno,min_memory) :
    GPUs = np.arange(pynvml.nvmlDeviceGetCount())
    device = GPUs[-1]
    #wait in line
    while not check_waiting_list(fileno) :
        time.sleep(0.5)
    #your turn, search for idle gpu!
    while not check_available(device,min_memory) : 
        if device==0 : device = GPUs[-1]
        else : device -= 1
        time.sleep(0.5)
    device = int(device)
    print('Using device: gpu {}'.format(device))
    return device
