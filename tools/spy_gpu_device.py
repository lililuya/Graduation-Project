import time
import nvidia_smi

# 初始化nvidia-smi
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(1)  # 获取第一张显卡的句柄

# 持续监测并写入文件
print("----spy on gpu------")
with open("gpu_usage.log", "w") as file:
    while True:
        try:
            info = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = info.gpu
            memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            memory_used = memory_info.used / (1024**2)  # 转换为MB
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

            # 写入文件
            file.write(f"{timestamp} - GPU Usage: {gpu_usage}%, Memory Used: {memory_used:.2f} MB\n")
            file.flush()

            # 等待1秒
            time.sleep(1)

        except KeyboardInterrupt:
            break

# 清理资源
nvidia_smi.nvmlShutdown()