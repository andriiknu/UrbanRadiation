import matplotlib.pyplot as plt


def plot_train(history):
    # Plot accuracy
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()



# def monitor_resources(stop_monitoring, interval=1.0):

#     while not stop_monitoring.is_set():
#         gpus = GPUtil.getGPUs()
#         if gpus:
#             gpu = gpus[0]  # Assumes only one GPU is used
#             gpu_load = gpu.load * 100
#             gpu_mem_used = gpu.memoryUsed
#             gpu_mem_total = gpu.memoryTotal
#             max_gpu_usage = max(max_gpu_usage, gpu_load)
#         else:
#             gpu_load = gpu_mem_used = gpu_mem_total = 0

#         cpu_load = psutil.cpu_percent(interval=interval)
#         mem_info = psutil.virtual_memory()
#         mem_used = mem_info.used / (1024 ** 3)  # Convert to GB
#         mem_total = mem_info.total / (1024 ** 3)  # Convert to GB

#         print(f"GPU Load: {gpu_load:.2f}% | GPU Mem: {gpu_mem_used:.2f}GB / {gpu_mem_total:.2f}GB")
#         print(f"CPU Load: {cpu_load:.2f}% | RAM: {mem_used:.2f}GB / {mem_total:.2f}GB")

#         time.sleep(interval)

