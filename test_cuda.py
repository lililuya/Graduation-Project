import torch
print(torch.cuda.device_count())  # This will print the number of GPUs available
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])  # Names of GPUs
