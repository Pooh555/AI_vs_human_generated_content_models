# %%
import torch

# %%
def test_CUDA():
    print("Torch CUDA version:",torch.version.cuda)
    print("CUDA GPU avalibility:", torch.cuda.is_available())

    if (torch.cuda.is_available()):
        print("CUDA devices number:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("Initial CUDA device:", torch.cuda.device(0))
        print("CUDA device:", torch.cuda.device)
        print("CUDA device's name:", torch.cuda.get_device_name(0))
    else:
        print("No CUDA GPU detected.")

# %%
test_CUDA()

# %%
import tensorflow as tf

# %%
gpu_available = tf.test.is_gpu_available()

# %%
print(gpu_available)


