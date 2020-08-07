import sys


gpu = 0


def on_set(k, v):
    if k == 'gpu':
        global gpu
        gpu = int(v)


def on_get(k):
    if k == 'gpu':
        return str(gpu)


def on_init():
    import torch
    if not torch.cuda.is_available():
        sys.stderr.write(f"[select_gpus.on_init] Pytorch Cuda is not available!.")
        sys.stderr.flush()
        return True

    device_count = torch.cuda.device_count()

    if device_count <= gpu:
        sys.stderr.write(f"[select_gpus.on_init] Gpu's index is not available. (all gpus: {device_count}, select gpu: {gpu})")
        sys.stderr.flush()
        return True

    # Set GPU.
    torch.cuda.set_device(gpu)

    return True


def on_run():
    pass
