import torch

def pad_loop_torch(x: torch.Tensor, max_len: int):
    x_len = x.shape[-1]
    
    if x_len > max_len:
        stt = torch.randint(0, x_len - max_len + 1, (1,)).item()
        return x[..., stt : stt + max_len]
    elif x_len < max_len:
        num_repeats = (max_len // x_len) + 1
        padded_x = x.repeat(1, num_repeats)
        return padded_x[..., :max_len]
    else:
        return x