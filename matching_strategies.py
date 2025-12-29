#argmax function to find best matching patch
import torch
import torch.nn.functional as F

def find_best_match_argmax(s, width):
    best_match_idx = s.argmax().item()#argmax over the similarities
    y = best_match_idx // width
    x = best_match_idx % width
    return x, y

#will later add more matching strategies here

def find_best_match_window_softargmax(
    s: torch.Tensor,
    width: int,
    height: int,
    K: int = 5,
    temperature: float = 1.0,
):
    """
    s: [H*W] similarities
    width: W
    height: H
    K: window size in patches (odd number, e.g. 3,5,7)
    temperature: softmax temperature (softmax(s / temperature))
    returns (x_hat, y_hat) as continuous patch coordinates (floats)
    """
    assert K % 2 == 1, "K must be odd"

    # reshape to 2D similarity map [H, W]
    sim_map = s.view(height, width)  # [H, W]

    # hard argmax to find window center
    cx, cy = find_best_match_argmax(s, width)

    # half window
    r = K // 2

    # window bounds (clamped to image)
    y_min = max(cy - r, 0)
    y_max = min(cy + r + 1, height)  # exclusive
    x_min = max(cx - r, 0)
    x_max = min(cx + r + 1, width)   # exclusive

    # crop window
    window = sim_map[y_min:y_max, x_min:x_max]  # [h_win, w_win]

    # build coordinate grid for patches in window
    ys = torch.arange(y_min, y_max, device=s.device)
    xs = torch.arange(x_min, x_max, device=s.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [h_win, w_win]

    # softmax over window with temperature
    # larger temperature -> flatter distribution
    logits = window / temperature
    weights = F.softmax(logits.view(-1), dim=0)  # [h_win*w_win]

    grid_x_flat = grid_x.reshape(-1).float()
    grid_y_flat = grid_y.reshape(-1).float()

    # soft-argmax expectation in patch space
    x_hat = (weights * grid_x_flat).sum()
    y_hat = (weights * grid_y_flat).sum()

    return x_hat.item(), y_hat.item()