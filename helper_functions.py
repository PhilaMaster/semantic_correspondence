import torch


def extract_dense_features(model, img_tensor, training=False):
    """Extract dense features from DINOv2 model given an input image tensor."""
    context = torch.no_grad() if not training else torch.enable_grad()

    with context:
        #get tokens
        features_dict = model.forward_features(img_tensor)
        patch_tokens = features_dict['x_norm_patchtokens']  # [B, N_patches, D]

        #reshaping to dense feature map
        B, N, D = patch_tokens.shape
        H_patches = W_patches = int(N ** 0.5)  # per img 518x518 con patch 14: 37x37
        dense_features = patch_tokens.reshape(B, H_patches, W_patches, D)
    return dense_features


def extract_layer_features(model, img_tensor, layer_idx):
    with torch.no_grad():

        # get_intermediate_layers returns patch tokens only (CLS + storage are already stripped)
        patch_tokens = model.get_intermediate_layers(img_tensor, n=[layer_idx], norm=True)[0]  # [B, N_patches, D]
        
        #reshaping to dense feature map
        B, N, D = patch_tokens.shape
        H_patches = W_patches = int(N ** 0.5) 
        dense_features = patch_tokens.reshape(B, H_patches, W_patches, D)
    return dense_features


def pixel_to_patch_coord(x, y, original_size, patch_size=14, resized_size=518):
    """convert pixel coordinates to patch coordinates"""
    #scale to resized image
    scale_x = resized_size / original_size[0]
    scale_y = resized_size / original_size[1]
    x_resized = x * scale_x
    y_resized = y * scale_y

    #compute patch coordinates
    patch_x = int(x_resized // patch_size)
    patch_y = int(y_resized // patch_size)

    #clamp to valid range
    max_patch = resized_size // patch_size - 1
    patch_x = min(max(patch_x, 0), max_patch)
    patch_y = min(max(patch_y, 0), max_patch)

    return patch_x, patch_y


def patch_to_pixel_coord(patch_x, patch_y, original_size, patch_size=14, resized_size=518):
    """Convert patch coordinates back to pixel coordinates with a centering strategy"""
    #center of the patch in resized image
    x_resized = patch_x * patch_size + patch_size / 2
    y_resized = patch_y * patch_size + patch_size / 2

    #scale back to original image size
    scale_x = original_size[0] / resized_size
    scale_y = original_size[1] / resized_size
    x = x_resized * scale_x
    y = y_resized * scale_y

    return x, y