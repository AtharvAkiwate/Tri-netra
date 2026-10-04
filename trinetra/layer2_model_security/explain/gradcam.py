import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def generate_heatmap(handle, inputs, labels, target_class, output_dir):
    # Select an image that was predicted as target_class
    preds = np.argmax(handle.predict(inputs), axis=1)
    idx = np.where(preds == target_class)[0]
    
    if len(idx) == 0:
        idx = [0] # Fallback
    
    test_img = inputs[idx[0]:idx[0]+1]
    
    heatmap = None
    if handle.white_box_available and handle.format == "pytorch":
        # Grad-CAM style approximation for our SimpleCNN
        handle.model.eval()
        tensor_in = torch.tensor(test_img, dtype=torch.float32, requires_grad=True)
        
        # We need to capture gradients at the last conv layer
        # For our SimpleCNN, that's features[6] or we can just capture the output of features
        
        acts = None
        grads = None
        
        def forward_hook(module, input, output):
            nonlocal acts
            acts = output
            
        def backward_hook(module, grad_in, grad_out):
            nonlocal grads
            grads = grad_out[0]
            
        # Attach hooks to the last conv layer (Conv2d is at index 6)
        hook_f = handle.model.features[6].register_forward_hook(forward_hook)
        hook_b = handle.model.features[6].register_backward_hook(backward_hook)
        
        logits = handle.model(tensor_in)
        
        score = logits[0, target_class]
        score.backward()
        
        hook_f.remove()
        hook_b.remove()
        
        if acts is not None and grads is not None:
            acts = acts.detach().numpy()[0]
            grads = grads.detach().numpy()[0]
            weights = np.mean(grads, axis=(1, 2))
            
            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]
                
            cam = np.maximum(cam, 0)
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            heatmap = cam
    
    if heatmap is None:
        # Occlusion sensitivity for black-box/fallback
        # Slide a small patch (e.g., 4x4) over the image and observe score drop
        heatmap = np.zeros((32, 32), dtype=np.float32)
        base_logits = handle.predict(test_img)
        base_score = base_logits[0, target_class]
        
        patch_size = 4
        stride = 2
        for r in range(0, 32 - patch_size + 1, stride):
            for c in range(0, 32 - patch_size + 1, stride):
                occ_img = test_img.copy()
                occ_img[0, :, r:r+patch_size, c:c+patch_size] = 0.5 # Gray patch
                
                occ_logits = handle.predict(occ_img)
                occ_score = occ_logits[0, target_class]
                
                drop = max(0, base_score - occ_score)
                heatmap[r:r+patch_size, c:c+patch_size] += drop
                
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
    # Resize heatmap to 32x32 for plotting (if it's gradcam, it might be 8x8)
    import cv2
    if heatmap.shape != (32, 32):
        heatmap = cv2.resize(heatmap, (32, 32))
        
    # Plotting
    plot_path = os.path.join(output_dir, "heatmap.png")
    plt.clf()
    
    # Original image (transpose back to HWC)
    img_hwc = np.transpose(test_img[0], (1, 2, 0))
    plt.imshow(img_hwc)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(f"Threat Heatmap (Class {target_class})")
    plt.axis('off')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path
