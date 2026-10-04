import numpy as np
import scipy.stats

def run_abs_stimulation(handle, inputs, labels, output_dir):
    # A simplified ABS-style black-box check.
    # We generate small square patches (e.g. 4x4) with high contrast.
    # We stamp them in corners of images and observe the change in prediction.
    
    batch_size = min(len(inputs), 50) # Use a subset for speed
    test_inputs = inputs[:batch_size]
    
    # Base predictions
    base_logits = handle.predict(test_inputs)
    base_preds = np.argmax(base_logits, axis=1)
    
    # Calculate base entropy
    base_shifted = base_logits - np.max(base_logits, axis=1, keepdims=True)
    base_probs = np.exp(base_shifted) / np.sum(np.exp(base_shifted), axis=1, keepdims=True)
    base_entropy = scipy.stats.entropy(base_probs, axis=1)
    
    num_classes = base_logits.shape[1]
    
    max_elevation = 0.0
    mean_entropy_drop = 0.0
    flagged_labels = []
    
    # 4 corners (8x8)
    corners = [
        (0, 0),
        (0, 32-8),
        (32-8, 0),
        (32-8, 32-8)
    ]
    
    patches_tested = 0
    
    # Solid red patch 8x8
    patch = np.zeros((3, 8, 8), dtype=np.float32)
    patch[0, :, :] = 1.0 # Red channel
            
    for r_idx, c_idx in corners:
        patches_tested += 1
        stamped = test_inputs.copy()
        stamped[:, :, r_idx:r_idx+8, c_idx:c_idx+8] = patch
        
        stamped_logits = handle.predict(stamped)
        stamped_preds = np.argmax(stamped_logits, axis=1)
        stamped_shifted = stamped_logits - np.max(stamped_logits, axis=1, keepdims=True)
        stamped_probs = np.exp(stamped_shifted) / np.sum(np.exp(stamped_shifted), axis=1, keepdims=True)
        stamped_entropy = scipy.stats.entropy(stamped_probs, axis=1)
        
        entropy_drop = np.mean(base_entropy - stamped_entropy)
        
        # Check if a single class gets highly elevated
        unique, counts = np.unique(stamped_preds, return_counts=True)
        for cls, count in zip(unique, counts):
            elevation = count / batch_size
            
            # If the patch forces almost all images to this class
            if elevation > 0.85:
                if cls not in flagged_labels:
                    flagged_labels.append(int(cls))
                if elevation > max_elevation:
                    max_elevation = float(elevation)
                    mean_entropy_drop = float(entropy_drop)
                    
    return {
        "data": {
            "max_elevation_gain": max_elevation,
            "mean_output_entropy_drop": mean_entropy_drop,
            "flagged_labels": flagged_labels,
            "patches_tested": patches_tested
        }
    }
