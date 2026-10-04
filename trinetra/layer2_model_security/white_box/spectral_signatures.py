import os
import numpy as np
import matplotlib.pyplot as plt

def run_spectral_signatures(handle, inputs, preds, output_dir):
    activations = handle.get_activations(inputs)
    num_classes = np.max(preds) + 1
    
    max_outlier_score = 0.0
    threshold = 15.0
    flagged_classes = []
    
    suspect_class = None
    suspect_scores = []
    
    for c in range(num_classes):
        idx = np.where(preds == c)[0]
        if len(idx) < 10:
            continue
            
        acts_c = activations[idx]
        mean_act = np.mean(acts_c, axis=0)
        centered = acts_c - mean_act
        
        # SVD
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)
        top_right_vec = Vh[0]
        
        # Projection scores
        projections = np.dot(centered, top_right_vec)
        outlier_scores = projections ** 2
        
        # Median based thresholding (simplified robust stats)
        med = np.median(outlier_scores)
        mad = np.median(np.abs(outlier_scores - med))
        if mad == 0: mad = 1e-6
        
        # Max z-score
        z_scores = (outlier_scores - med) / (1.4826 * mad)
        c_max_score = np.max(z_scores)
        
        if c_max_score > max_outlier_score:
            max_outlier_score = float(c_max_score)
            suspect_class = c
            suspect_scores = outlier_scores.tolist()
            
    if suspect_class is not None and max_outlier_score > threshold:
        flagged_classes.append(suspect_class)
            
    plot_path = os.path.join(output_dir, "spectral_signature.png")
    if suspect_class is not None:
        plt.clf()
        plt.hist(suspect_scores, bins=20)
        plt.title(f"Spectral Signature Scores (Class {suspect_class})")
        plt.xlabel("Squared Projection")
        plt.ylabel("Count")
        plt.savefig(plot_path)
    plt.close()
    
    return {
        "data": {
            "max_outlier_score": max_outlier_score,
            "threshold": threshold,
            "flagged_classes": flagged_classes
        },
        "plot": plot_path if suspect_class is not None else None
    }
