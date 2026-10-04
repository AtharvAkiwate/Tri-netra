import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def run_activation_clustering(handle, inputs, preds, output_dir):
    # Get activations for all inputs
    activations = handle.get_activations(inputs)
    num_classes = np.max(preds) + 1
    
    max_silhouette = -1.0
    suspect_class = None
    suspect_sizes = []
    suspect_rel_size = 0.0
    
    flagged_classes = []
    
    plt.figure(figsize=(10, 8))
    
    for c in range(num_classes):
        idx = np.where(preds == c)[0]
        if len(idx) < 10:
            continue
            
        acts_c = activations[idx]
        
        # PCA to 10 dims
        n_components = min(10, acts_c.shape[0], acts_c.shape[1])
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(acts_c)
        
        # KMeans
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(reduced)
        
        # Silhouette
        try:
            sil = silhouette_score(reduced, clusters)
        except ValueError:
            sil = 0.0
            
        c0_size = np.sum(clusters == 0)
        c1_size = np.sum(clusters == 1)
        rel_size = min(c0_size, c1_size) / len(idx)
        
        if sil > max_silhouette:
            max_silhouette = float(sil)
            suspect_class = c
            suspect_sizes = [int(c0_size), int(c1_size)]
            suspect_rel_size = float(rel_size)
            
    if suspect_class is not None and max_silhouette > 0.45 and suspect_rel_size > 0.25:
        flagged_classes.append(suspect_class)
            
    plot_path = os.path.join(output_dir, "activation_cluster.png")
    if suspect_class is not None:
        plt.savefig(plot_path)
    plt.close()
    
    return {
        "data": {
            "silhouette_score": max_silhouette,
            "cluster_sizes": suspect_sizes,
            "flagged_classes": flagged_classes,
            "relative_size_of_small_cluster": suspect_rel_size
        },
        "plot": plot_path if suspect_class is not None else None
    }
