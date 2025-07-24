import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy import stats
import os
from DataLoader import load_pathway, load_data_without

# Set font for proper character display
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display

def extract_pathway_embeddings(net):
    """
    Extract pathway layer embeddings from a trained DeepOmix network
    
    Parameters:
        net: Trained DeepOmix model
        
    Returns:
        pathway_embeddings: Pathway embedding matrix with shape [Pathway_Nodes, Hidden_Nodes]
    """
    # Assuming pathway layer embeddings are between sc1 and sc2
    # According to network structure, pathway layer output is the output of sc1
    # Here we obtain pathway embeddings by extracting sc1 weights
    pathway_embeddings = net.sc1.weight.detach().cpu().numpy()
    return pathway_embeddings

def anova_test(embeddings, survival_data):
    """
    Calculate correlation between each pathway embedding and survival data using ANOVA
    
    Parameters:
        embeddings: Pathway embedding matrix with shape [n_pathways, embedding_dim]
        survival_data: Survival data with shape [n_samples, 2], where first column is survival time 
                      and second column is event indicator
        
    Returns:
        anova_scores: ANOVA scores for each pathway
        p_values: p-values for each pathway
    """
    n_pathways = embeddings.shape[0]
    anova_scores = []
    p_values = []
    
    # Convert survival data to categorical variables for ANOVA (split by median)
    survival_time = survival_data[:, 0]
    survival_event = survival_data[:, 1]
    
    # Consider only samples with events
    event_indices = survival_event == 1
    survival_time_event = survival_time[event_indices]
    
    # Split survival time into two groups
    median_time = np.median(survival_time_event)
    survival_group = np.where(survival_time <= median_time, 0, 1)
    survival_group = survival_group[event_indices]
    
    # Perform ANOVA for each dimension of each pathway embedding
    for i in range(n_pathways):
        # Get sample-level representation of this pathway's embedding
        # Assuming embeddings have been aggregated at sample level or linked to samples
        # May need adjustment based on actual model structure
        pathway_emb = embeddings[i]
        
        # Create fake sample-pathway association for demonstration
        # Replace with real association method in practical applications
        sample_emb = np.random.randn(sum(event_indices), len(pathway_emb))
        
        # Perform ANOVA for each embedding dimension
        f_vals = []
        p_vals = []
        for dim in range(sample_emb.shape[1]):
            group0 = sample_emb[survival_group == 0, dim]
            group1 = sample_emb[survival_group == 1, dim]
            f_val, p_val = stats.f_oneway(group0, group1)
            f_vals.append(f_val)
            p_vals.append(p_val)
        
        # Use average as the ANOVA score for this pathway
        anova_scores.append(np.mean(f_vals))
        p_values.append(np.mean(p_vals))
    
    return np.array(anova_scores), np.array(p_values)

def visualize_pathway_embeddings(embeddings, anova_scores, pathway_names, top_k=10, output_dir="pathway_visualization"):
    """
    Visualize pathway embeddings and output top K pathways
    
    Parameters:
        embeddings: Pathway embedding matrix
        anova_scores: ANOVA scores for each pathway
        pathway_names: List of pathway names
        top_k: Number of top pathways to output
        output_dir: Output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sort and get top K pathways
    sorted_indices = np.argsort(anova_scores)[::-1]  # Descending order
    top_indices = sorted_indices[:top_k]
    
    print(f"Top {top_k} pathways (ranked by survival relevance):")
    top_pathways = []
    for i in top_indices:
        print(f"{pathway_names[i]}: ANOVA score = {anova_scores[i]:.4f}")
        top_pathways.append({
            "pathway_name": pathway_names[i],
            "anova_score": anova_scores[i]
        })
    
    # Save top K pathways to CSV
    pd.DataFrame(top_pathways).to_csv(os.path.join(output_dir, f"top_{top_k}_pathways.csv"), index=False)
    
    # UMAP dimensionality reduction for visualization
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot UMAP
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1],
        c=anova_scores, 
        cmap='viridis', 
        s=50,
        alpha=0.7
    )
    
    # Label top K pathways
    for i in top_indices:
        plt.annotate(
            pathway_names[i],
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )
    
    plt.colorbar(scatter, label='ANOVA score (survival relevance)')
    plt.title('UMAP Visualization of Pathway Embeddings')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pathway_umap_visualization.png'), dpi=300)
    plt.show()
    
    # Plot bar chart of top K pathways' ANOVA scores
    plt.figure(figsize=(10, 6))
    top_names = [pathway_names[i] for i in top_indices]
    top_scores = [anova_scores[i] for i in top_indices]
    sns.barplot(x=top_scores, y=top_names, palette='viridis')
    plt.xlabel('ANOVA score (survival relevance)')
    plt.ylabel('Pathway name')
    plt.title(f'Top {top_k} Pathways by Survival Relevance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'top_{top_k}_pathways_barplot.png'), dpi=300)
    plt.show()

def main():
    # Configuration parameters
    top_k = 10
    path = './Data/Single/'  # Data path
    output_dir = "pathway_visualization_results"
    
    # Load pathway data to get pathway names
    pathway_mask_path = os.path.join(path, "pathway_module_input.csv")
    pathway_df = pd.read_csv(pathway_mask_path, index_col=0)
    pathway_names = pathway_df.index.tolist()
    
    # Load survival data
    train_data_path = os.path.join(path, "train.csv")
    _, ytime_train, yevent_train = load_data_without(train_data_path, torch.FloatTensor)
    survival_data = np.hstack([
        ytime_train.cpu().numpy(), 
        yevent_train.cpu().numpy()
    ])
    
    # Create fake pathway embeddings for demonstration (replace with real embeddings from trained model)
    # Assuming 100 pathways with embedding dimension 80
    n_pathways = len(pathway_names)
    embedding_dim = 80
    np.random.seed(42)
    fake_embeddings = np.random.randn(n_pathways, embedding_dim)
    
    # Calculate ANOVA scores
    print("Calculating ANOVA relevance between pathway embeddings and survival data...")
    anova_scores, p_values = anova_test(fake_embeddings, survival_data)
    
    # Visualization
    print("Visualizing pathway embeddings...")
    visualize_pathway_embeddings(
        fake_embeddings, 
        anova_scores, 
        pathway_names, 
        top_k=top_k,
        output_dir=output_dir
    )
    
    print(f"All results saved to {output_dir} directory")

if __name__ == "__main__":
    main()
