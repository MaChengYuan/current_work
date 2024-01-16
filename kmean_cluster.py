import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


def plot_tsne(model,tokenizer , df , max_len=40):
    df['output'] = df['sentence'].apply(lambda x : process(model,tokenizer,x,max_len)[:max_len][0].detach())
    
    X_tensor = df.output
    X = np.array([t.numpy() for t in X_tensor])
    y = df.label
    # We want to get TSNE embedding with 2 dimensions
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    tsne_result.shape
    # (1000, 2)
    # Two dimensions for each of our images
    
    kl_divergence_ = tsne.kl_divergence_
    print(f'kl_divergence_: {kl_divergence_}')
    
    # Plot the result of our TSNE with the label color coded
    # A lot of the stuff here is about making the plot look pretty and not TSNE
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    return kl_divergence_


def preprocess(tokenizer,sentence,max_len):
#1.Tokenize the sequence:
    tokens=tokenizer.tokenize(sentence)
    tokens = tokens[:max_len]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    # max_len
    
    padded_tokens=tokens +['[PAD]' for _ in range(max_len-len(tokens))]
    
    attn_mask=[ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]
    
    
    seg_ids=[0 for _ in range(len(padded_tokens))]
    
    sent_ids=tokenizer.convert_tokens_to_ids(padded_tokens)
    
    token_ids = torch.tensor(sent_ids).unsqueeze(0) 
    attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
    seg_ids   = torch.tensor(seg_ids).unsqueeze(0)
    return token_ids,attn_mask,seg_ids
    
def get_output(model,token_ids,attn_mask):
    
    output = model(token_ids, attention_mask = attn_mask)

    return output
# last_hidden_state , pooler_output
def process(model,tokenzier,sentence,max_len):
    token_ids,attn_mask,seg_ids = preprocess(tokenzier,sentence,max_len)
    output = get_output(model,token_ids,attn_mask)
    
    return output[1]


#plotting
# watch out label and color numbers 
import time
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt
from sklearn import preprocessing


def kmeans_analysis_len(model,tokenizer , df , max_len):    
    df['output'] = df['sentence'].apply(lambda x : process(model,tokenizer,x,max_len)[:max_len].detach())
    
    n_clusters = len(df.label.unique())
    X = np.array(df['output'].tolist())
    
    X = np.squeeze(X, axis=(1,))
    X = preprocessing.normalize(X)
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10)
    k_means.fit(X)
    X_dist = k_means.transform(X)**2
    df['distance'] = X_dist.sum(axis=1).round(2)
    y = df['label'].tolist()
    
    distance_by_each_label = df.groupby('label').sum('distance').distance.tolist()

    return distance_by_each_label
def kmeans_analysis(model,tokenizer , df , max_len):    
    df['output'] = df['sentence'].apply(lambda x : process(model,tokenizer,x,max_len)[:max_len].detach())
    
    n_clusters = len(df.label.unique())
    X = np.array(df['output'].tolist())
    
    X = np.squeeze(X, axis=(1,))
    X = preprocessing.normalize(X)
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10)
    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0
    
    # distance
    X_dist = k_means.transform(X)**2
    df['distance'] = X_dist.sum(axis=1).round(2)
    y = df['label'].tolist()
    
    distance_by_each_label = df.groupby('label').sum('distance').distance.tolist()
    
    #farest point
    max_indices = []
    for label in np.unique(k_means.labels_):
        X_label_indices = np.where(y==label)[0]
        max_label_idx = X_label_indices[np.argmax(X_dist[y==label].sum(axis=1))]
        max_indices.append(max_label_idx)
    
    batch_size = 8
    
    mbk = MiniBatchKMeans(
        init="k-means++",
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init=10,
        max_no_improvement=10,
        verbose=0,
    )
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0

    
    k_means_cluster_centers = k_means.cluster_centers_
    order = pairwise_distances_argmin(k_means.cluster_centers_, mbk.cluster_centers_)
    mbk_means_cluster_centers = mbk.cluster_centers_[order]
    
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)

    
    
    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    
    from matplotlib import colors as mcolors

    
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    
    # Sort colors by hue, saturation, value and name.
    by_hsv = ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    colors = [name for hsv, name in by_hsv]

    
    # KMeans
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(n_clusters), colors[:n_clusters]):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".",label = k )
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
        
       # farest points 
       # ax.plot(X[max_indices, 0], X[max_indices, 1], "o", color='black',markersize=3)
    ax.set_title("KMeans")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend()


    # MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors[:n_clusters]):
        my_members = mbk_means_labels == k
        cluster_center = mbk_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".",label =k )
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
            
        )
    ax.set_title("MiniBatchKMeans")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend()

    
    # Initialize the different array to all False
    different = mbk_means_labels == 4
    ax = fig.add_subplot(1, 3, 3)
    
    for k in range(n_clusters):
        different += (k_means_labels == k) != (mbk_means_labels == k)
    
    identical = np.logical_not(different)
    ax.plot(X[identical, 0], X[identical, 1], "w", markerfacecolor="#bbbbbb", marker=".")
    ax.plot(X[different, 0], X[different, 1], "w", markerfacecolor="m", marker=".")
    ax.set_title("Difference")
    ax.set_xticks(())
    ax.set_yticks(())
    plt.legend()
    plt.show()

    return distance_by_each_label
    