import pandas as pd
import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path
import pickle
import argparse
import scanpy as sc
import streamlit as st
import altair as alt
import json

for k, v in st.session_state.to_dict().items():
   st.session_state[k] = v
   
st.header("Clustering Analysis and Batch Effect Visualization")

st.subheader("Dimension Reduction and Visualization")
st.markdown("""From the raw data, it is very hard to establish identity of each cell, due to the high dimension and the sparsity of the data. 
In single-cell biology, people are particularly interest in the heterogeneous cell states, from the perspective of functionality and cell types. In light of this, we project the original high-dimeniosnal features into lower dimension space and conduct clustering analysis.
Specifically, we leverage a famous dimensionality reduction algorithm called `umap`. The high dimensional data are thus projected into 2 dimensional space. 
Compared to `truncated SVD` and `PCA`, `umap` can better preserve the topological structure of the data. 
In the figure below, we present a `umap` visualization, where each point stands for a cell.""")
# print(st.session_state['tgt'])
# tab_kmeans, tab_leiden, tab_louvain, tab_celltype = st.tabs(['kmeans', 'leiden', 'louvain', 'cell type'])
# with tab_kmeans:
#    st.pyplot(sc.pl.umap(st.session_state['tgt'], color=['kmeans'], title='KMeans'))

# with tab_leiden:
#    st.pyplot(sc.pl.umap(st.session_state['tgt'], color=['cell_type'], title='Leiden'))

# with tab_louvain:
#    st.pyplot(sc.pl.umap(st.session_state['tgt'], color=['cell_type'], title='Louvain'))

# with tab_celltype:
#    st.pyplot(sc.pl.umap(st.session_state['tgt'], color=['cell_type'], title='Cell types'))
st.pyplot(sc.pl.umap(st.session_state['tgt'], color=['kmeans', 'louvain', 'leiden', 'cell_type'], ncols=2))

st.markdown("""From the visualization, we can see cells indeed present some clustering structures in the low-dimensional space. 
   To further identify these clusters, we apply various clustering algorithms and color each cell with the cluster labels. `kmeans`, `louvain` and `leiden` refer two three popular clustering methods, while `cell type` correspond to manually curated labels that can be considered as ground-truth label.
   Compared to `kmeans`, `leiden` and `louvain` are more advanced and popular clustering algorithm in single-cell analysis, and they indeed generate more consistent labels with ground-truth cell type annotation in our visualization. 
   
   However, we also notice that the clustering and visualizations are not perfectly aligned with the cell type labels, especially of `HSC` cells. This is because we overlooked a severe issue, **batch effect**.""")

st.subheader("Batch Effect")
st.markdown("""Batch effect refers to the variation brought by the experiments, donors and other conditions, instead of the true biological signals. This can largely affect performance of any machine learning algorithms. We can visualize the batch effect in this way:""")
st.pyplot(sc.pl.umap(st.session_state['tgt'], color=['day', 'donor']))
st.markdown("""In this new visualization, we color the umap visualization by `day` and `donor`. `day` and `donor` indicate that the cells are collected from different days and donors, which might distort the true biological signals (e.g., cell types).
  From the visualization, cells from different donors are mixed together, which means `donor` does not bring severe batch effect. In contrast, we can clearly observe the differences between cells from different `day`. It is not clear whether this distinction comes from true biology signals or technical noise.""")

st.subheader("How could batch effect affect regression?")
st.markdown("""Let's go back to our topic, which is to *predict gene expression levels from chromatin openness*. **Batch effect** becomes an significant challenge when training a machine learning model, because the data do not follow the basic `i.i.d` (a.k.a, independent and identically distributed) assumption of most machine learning algorithms.
In machine learning research, this is a well-known problem called `out-of-distribution (OOD)`. Imagine that when the test data are collected from a different `day` or `donor`, it might have a completely different distributions from the training data.
Formally,  when training a model on an input $X \\in \\mathcal{X}$ (i.e., the covariate variable) to predict an output $Y \\in \\mathcal{Y}$, the joint distribution can be denoted as $P(Y, X)$ or $P(Y \\mid X) P(X)$. 
This distribution shift in input variables between training and test data, i.e., $P_{tr}(X) ≠ P_{te}(X)$, is a common type of distribution shift, known as `covariate shift`.""")

for k, v in st.session_state.to_dict().items():
   st.session_state[k] = v

# st.subheader("Concept Shift and Feature-target Correlation")
# with st.container():
#     st.markdown("""In this subsection, we aim to study the potential concept shift in the data via exploring correlation between input features and targets. 
#         We first provide an overview via a violin plot of correlations between targets and corresponding features:
#         """)

#     fig, axs = plt.subplots(ncols = 3, sharey=True)
#     for i, feat in enumerate(['gam', 'enh', 'eqtl']):
#         src = load_data(feat).X
#         means1 = np.mean(src, axis=0)[None, :]
#         means2 = np.mean(tgt, axis=0)[None, :]
#         stddevs1 = np.std(src, axis=0)[None, :]
#         stddevs2 = np.std(tgt, axis=0)[None, :]
        
#         # Calculate the correlation coefficients in a vectorized manner
#         numerator = np.sum((src - means1) * (tgt - means2), axis=0)
#         denominator = (src.shape[0] - 1) * stddevs1 * stddevs2  # "n-1" used for a standard unbiased estimator
#         correlations = numerator / denominator
#         print(correlations[0].shape)
#         sns.violinplot(correlations[0], ax=axs[i])
#         axs[i].set_title(f'{feat}')
#     axs[0].set_ylabel('Pearson Correlation')
#     st.pyplot(fig)

# with st.container():
#     st.markdown("""We see that the correlation between the feature and targets are very close to $0$. One potential reason is the **random dropout and sparsity**. 
#         Let's improve the correlation calculation by removing zero values. Specifically, for each pair of feature and target, we drop the datapoints with any missing values. When the remaining data are more than 10 cells, we calculate the correlations.
#         """)
#     fig, axs = plt.subplots(ncols = 3, sharey=True)
#     for i, feat in enumerate(['gam', 'enh', 'eqtl']):
#         correlations = []
#         src = load_data(feat).X
#         for j in range(src.shape[1]):
#             src_temp = src[:, j]
#             tgt_temp = tgt[:, j]
#             src_temp, tgt_temp = src_temp[(src_temp>0) & (tgt_temp > 0)], tgt_temp[(src_temp>0) & (tgt_temp > 0)]
#             if src_temp.shape[0]>10:
#                 correlations.append(spearmanr(src_temp, tgt_temp)[0])
#             # else:
#             #     correlations.append(0)
#         sns.violinplot(correlations, ax=axs[i])
#         axs[i].set_title(f'{feat} ({len(correlations)}/1115)')
#     axs[0].set_ylabel('Pearson Correlation')
#     st.pyplot(fig)


for k, v in st.session_state.to_dict().items():
   st.session_state[k] = v