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
st.markdown("""From the raw data, it is very hard to establish the identity of each cell, due to the high dimension and the sparsity of the data. 
In single-cell biology, people are particularly interested in the heterogeneous cell states, from the perspective of functionality and cell types. In light of this, we project the original high-dimensional features into lower-dimension space and conduct clustering analysis.
Specifically, we leverage a famous dimensionality reduction algorithm called `umap`. The high dimensional data are thus projected into 2-dimensional space. 
Compared to `truncated SVD` and `PCA`, `umap` can better preserve the topological structure of the data. 
In the figure below, we present a `umap` visualization, where each point stands for a cell.""")

st.pyplot(sc.pl.umap(st.session_state['tgt'], color=['kmeans', 'louvain', 'leiden', 'cell_type'], ncols=2))

st.markdown("""From the visualization, we can see cells indeed present some clustering structures in the low-dimensional space. 
   To further identify these clusters, we apply various clustering algorithms and color each cell with the cluster labels. `kmeans`, `louvain` and `leiden` refer to three popular clustering methods, while `cell type` correspond to manually curated labels that can be considered as a ground-truth label.
   Compared to `kmeans`, `leiden` and `louvain` are more advanced and popular clustering algorithms in single-cell analysis, and they indeed generate more consistent labels with ground-truth cell type annotation in our visualization. 
   
   However, we also notice that the clustering and visualizations are not perfectly aligned with the cell type labels, especially of `HSC` cells. This is because we overlooked a severe issue, **batch effect**.""")

st.subheader("Batch Effect")
st.markdown("""Batch effect refers to the variation brought by the experiments, donors and other conditions, instead of the true biological signals. This can largely affect performance of any machine learning algorithms. We can visualize the batch effect in this way:""")
st.pyplot(sc.pl.umap(st.session_state['tgt'], color=['day', 'donor']))
st.markdown("""In this new visualization, we color the umap visualization by `day` and `donor`. `day` and `donor` indicates that the cells are collected from different days and donors, which might distort the true biological signals (e.g., cell types).
  From the visualization, cells from different donors are mixed together, which means `donor` does not bring a severe batch effect. In contrast, we can clearly observe the differences between cells from different `day`. It is not clear whether this distinction comes from true biology signals or technical noise.""")

st.subheader("How could batch effect affect regression?")
st.markdown("""Let's go back to our topic, which is to *predict gene expression levels from chromatin openness*. **Batch effect** becomes a significant challenge when training a machine learning model, because the data do not follow the basic `i.i.d` (a.k.a, independent and identically distributed) assumption of most machine learning algorithms.
In machine learning research, this is a well-known problem called `out-of-distribution (OOD)`. Imagine that when the test data are collected from a different `day` or `donor`, it might have a completely different distribution from the training data.
Formally,  when training a model on an input $X \\in \\mathcal{X}$ (i.e., the covariate variable) to predict an output $Y \\in \\mathcal{Y}$, the joint distribution can be denoted as $P(Y, X)$ or $P(Y \\mid X) P(X)$. 
This distribution shift in input variables between training and test data, i.e., $P_{tr}(X) ≠ P_{te}(X)$, is a common type of distribution shift, known as `covariate shift`.""")

for k, v in st.session_state.to_dict().items():
   st.session_state[k] = v
