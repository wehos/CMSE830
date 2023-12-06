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

st.header("Data Navigation")
st.markdown(f"""
In this section, we present an overview of the data. The dataset we explore contains **105,933 cells**, produced by *10X multiome* platform.
The data of each cell consists of two modalities, `ATAC-seq` (chromatin/DNA/epigenomic feature) and `scRNA-seq` (RNA/transcriptomic feature). `ATAC-seq` data has **228942 features**, each refer to a certain region on the chromatin. 
`scRNA-seq` data has **23,418 features**, each refers to a specific type of gene. The data come in the form of **Data Matrix**, each row represents a cell and each column represents a feature.
Values in the matrix correspond to the abundance of certain signal in a cell (i.e., openness of the chromatin regions and expression level of genes.)
""")
with st.container():
    st.subheader("Data Visualization")
    st.markdown("""
    Since the data has extremely **high dimension**, we visualize a subset of the data in form of hetamap plot.  
    Specifically, we randomly select 1000 cells and 100 features from selected modality.""")

    tab_atac, tab_rna = st.tabs(["scATAC-seq", "scRNA-seq"])
    with tab_atac:
        st.pyplot(sc.pl.heatmap(st.session_state['src'], st.session_state['src'].var.index, groupby='cell_type', cmap='viridis', dendrogram=True))
    with tab_rna:
        st.pyplot(sc.pl.heatmap(st.session_state['tgt'][:1000], st.session_state['tgt'].var.index[:100], groupby='cell_type', cmap='viridis', dendrogram=True))
    
    st.markdown("""
        As shown above, most positions in the `atac-seq` heatmap plot are very dark (they are exactly 0). This issue is known as "**data sparsity**". 

        On the other hand, the `rna-seq` heatmap (in the second tab) contains a lot of light vertical bars. These bars correspond to housekeeping genes, which are expressed in most of the cells. 
        There are also highly variable genes and rare genes, which might be used as biological markers for cell state identification.
        
        To further highlight the **data sparsity**, we provide a histogram of values in the data matrix (note that the count has been log transformed). Now we can clearly see that more than $97\\%$ of the `atac-seq` data and around $85\\%$ of `rna-seq` data are zero.""")
    
    fig, axs = plt.subplots(ncols = 2)
    ts = {'input': 'scATAC-seq', 'target': 'scRNA-seq'}
    hist = st.session_state['hist']
    for i, k in enumerate(hist):
        axs[i].bar(range(len(hist[k][0])), hist[k][0], width=1, edgecolor='black', align='edge')
        axs[i].set_xticks(range(len(hist[k][0])), [f"{edge:.0e}" for edge in hist[k][1][:-1]], rotation=45)
        axs[i].set_xlabel('Value Range')
        if i==0:
            axs[i].set_ylabel('Count (log10)')
        axs[i].set_title(f'Histogram of {ts[k]} Data')
    st.pyplot(fig)


