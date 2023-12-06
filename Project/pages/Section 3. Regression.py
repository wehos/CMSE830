import pandas as pd
import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, PredictionErrorDisplay, mean_squared_error
from sklearn.model_selection import cross_val_predict
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

st.header("Feature Selection and Regression Models")
st.subheader("Batch Effect in Different Feature Selection Strategies")
with st.container():
    st.markdown("""Feature selection is a common preprocessing step when dealing with high-dimensional single-cell data. 
        In this section, we start from evaluating the batch effect in different feature selection strategies. 
        Specifically, we applied four feature selection strategy on the original raw count of `scATAC-seq` data:
* **Truncated SVD**, or `svd`. These are top 512 principle somponents of the original data.
* **Gene activity matrix**, or `gam`, a **biology-informed feature**. These features contain 1115 pesudo expression levels that correspond to specific genes.
* **Expression quantitative trait loci (eQTL)**, or `eqtl`, a **biology-informed feature**. These features aggregate potential genomic factors that correspond to 1115 specific genes.
* **Enhancers**, or `enh`, a **biology-informed feature**. These features summarize enhancer opennesses that correspond to 1115 specific genes.
        """)

    option1_feat = st.selectbox("Which features would you like to probe?", ("svd", "gam", "eqtl", "enh"))
    adata = st.session_state[option1_feat]
    st.pyplot(sc.pl.umap(adata, color=['cell_type', 'day', 'donor']))
    option1_factor = st.selectbox("For the feature you selected, select a major cell type to zoom in:", ('HSC', 'NeuP', 'MasP', 'EryP'))
    adata_ct = adata[adata.obs['cell_type'] == option1_factor]
    st.pyplot(sc.pl.umap(adata_ct, color=['day', 'donor']))

    st.markdown("""From the visualization, we observed that:
*  `svd` encodes clearest biological signals from cell types.
*  `svd` preserves batch effect for `day` the most.
*  Cells of different `cell types` are associated with the `day` effects to varying degrees.
        """)

st.subheader("Comparing Different Feature Selection Strategies")
st.markdown("""
    Based on the aforementioned observation, we may assume that `svd` would be the best feature selection strategy out of the four strategies. 
    Now we can verify this through a simple linear regression model. 
    We leverage a 5-fold cross-validation to evaluate the model on different features, and visualize it in a scatter plot.
""")


y = np.asarray(st.session_state['tgt'].X.todense())
tabs = st.tabs(["svd", "gam", "eqtl", "enh"])
for i, feat in enumerate(["svd", "gam", "eqtl", "enh"]):
    with tabs[i]:
        y_pred = st.session_state[feat].obsm['pred']
        PredictionErrorDisplay.from_predictions(
            y,
            y_pred,
            kind="actual_vs_predicted",
            subsample=100,
            # ax=axs[0],
            random_state=0,
        )
        plt.title('MSE %.4f'%(mean_squared_error(y, y_pred)))
        st.pyplot(plt.show())

st.markdown("""
    Both the visualization and MSE loss confirm our assumption that the `svd` feature gives us best performance.
""")

with st.container():
    st.subheader("Comparing Different Regression Models")
    st.markdown("""
        Now we can continue to verify which regression method is more suitable for the prediction with `svd` features.
        Since the model may suffer from distribution shift, we need a particularly robust regression model. 
        In light of this, we select a set of models that has less complexity. They are:
    * Kernel ridge regression (KRR)
    * Lasso regression
    * Elastic net
    * Support vector regression (SVR)
    """)
    option = st.selectbox(
        'Which model are you interested in?',
        ('KRR', 'Lasso', 'Elastic', 'SVR'))
    
    y_pred = st.session_state[f'svd_pred_{model}']

    PredictionErrorDisplay.from_predictions(
        y,
        y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        random_state=0,
    )
    plt.title('MSE %.4f'%(mean_squared_error(y, y_pred)))
    st.pyplot(plt.show())


for k, v in st.session_state.to_dict().items():
   st.session_state[k] = v

