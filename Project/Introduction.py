import pandas as pd
import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import cross_val_predict
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
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
import sklearn
from sklearn.cluster import KMeans

for k, v in st.session_state.to_dict().items():
   st.session_state[k] = v

st.title("Predicting Gene Expression from Chromatin Openness: Challenges and Opportunities")
st.subheader("Author: Hongzhi Wen")

PROCESSED_DATA_DIR = './Project/data'
SVD_FEAT_MERGE_PATH = f'{PROCESSED_DATA_DIR}/merge_svd512.h5ad'
GAM_FEAT_MERGE_PATH = f'{PROCESSED_DATA_DIR}/merge_gam_w_target_1115.h5ad'
EQTL_FEAT_MERGE_PATH =  f'{PROCESSED_DATA_DIR}/merge_eqtl_max_1115.h5ad'
ENH_FEAT_MERGE_PATH =  f'{PROCESSED_DATA_DIR}/merge_enh_max_1115.h5ad'
COMMON_GENE_PATH = f'{PROCESSED_DATA_DIR}/common_genes_1115.npy'
TARGET_PATH = f'{PROCESSED_DATA_DIR}/merge_target_1115.h5ad'
SOURCE_PATH = f'{PROCESSED_DATA_DIR}/atac_raw_subset.h5ad'
feature_path_dict = {'svd': SVD_FEAT_MERGE_PATH, 'gam':GAM_FEAT_MERGE_PATH, 'eqtl': EQTL_FEAT_MERGE_PATH, 'enh': ENH_FEAT_MERGE_PATH, 'tgt': TARGET_PATH, 'src': SOURCE_PATH}

st.set_option('deprecation.showPyplotGlobalUse', False)
@st.cache_resource
def load_data(feat):
    adata = ad.read_h5ad(feature_path_dict[feat])
    adata.obs = adata.obs.astype('str')
    if feat == 'tgt':
        kmeans = KMeans(n_clusters=7, random_state=0).fit(adata.obsm['X_pca']) 
        adata.obs['kmeans'] = kmeans.labels_.astype(str)
        sc.tl.leiden(adata, key_added = "leiden", resolution=0.5)
        sc.tl.louvain(adata, key_added = "louvain", resolution=0.5)
    elif feat != 'src':
        X = np.asarray(adata.X)
        y = np.asarray(load_data('tgt').X.todense())
        np.random.seed(0)
        lr = LinearRegression()
        adata.obsm['pred'] = cross_val_predict(lr, X, y, cv=5)
    return adata

@st.cache_resource
def get_svd_prediction(option):
    np.random.seed(0)
    X = np.asarray(st.session_state['svd'].X)
    y = np.asarray(st.session_state['tgt'].X.todense())#[:, st.session_state['tgt'].var['highly_variable']].X.todense())
    
    model = st.session_state['model_dict'][option]
    y_pred = cross_val_predict(model, X, y, cv=5)
    return y_pred

np.random.seed(0)
st.session_state['tgt'] = load_data('tgt')
st.session_state['src'] = load_data('src')
st.session_state['eqtl'] = load_data('eqtl')
st.session_state['gam'] = load_data('gam')
st.session_state['enh'] = load_data('enh')
st.session_state['svd'] = load_data('svd')
st.session_state['model_dict'] = {
    'KRR': KernelRidge(),
    'Lasso': Lasso(),
    'Elastic': ElasticNet(),
    'SVR': SVR(),
}
for option in ('KRR', 'Lasso', 'Elastic'):
    st.session_state[f'svd_pred_{option}'] = get_svd_prediction(option)
with open(f'{PROCESSED_DATA_DIR}/mapping.json') as f:
    st.session_state['mapping'] = json.load(f)
with open(f'{PROCESSED_DATA_DIR}/hist.json') as f:
    hist = json.load(f)
    for k in hist:
        hist[k][0] = np.log(hist[k][0]) / np.log(10)
st.session_state['hist'] = hist

st.markdown('''This webapp explores a scenario where we measure the **openness of cromatin** and try to predict **gene expression levels** in **individual cells**.
    Essentially this is a multivariate regression task, and is consistent with a biological process called `transcription`. 
    The reason to study this regression task is because an accurate regression model can reveal the underlying process of transcription.

We arrange this webapp into 3 sections, 
* Section 1. Data navigation
* Section 2. Clustering analysis and batch effect visualization
* Section 3. Feature selection and regression models

**In the first section**, we quickly introduce the data. Particularly, we will highlight two major challenges for analyzing single-cell data, i.e., **high dimensionality** and **sparsity**.

**In the second section**, we utilize dimensionality reduction and clustering methods to demonstrate the advanced charactersitic of single-cell data, especially **batch effect**. This will bring us deeper insight into the problem.

**In the final section**, we illustrate the deployment of regression models for predicting the expression of specific genes. Moreover, we compare the efficacy of various feature selection techniques and regression models. 
The results can inspire us to develop more sophisticated methods for forecasting gene expressions using epigenetic data, thereby unraveling the enigmatic principles governing the codes of life.

---

##### Short Bio of the Author
Hongzhi Wen is a third year Ph.D. student in Computer Science at Michigan State University, advised by Dr. Jiliang Tang.
His research focuses on applying graph neural networks and transformers to large-scale real-world datasets, especially single-cell analysis. He has published papers in top AI conferences, including ICLR, NeurIPS, KDD and CIKM.

''')

for k, v in st.session_state.to_dict().items():
   st.session_state[k] = v

