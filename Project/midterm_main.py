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

st.set_option('deprecation.showPyplotGlobalUse', False)
PROCESSED_DATA_DIR = './Project/data'
SVD_FEAT_MERGE_PATH = f'{PROCESSED_DATA_DIR}/merge_svd512.h5ad'
GAM_FEAT_MERGE_PATH = f'{PROCESSED_DATA_DIR}/merge_gam_w_target_1115.h5ad'
EQTL_FEAT_MERGE_PATH =  f'{PROCESSED_DATA_DIR}/merge_eqtl_max_1115.h5ad'
ENH_FEAT_MERGE_PATH =  f'{PROCESSED_DATA_DIR}/merge_enh_max_1115.h5ad'
TARGET_PATH = f'{PROCESSED_DATA_DIR}/merge_target_1115.h5ad'
COMMON_GENE_PATH = f'{PROCESSED_DATA_DIR}/common_genes_1115.npy'
feature_path_dict = {'svd': SVD_FEAT_MERGE_PATH, 'gam':GAM_FEAT_MERGE_PATH, 'eqtl': EQTL_FEAT_MERGE_PATH, 'enh': ENH_FEAT_MERGE_PATH, 'tgt': TARGET_PATH}
THRESHOLD = 5
with open(f'{PROCESSED_DATA_DIR}/mapping.json') as f:
    mapping = json.load(f)
HCG = ['ENSG00000069966', 'ENSG00000141452', 'ENSG00000142686', 'ENSG00000154122', 'ENSG00000158805', 'ENSG00000173917', 'ENSG00000177409', 'ENSG00000196187']

@st.cache_data
def load_data(feat):
    adata = ad.read_h5ad(feature_path_dict[feat])
    adata.obs = adata.obs.astype('str')
    idx = np.random.permutation(adata.shape[0])
    return adata[idx[:1000]]

@st.cache_data
def get_corr_data_feat():
    corr_data_feat = {}
    for i, feat in enumerate(['gam', 'enh', 'eqtl']):
        src = load_data(feat).X
        means1 = np.mean(src, axis=0)[None, :]
        means2 = np.mean(tgt, axis=0)[None, :]
        stddevs1 = np.std(src, axis=0)[None, :]
        stddevs2 = np.std(tgt, axis=0)[None, :]
        
        # Calculate the correlation coefficients in a vectorized manner
        numerator = np.sum((src - means1) * (tgt - means2), axis=0)
        denominator = (src.shape[0] - 1) * stddevs1 * stddevs2  # "n-1" used for a standard unbiased estimator
        correlations = numerator / denominator
        corr_data_feat[feat] = correlations
    return corr_data_feat

@st.cache_data
def get_corr_data_feat_nz():
    corr_data_feat_nz = {}
    for i, feat in enumerate(['gam', 'enh', 'eqtl']):
        correlations = []
        src = load_data(feat).X
        for j in range(src.shape[1]):
            src_temp = src[:, j]
            tgt_temp = tgt[:, j]
            src_temp, tgt_temp = src_temp[(src_temp>0) & (tgt_temp > 0)], tgt_temp[(src_temp>0) & (tgt_temp > 0)]
            if src_temp.shape[0]>THRESHOLD:
                correlations.append(pearsonr(src_temp, tgt_temp)[0])
        corr_data_feat_nz[feat] = correlations
    return corr_data_feat_nz
    
@st.cache_data
def get_corr_data():
    corr_data = {}
    for feat in ['gam', 'enh', 'eqtl']:
        corr_data[feat] = {}
        for factor in ['day', 'donor', 'cell_type']:
            corr_data[feat][factor] = {}
            options = ['NeuP', 'EryP', 'MasP', 'HSC'] if factor == 'cell_type' else load_data('gam').obs[factor].unique()
            for cat in options:
                src = load_data(feat)
                tgt2 = tgt[src.obs[factor]==cat]
                src = src[src.obs[factor]==cat].X
                corr_data[feat][factor][cat] = []
                for j in range(src.shape[1]):
                    src_temp = src[:, j]
                    tgt_temp = tgt2[:, j]
                    src_temp, tgt_temp = src_temp[(src_temp>0) & (tgt_temp > 0)], tgt_temp[(src_temp>0) & (tgt_temp > 0)]
                    if src_temp.shape[0]>THRESHOLD:
                        corr_data[feat][factor][cat].append(pearsonr(src_temp, tgt_temp)[0])
    return corr_data

tgt = np.asarray(load_data('tgt').X.todense())
corr_data = get_corr_data()
corr_data_feat = get_corr_data_feat()
corr_data_feat_nz = get_corr_data_feat_nz()

st.title("Predicting Gene Expression from Chromatin Openness: Background and Challenges")
st.subheader("Author: Hongzhi Wen")
st.markdown('''In this project, we explore a scenario where we measure the **openness of cromatin** and try to predict **gene expression levels** in **individual cells**.  Essentially this is a multivariate regression task.  
The purpose of this project is to demonstrate the ***challenges*** when deploying machine learning models to predict transcriptomics from epigenomics in single cells, including:
* High Dimensionality
* Sparsity
* Batch effect

In this, we will focus more on **batch effect**. *WARNING: Due to the limited computation resource for the online version, the data have been subsetted. Some observations might be different from original notes.*''')


st.header("Background")
with st.container():
    st.subheader("High Dimensionality and Sparsity Issue")
    st.markdown("""In this section, we present an overview of the data and highlight the **sparsity issue**. 
    The input data in this project is obtained from *ATAC-seq* technique, and the target data is measured by *scRNA-seq* technology. The dataset is produced by *10X multiome* platform, where the input and target are jointly measured for each single cell.    
    As shown in the heatmap plot before, the data can be presented as a data matrix. Each row represents a cell, each column represents a feature (either a `gene` or a `region` on chromatin). 
    For the target data, each feature is a `gene`, and values in the matrix represent the expression level of the `gene` in each cell.     
    Here we subset 56 most highly-variable genes and 10,000 cells from the target data. Note that the original dataset contains more than 100,000 cells and 22,000 genes, so it's extremly **high-dimensional**.""")

    adata = load_data('tgt')
    st.pyplot(sc.pl.heatmap(adata, adata.var.index[adata.var['highly_variable']], groupby='cell_type', cmap='viridis', dendrogram=True))
    st.markdown("""The heatmap above is the *target data*. We will not plot the raw *input data* because there are more than **100,000 features** in the input ATAC-seq data (each feature represents a certain region on chromatin, and the values indicate the openness).
        Instead, we conduct **preprocessing methods**, and generate four sets of features from the original features.  
        To further highlight the **sparsity issue**, we provide a histogram of values in the data matrix (note that the count has been log transformed).""")
    with open(f'{PROCESSED_DATA_DIR}/hist.json') as f:
        fig, axs = plt.subplots(ncols = 2)
        ts = {'input': 'scATAC-seq', 'target': 'scRNA-seq'}
        hist = json.load(f)
        for k in hist:
            hist[k][0] = np.log(hist[k][0]) / np.log(10)
        for i, k in enumerate(hist):
            axs[i].bar(range(len(hist[k][0])), hist[k][0], width=1, edgecolor='black', align='edge')
            axs[i].set_xticks(range(len(hist[k][0])), [f"{edge:.0e}" for edge in hist[k][1][:-1]], rotation=45)
            axs[i].set_xlabel('Value Range')
            axs[i].set_ylabel('Count (log10)')
            axs[i].set_title(f'Histogram of {ts[k]} Data')
        st.pyplot(fig)

with st.container():
    st.subheader("Batch Effect")
    st.markdown("""From the heatmap, it is still not very clear how cells distinguish from each other from the perspective of functionality and cell types. Now, we project the original high-dimeniosnal features into 2 dimtional space via `umap` algorithm. 
        In the figure below, each point stands for a cell.""")
    st.pyplot(sc.pl.umap(adata, color=['cell_type']))

    st.markdown("""From the visualization, we can see clusters that are consistent with ground-truth cell type annotation. However, it is still noisy. One underlying issue is the `batch effect`, where the experiments, donors and other conditions bring additional variables, which can largely affect the prediction and other analysis. Let's take a look:""")
    st.pyplot(sc.pl.umap(adata, color=['day', 'donor']))
    st.markdown("""In this new visualization, we color the umap visualization by `day` and `donor`. `day` and `donor` indicate that the cells are collected from different days and donors, which might distort the true biological signals (e.g., cell types).
        From the visualization, cells from different donors are mixed together, which means `donor` does not bring severe batch effect. In contrast, we can clearly observe the differences between cells from different `day`. It is not clear whether this distinction comes from true biology signals or technical noise.""")

st.header("Understanding Batch Effect from Distribution Shift")
st.markdown("""Let's go back to our topic, which is to *predict gene expression levels from chromatin openness*. **Batch effect** becomes an significant challenge when training a machine learning model, because the data do not follow the basic `i.i.d` (a.k.a, independent and identically distributed) assumption of most machine learning algorithms.
    In machine learning research, this is a well-known problem called `out-of-distribution (OOD)`. Imagine that when the test data are collected from a different `day` or `donor`, it might have a completely different distributions from the training data.
    From a distribution shift perspective, the potential problem of OODs can be broken down into two directions: `covariate shift` and `concept shift`.   
    Formally,  When training a model on an input $X \\in \\mathcal{X}$ (i.e., the covariate variable) to predict an output $Y \\in \\mathcal{Y}$, the joint distribution can be denoted as $P(Y, X)$ or $P(Y \mid X) P(X)$. In this context, the `covariate shift` refers to the distribution shift in input variables between training and test data. 
    i.e., $P_{tr}(X) ≠ P_{te}(X)$. 
    On the other hand, `concept shift` depicts the shift in conditional distribution $P(Y \mid X)$ between training and test data, i.e., $P_{tr}(Y \mid X) ≠  P_{te}(Y \mid X)$.""")

with st.container():
    st.subheader("Covariate Shift and Input Batch Effect")
    st.markdown("""In this subsection, we aim to study the potential covariate shift in the data via exploring batch effect in input features. 
        We produced four sets of features from the original raw count of scATAC-seq data:
    * Truncated SVD, or `svd`. These are top 512 principle somponents of the original data.
    * Gene activity matrix, or `gam`, a **biology-informed feature**. These features contain 1115 pesudo expression levels that correspond to specific genes.
    * Expression quantitative trait loci (eQTL), or `eqtl`, a **biology-informed feature**. These features aggregate potential genomic factors that correspond to 1115 specific genes.
    * Enhancers, or 'enh', a **biology-informed feature**. These features summarize enhancer openesses that correspond to 1115 specific genes.
        """)

    option1_feat = st.selectbox("Which features would you like to probe?", ("svd", "gam", "eqtl", "enh"))
    adata = load_data(option1_feat)
    st.pyplot(sc.pl.umap(adata, color=['cell_type', 'day', 'donor']))
    option1_factor = st.selectbox("For the feature you selected, select a major cell type to zoom in:", ('HSC', 'NeuP', 'MasP', 'EryP'))
    adata_ct = adata[adata.obs['cell_type'] == option1_factor]
    st.pyplot(sc.pl.umap(adata_ct, color=['day', 'donor']))

    st.markdown("""From the visualization, we observed that:
        * `svd` encodes clearest signals from `cell types`.
        * `svd` preserves batch effect for `day` the most.
        * Cells of different `cell types` are associated with the `day` effects to varying degrees.
        """)

with st.container():
    st.subheader("Concept Shift and Feature-target Correlation")
    st.markdown("""In this subsection, we aim to study the potential concept shift in the data via exploring correlation between input features and targets. 
        We first provide an overview via a box plot of correlations between targets and corresponding features:
        """)

    fig, axs = plt.subplots(ncols = 3, sharey=True)
    for i, feat in enumerate(['gam', 'enh', 'eqtl']):
        correlations = corr_data_feat[feat]
        sns.boxplot(correlations, ax=axs[i])
        axs[i].set_title(f'{feat}')
    axs[0].set_ylabel('Pearson Correlation')
    st.pyplot(fig)

with st.container():
    st.markdown("""We see that the correlation between the feature and targets are very close to $0$. One potential reason is the **random dropout and sparsity**. 
        Let's improve the correlation calculation by removing zero values. Specifically, for each pair of feature and target, we drop the datapoints with any missing values. When the remaining data are more than 10 cells, we calculate the correlations.
        """)
    fig, axs = plt.subplots(ncols = 3, sharey=True)
    for i, feat in enumerate(['gam', 'enh', 'eqtl']):
        correlations = get_corr_data_feat_nz[feat]
        sns.boxplot(correlations, ax=axs[i])
        axs[i].set_title(f'{feat} ({len(correlations)}/1115)')
    axs[0].set_ylabel('Pearson Correlation')
    st.pyplot(fig)

with st.container():
    st.markdown("""We indeed observe a much higher correlation when zero values are rmoved. Another potential issue is that the conditional distribution between features and targets may vary among batches.
        To investigate this issue, we break down the correlation w.r.t different factors:""")
    col1, col2= st.columns(2)
    with col1:
        option2_factor = st.selectbox("Select a factor:", ("day", "donor", "cell_type"))
    with col2:
        options = ['NeuP', 'EryP', 'MasP', 'HSC'] if option2_factor=='cell_type' else load_data('gam').obs[option2_factor].unique()
        option2_feat = st.multiselect("Categories:", options, options)
    fig, axs = plt.subplots(ncols=3, sharey=True)
    for i, feat in enumerate(['gam', 'enh', 'eqtl']):
        correlations = []
        for cat in option2_feat:
            correlations.append(corr_data[feat][option2_factor][cat])
        sns.boxplot(correlations, ax=axs[i])
        axs[i].set_xlabel(option2_factor)
        axs[i].set_xticklabels(option2_feat, fontdict={'fontsize': 'small'})
        axs[i].set_title(f'{feat}')
    axs[0].set_ylabel('Pearson Correlation')
    st.pyplot(fig)

with st.container():
    st.markdown("""We now clearly notice differences among `day` and `cell_type`. Especially on day 2, `enh` has a very high feature-target correlation. 
        Therefore, we decide to zoom in and focus on `enh`. Here we provide a list of highly correlated feature-target pairs on day 2, and see how these correlations vary in different days with a interactive scatter plot:""")

    optione3_feature = st.selectbox("Select a gene:", HCG, format_func=lambda x: mapping[x])

    enh_data = load_data('enh')#[:, optione3_feature].X[:, 0]e
    enh_data.obs = enh_data.obs.astype('str')
    tgt_data = np.asarray(load_data('tgt')[:, optione3_feature].X.todense())[:, 0]
    enh_data, tgt_data = enh_data[enh_data.obs['cell_type'].isin(['MasP', 'NeuP', 'HSC'])], tgt_data[enh_data.obs['cell_type'].isin(['MasP', 'NeuP', 'HSC'])] 
    df = pd.DataFrame({'x': enh_data[:, optione3_feature].X[:, 0], 
                       'y': tgt_data, 
                       'day': enh_data.obs['day'], 
                       'donor': enh_data.obs['donor'], 
                       'cell_type': enh_data.obs['cell_type'],
                       'nonzero': (enh_data[:, optione3_feature].X[:, 0]>0)&(tgt_data>0)}).iloc[:5000]

    points = alt.Chart(df, title=mapping[optione3_feature]).mark_circle().encode(x="x", y="y", color="day", tooltip=['donor','cell_type']).properties(
        width=600, 
        height=600,
    )
    line = points.transform_filter(alt.datum.nonzero == True).transform_regression(
        'x', 'y', groupby=['day']
    ).mark_line()
    st.altair_chart((points+line).interactive())

