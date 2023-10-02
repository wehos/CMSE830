import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
st.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.read_csv('breast_cancer.csv')
df['Estrogen Status'] = df['Estrogen Status'].map(lambda x: 1 if x=='Positive' else 0)
df['Progesterone Status'] = df['Progesterone Status'].map(lambda x: 1 if x=='Positive' else 0)
df['Status'] = df['Status'].map(lambda x: 1 if x=='Alive' else 0)
features = df.drop('Status', axis=1).columns
num_feat = ['Age','Tumor Size','Estrogen Status', 'Progesterone Status', 'Regional Node Examined', 'Reginol Node Positive', 'Survival Months']
st.write("""# Breast_cancer from Kaggle
    
    //Get the data at https://www.kaggle.com/datasets/reihanenamdari/breast-cancer

    We present different visualizations in tabs. What you can tell from them:
    * Which features are more informative for breast cancer diagnosis?
    * Which features contain outliers?
    * Which features are highly correlated and might be redundant? Which features are independent?
    """)

sec1, sec2, sec3 = st.tabs(['Distribution', 'Boxplot', 'Correlation'])


########## Tab1 ###########
sec1.write("""
## Distribution Plot
Can you tell from visualizations which features are more informative for breast cancer diagnosis?
""")

selected = sec1.selectbox(
    'Which features are you interested in?',
    features, key='se1')

fig = sns.histplot(data=df, x=selected, hue='Status', element='step')
# plt.title(selected)
sec1.pyplot()

########## Tab2 ###########
sec2.write("""
## Box Plot
Can you tell from visualizations which features contain outliers?
""")
selected2 = sec2.selectbox(
    'Which features are you interested in?',
    num_feat, key='se2')

sns.boxplot(data=df, x=selected2, orient='h')
# plt.title(selected2)
sec2.pyplot()

########## Tab3 ###########
sec3.write("""
## Correlation Plot
Can you tell which features are highly correlated and might be redundant? Which features are independent?
""")
selected3 = sec3.multiselect(
    'Which features are you interested in?',
    num_feat,
    ['Age', 'Tumor Size', 'Estrogen Status', 'Progesterone Status'], key='se3')

corr=df[selected3 + ['Status']].corr()
# sns.pairplot(df[['Status']+selected3],
#              hue="diagnosis",
#              palette=["blue", "green"],
#              markers=["o", "s"])
# # plt.title(selected3)
# sec3.pyplot()
sec3.dataframe(corr.style.background_gradient(cmap='coolwarm').format(precision=3))

