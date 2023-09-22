import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.read_csv('data.csv')
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
columns_mean = [
    'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]
df_mean = pd.DataFrame(df, columns=columns_mean)

st.write("""# Wisconsin Breast Cancer Diagnostic dataset
    Different plot we presented in tabs:  
    * Distribution  
    * Boxplot  
    * Pair Plot  

    Have fun with data exploration!
    """)

sec1, sec2, sec3 = st.tabs(['Distribution', 'Boxplot', 'Pair Plot'])


########## Tab1 ###########
sec1.write("""
## Distribution Plot
Can you tell from visualizations which features are more informative for breast cancer diagnosis?
""")

selected = sec1.selectbox(
    'Which features are you interested in?',
    [d for d in df.columns if d!='diagnosis'], key='se1')

fig = sns.histplot(data=df, x=selected, hue='diagnosis', element='step')
# plt.title(selected)
sec1.pyplot()

########## Tab2 ###########
sec2.write("""
## Box Plot
Can you tell from visualizations which features contain outliers?
""")
selected2 = sec2.selectbox(
    'Which features are you interested in?',
    [d for d in df.columns if d!='diagnosis'], key='se2')

sns.boxplot(data=df, x=df[selected2], orient='h')
# plt.title(selected2)
sec2.pyplot()

########## Tab3 ###########
sec3.write("""
## Pair Plot
Can you tell which features are highly correlated and might be redundant? Which features are independent?
""")
selected3 = sec3.multiselect(
    'Which features are you interested in?',
    [d for d in columns_mean if d!='diagnosis'],
    ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'], key='se3')

sns.pairplot(df_mean[['diagnosis']+selected3],
             hue="diagnosis",
             palette=["blue", "green"],
             markers=["o", "s"])
# plt.title(selected3)
sec3.pyplot()

