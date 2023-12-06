import streamlit as st

st.header("Conclusion")
st.markdown("""
This webapp studies a interesting problem, which is to predict **gene expression levels** from **cromatin openness** using regression models.
We explored the sparsity and batch effect in the data, and found that `svd` is a better preprocessing way than other biology-informed feature selection methods.
In addition, we compared different regression models, and found that xxx is the most suitable one among 4 selected methods.
We hope this webapp can give you a preliminary understanding about the single-cell data, and inspire the development of more advanced methods to uncover the transcription process, the mystical principles governing the codes of life.
""")

for k, v in st.session_state.to_dict().items():
   st.session_state[k] = v
