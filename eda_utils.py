import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def show_eda(df):
    st.subheader("📊 EDA Visualizations")
    
    st.write("**Age Distribution**")
    fig = sns.histplot(df['age'], kde=True)
    st.pyplot(fig.figure)
    
    st.write("**Charges vs Age (Colored by Smoker)**")
    fig = sns.scatterplot(x='age', y='charges', hue='smoker', data=df)
    st.pyplot(fig.figure)

    st.write("**Correlation Heatmap**")
    corr = df.corr(numeric_only=True)
    fig = sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(fig.figure)
