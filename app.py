import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("ExplainMl")

st.write("This dashboard trains a ML model on your dataset and explains your prediction using SHAP")

uploaded_file=st.file_uploader("Upload CSV dataset",type=["csv"])

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    target= st.selectbox("Select the target columns",df.columns)

    x=df.drop(target,axis=1)
    y=df[target]

    if not pd.api.types.is_numeric_dtype(y):
        st.error(f"⚠️ Target column '{target}' is not numeric. Please select a numeric column for regression.")
        st.stop()

    X=pd.get_dummies(x)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    model=RandomForestRegressor()
    model.fit(X_train,y_train)

    st.subheader("Feature Importance")
    importance=model.feature_importances_
    feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp.plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Features")
    plt.tight_layout()
    st.pyplot(fig)
    explainer=shap.TreeExplainer(model)
    shap_values=explainer.shap_values(X_test, check_additivity=False)
    st.subheader("Global Model Explanation")
    shap.summary_plot(shap_values,X_test,show=False)
    fig2=plt.gcf()
    st.pyplot(fig2)

    st.subheader("Explain Individual Prediction")

    index=st.slider("select row",0,len(X_test)-1,0)

    sample=X_test.iloc[[index]]

    prediction=model.predict(sample)[0]
    st.write("Predicted Value:",prediction)

   
    shap.force_plot(
        explainer.expected_value,
        shap_values[index],
        sample,
        matplotlib=True,
        show=False,
    )
    fig3=plt.gcf()
    st.pyplot(fig3)
