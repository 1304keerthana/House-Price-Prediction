import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("🏠 House Price Prediction")

file = st.file_uploader("Upload Boston Housing CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write(df.head())

    features = ['RM', 'LSTAT', 'PTRATIO', 'INDUS', 'NOX', 'AGE']
    target = 'MEDV'

    df = df[features + [target]].dropna()

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("📊 Performance")
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R²:", r2_score(y_test, y_pred))

    st.subheader("📈 Coefficients")
    st.write(pd.DataFrame(model.coef_, features, columns=["Impact"]))

    # Plot
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    st.pyplot(plt)
