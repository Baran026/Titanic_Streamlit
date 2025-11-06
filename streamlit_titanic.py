import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic Analysis", layout="wide")
st.title("Titanic Overleving Analyse 🚢")

# -----------------------------
# TAB 1: Data
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data", "Preprocessing", "Visualisaties", "Model", "Download"]
)

with tab1:
    st.header("Upload je datasets")
    train_file = st.file_uploader("Upload train.csv", type="csv")
    test_file = st.file_uploader("Upload test.csv", type="csv")

    if train_file is not None:
        train = pd.read_csv(train_file)
        st.subheader("Train dataset")
        st.dataframe(train.head())

    if test_file is not None:
        test = pd.read_csv(test_file)
        test["Survived"] = 0
        st.subheader("Test dataset")
        st.dataframe(test.head())

# -----------------------------
# TAB 2: Preprocessing
# -----------------------------
with tab2:
    st.header("Preprocessing & Feature Engineering")
    if 'train' in locals() and 'test' in locals():
        # Vul missende waarden
        for df in [train, test]:
            df["Age"] = df["Age"].fillna(df["Age"].median())
            df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
            df["Fare"] = df["Fare"].fillna(df["Fare"].median())
            
            # Feature engineering
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['FamilyClass'] = pd.cut(
                df["FamilySize"],
                bins=[0, 1, 4, 6, 11],
                labels=["Solo", "Small", "Medium", "Large"]
            )
            df["AgeGroup"] = pd.cut(
                df["Age"],
                bins=[0, 12, 18, 60, 120],
                labels=["Child", "Teen", "Adult", "Senior"]
            )
            df["Prizeclass"] = pd.cut(
                df["Fare"],
                bins=[0, 50, 100, 200, 300, train["Fare"].max()],
                labels=["0-50", "50-100", "101-200", "201-300", "300+"],
                include_lowest=True
            )
        
        st.write("Missende waarden zijn ingevuld en nieuwe features zijn toegevoegd:")
        st.dataframe(train.head())

# -----------------------------
# TAB 3: Visualisaties
# -----------------------------
with tab3:
    st.header("Visualisaties")
    if 'train' in locals():
        st.subheader("Scatterplot: Age vs Fare (Survived)")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train, ax=ax)
        ax.set_title("Ticketprijs per leeftijd en overleving")
        st.pyplot(fig)

        st.subheader("Barplots van Survival per feature")
        features_to_plot = ["Pclass", "Sex", "Embarked", "FamilyClass", "AgeGroup", "Prizeclass"]
        for f in features_to_plot:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x=f, y="Survived", data=train, ax=ax)
            ax.set_ylabel("Survived %")
            st.pyplot(fig)

# -----------------------------
# TAB 4: Model / Predictions
# -----------------------------
with tab4:
    st.header("Voorspellingen maken")
    if 'train' in locals() and 'test' in locals():
        Features = ["Pclass", "Sex", "FamilyClass", "AgeGroup", "Prizeclass", "Embarked"]
        p_global = train["Survived"].mean()
        alpha = 50
        beta = 0.05
        Kansen = {}
        for feature in Features:
            grouped = train.groupby(feature)["Survived"].agg(["mean", "count"])
            grouped["smoothed"] = (grouped["mean"] * grouped["count"] + alpha * p_global) / (grouped["count"] + alpha)
            Kansen[feature] = (grouped["smoothed"] * 100).to_dict()

        for f in Features:
            test[f"{f}_Kans"] = test[f].map(Kansen[f]).astype(float)

        kans_cols = [f"{f}_Kans" for f in Features]

        def calc_weighted_avg(row, kans_cols):
            weights = []
            values = []
            for col in kans_cols:
                val = row[col]
                if pd.isna(val):
                    continue
                w = 1 + beta * abs(val - 50)
                values.append(val * w)
                weights.append(w)
            return sum(values) / sum(weights) if weights else np.nan

        test["AVG_Kans"] = test.apply(lambda r: calc_weighted_avg(r, kans_cols), axis=1)
        test["Survived"] = (test["AVG_Kans"] > 39).astype(int)

        st.write("Globale overlevingskans in train:", train["Survived"].mean())
        st.write("Voorspelde overlevingskans in test:", test["Survived"].mean())

# -----------------------------
# TAB 5: Download
# -----------------------------
with tab5:
    st.header("Download submission.csv")
    if 'test' in locals():
        submission = pd.DataFrame({
            "PassengerId": test["PassengerId"],
            "Survived": test["Survived"]
        })
        csv = submission.to_csv(index=False).encode("utf-8")
        st.download_button("Download submission.csv", csv, "submission.csv", "text/csv")
