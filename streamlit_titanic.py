import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page config
st.set_page_config(page_title="Titanic Analysis", layout="wide")

# Basispad van het script
BASE_DIR = os.path.dirname(__file__)

# Data inladen
try:
    train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
except FileNotFoundError:
    st.error("train.csv of test.csv niet gevonden! Zorg dat ze in dezelfde folder staan als het script.")
    st.stop()

# Data preprocessing
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
train["Fare"] = train["Fare"].fillna(train["Fare"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Embarked"] = test["Embarked"].fillna(train["Embarked"].mode()[0])
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# Feature engineering
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['FamilyClass'] = pd.cut(train["FamilySize"], bins=[0, 1, 4, 6, 11], labels=["Solo", "Small", "Medium", "Large"])
train["AgeGroup"] = pd.cut(train["Age"], bins=[0, 12, 18, 60, 120], labels=["Child", "Teen", "Adult", "Senior"])
train["Prizeclass"] = pd.cut(train["Fare"], bins=[0, 50, 100, 200, 300, train["Fare"].max()],
                             labels=["0-50", "50-100", "101-200", "201-300", "300+"], include_lowest=True)

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
test['FamilyClass'] = pd.cut(test["FamilySize"], bins=[0, 1, 4, 6, 11], labels=["Solo", "Small", "Medium", "Large"])
test["AgeGroup"] = pd.cut(test["Age"], bins=[0, 12, 18, 60, 120], labels=["Child", "Teen", "Adult", "Senior"])
test["Prizeclass"] = pd.cut(test["Fare"], bins=[0, 50, 100, 200, 300, train["Fare"].max()],
                            labels=["0-50", "50-100", "101-200", "201-300", "300+"], include_lowest=True)

# Streamlit titel
st.title("Titanic Data Analysis")

# Plot: Scatterplot Age vs Fare
st.subheader("Ticketprijs vs Leeftijd (overlevenden)")
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train, ax=ax)
plt.xlabel('Leeftijd')
plt.ylabel('Ticketprijs (euro)')
st.pyplot(fig)

# Plot: Survived per Pclass
st.subheader("Overleving per Pclass")
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x="Pclass", y="Survived", data=train, ax=ax)
plt.ylabel('Overlevingspercentage')
st.pyplot(fig)

# Plot: Survived per Sex
st.subheader("Overleving per Geslacht")
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x="Sex", y="Survived", data=train, ax=ax)
plt.ylabel('Overlevingspercentage')
st.pyplot(fig)

# Plot: Survived per Embarked
st.subheader("Overleving per Vertrekhaven")
fig, ax = plt.subplots(figsize=(6,4))
sns.barplot(x="Embarked", y="Survived", data=train, ax=ax)
plt.ylabel('Overlevingspercentage')
st.pyplot(fig)

# Optioneel: Toon eerste 5 rijen
st.subheader("Voorbeeld van dataset")
st.dataframe(train.sample(5, random_state=42))
