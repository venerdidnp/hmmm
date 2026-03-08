import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Sistem Diagnosa Medis AI", layout="wide")
st.title("🏥 Sistem Pakar Diagnosa Medis (DT & RF)")

# --- Navigasi Sidebar ---
menu = st.sidebar.selectbox("Pilih Kasus Diagnosa:", ["Diabetes (Decision Tree)", "Penyakit Jantung (Random Forest)"])

# 1. MODEL DIABETES (Sesuai Modul 5.B & Hasil Run Mas)
def model_diabetes():
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    pima = pd.read_csv("diabetes.csv", header=0, names=col_names)
    # Urutan kolom fitur sesuai screenshot Mas
    feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
    X = pima[feature_cols]
    y = pima.Outcome
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf.fit(X_train, y_train)
    return clf

# 2. MODEL JANTUNG (Sesuai Dataset Mas yang berisi 5 kolom)
def model_heart():
    # Gunakan nama kolom yang ada di screenshot Mas
    df = pd.read_csv('heart_v2.csv')
    X = df.drop('heart disease', axis=1)
    y = df['heart disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    # Gunakan hasil Grid Search Mas
    rf_best = RandomForestClassifier(max_depth=5, min_samples_leaf=10, n_estimators=10, random_state=42)
    rf_best.fit(X_train, y_train)
    return rf_best

# --- Tampilan Halaman ---
if menu == "Diabetes (Decision Tree)":
    st.subheader("Form Input Parameter Diabetes")
    clf = model_diabetes()
    
    c1, c2 = st.columns(2)
    with c1:
        glu = st.number_input("Kadar Glukosa", 0, 200, 120)
        ins = st.number_input("Kadar Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
    with c2:
        preg = st.number_input("Kehamilan", 0, 20, 1)
        bp = st.number_input("Tekanan Darah", 0, 150, 70)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Usia", 1, 120, 30)

    if st.button("Diagnosa Sekarang"):
        # Urutan input harus sama dengan feature_cols di atas
        input_data = [[preg, ins, bmi, age, glu, bp, dpf]]
        res = clf.predict(input_data)
        if res[0] == 1:
            st.error("⚠️ Hasil: Pasien Terindikasi Diabetes")
        else:
            st.success("✅ Hasil: Pasien Sehat")

else:
    st.subheader("Form Input Parameter Penyakit Jantung")
    rf = model_heart()
    
    # Input disesuaikan hanya 4 fitur sesuai dataset Mas
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Usia", 1, 100, 50)
        sex = st.selectbox("Jenis Kelamin", [1, 0], format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
    with c2:
        bp = st.number_input("Tekanan Darah (BP)", 50, 250, 120)
        chol = st.number_input("Kadar Kolesterol", 100, 600, 200)

    if st.button("Diagnosa Sekarang"):
        # Pastikan urutan age, sex, BP, cholestrol sesuai urutan drop() tadi
        input_data = [[age, sex, bp, chol]]
        res = rf.predict(input_data)
        if res[0] == 1:
            st.error("⚠️ Hasil: Pasien Terindikasi Penyakit Jantung")
        else:
            st.success("✅ Hasil: Pasien Sehat")