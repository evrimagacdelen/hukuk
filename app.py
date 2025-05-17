import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# === CustomLawClassifier sınıfını tekrar tanımlıyoruz
class CustomLawClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model_class=LogisticRegression):
        self.base_model_class = base_model_class
        self.models = []

    def fit(self, X, Y):
        self.models = []
        for i in range(Y.shape[1]):
            unique_vals = np.unique(Y[:, i])
            if len(unique_vals) == 1:
                model = DummyClassifier(strategy="constant", constant=unique_vals[0])
            else:
                model = self.base_model_class(solver='liblinear', max_iter=200)
            model.fit(X, Y[:, i])
            self.models.append(model)
        return self

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.array(preds).T

# === Streamlit Arayüzü
st.title("Kamu Zararı Tahmin Aracı")
st.markdown("Bu uygulama girilen dava metnine göre ilgili **kanunları** ve **kamu zararı durumunu** tahmin eder.")

# === Model Yükleyici
@st.cache_resource
def load_models():
    with open("legal_models.pkl", "rb") as f:
        return pickle.load(f)

models_data = load_models()
law_model = models_data['law_model']
damage_model = models_data['damage_model']
vectorizer = models_data['vectorizer']
mlb_classes = models_data['mlb_classes']

# === Tahmin Fonksiyonu
def predict_case(text):
    X = vectorizer.transform([text])
    law_prediction = law_model.predict(X)[0]
    damage_prediction = damage_model.predict(X)[0]
    predicted_laws = [mlb_classes[i] for i, val in enumerate(law_prediction) if val == 1]
    return predicted_laws, "VAR" if damage_prediction == 1 else "YOK"

# === Kullanıcı Girdisi
input_text = st.text_area("Dava metnini buraya girin:", height=250)

# === Tahmin Butonu
if st.button("🔍 Tahmin Et"):
    if not input_text.strip():
        st.warning("Lütfen bir metin girin.")
    else:
        with st.spinner("Tahmin yapılıyor..."):
            laws, damage = predict_case(input_text)
            st.success("✅ Tahmin tamamlandı!")

            st.subheader("📘 Tahmin Edilen Kanunlar:")
            if laws:
                for k in laws:
                    st.markdown(f"- {k}")
            else:
                st.markdown("⚠️ Kanun tahmini yapılamadı.")

            st.subheader("Kamu Zararı:")
            st.markdown(f"**{damage}**")
