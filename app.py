import streamlit as st
import pickle
import numpy as np

# Gerekli kütüphaneleri ve temel sınıfları import ediyoruz.
# Pickle'ın özel sınıfı (CustomLawClassifier) çözebilmesi için bunlar gereklidir.
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

# DÜZELTME 1: CustomLawClassifier sınıfının tanımını buradan SİLİYORUZ.
# Pickle dosyası zaten bu sınıfın yapısını biliyor, tekrar tanımlamak gereksiz ve riskli.

# === Streamlit Arayüzü
st.title("Kamu Zararı Tahmin Aracı")
st.markdown("Bu uygulama girilen dava metnine göre ilgili **kanunları** ve **kamu zararı durumunu** tahmin eder.")

# === Model Yükleyici
@st.cache_resource
def load_models():
    # DÜZELTME 2: Eğitimde kaydettiğiniz doğru dosya adını kullanıyoruz.
    with open("legal_models.pkl", "rb") as f:
        models_data = pickle.load(f)
    return models_data

# DÜZELTME 3: .pkl dosyasındaki TÜM doğru anahtarları yüklüyoruz.
try:
    models_data = load_models()
    law_model = models_data['law_model']
    damage_model = models_data['damage_model']
    vectorizer_laws = models_data['vectorizer_laws']       # Kanun için ayrı vektörleştirici
    vectorizer_damage = models_data['vectorizer_damage'] # Kamu zararı için ayrı vektörleştirici
    mlb_classes = models_data['mlb_classes']
except FileNotFoundError:
    st.error("Model dosyası ('final_models_combined.pkl') bulunamadı. Lütfen dosyanın doğru yolda olduğundan emin olun.")
    st.stop()
except KeyError as e:
    st.error(f"Model dosyasında beklenen anahtar bulunamadı: {e}. Lütfen eğitim script'i ile .pkl dosyasının uyumlu olduğundan emin olun.")
    st.stop()


# === Tahmin Fonksiyonu
def predict_case(text, law_vec, damage_vec, law_mdl, damage_mdl, classes):
    # DÜZELTME 4: Her model için kendi doğru vektörleştiricisini kullanıyoruz.
    
    # Kanun tahmini için 'vectorizer_laws' kullanılıyor
    X_laws = law_vec.transform([text])
    law_prediction_vector = law_mdl.predict(X_laws)[0]
    predicted_laws = [classes[i] for i, val in enumerate(law_prediction_vector) if val == 1]
    
    # Kamu Zararı tahmini için 'vectorizer_damage' kullanılıyor
    X_damage = damage_vec.transform([text])
    damage_prediction_code = damage_mdl.predict(X_damage)[0]
    has_public_damage = "VAR" if damage_prediction_code == 1 else "YOK"

    return predicted_laws, has_public_damage

# === Kullanıcı Girdisi
input_text = st.text_area("Dava metnini buraya girin:", height=200, placeholder="Örnek: Eşinden boşanan personele aile yardımı ödemesinin yapılması...")

# === Tahmin Butonu
if st.button("🔍 Tahmin Et", type="primary"):
    if not input_text.strip():
        st.warning("Lütfen bir metin girin.")
    else:
        with st.spinner("Tahmin yapılıyor..."):
            # DÜZELTME 5: Fonksiyona gerekli tüm model ve vektörleştiricileri iletiyoruz.
            laws, damage = predict_case(
                input_text, 
                vectorizer_laws, 
                vectorizer_damage, 
                law_model, 
                damage_model, 
                mlb_classes
            )
            
            st.success("✅ Tahmin tamamlandı!")

            st.subheader("📘 Tahmin Edilen Kanunlar:")
            if laws:
                for k in laws:
                    st.markdown(f"- {k}")
            else:
                st.markdown("⚠️ İlişkili bir kanun bulunamadı.")

            st.subheader("Kamu Zararı Durumu:")
            # Sonucu daha belirgin hale getirelim
            if damage == "VAR":
                st.markdown(f"**<p style='color:red;'>{damage}</p>**", unsafe_allow_html=True)
            else:
                st.markdown(f"**<p style='color:green;'>{damage}</p>**", unsafe_allow_html=True)

