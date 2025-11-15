import streamlit as st
import pickle
import numpy as np
import os

# Gerekli kütüphaneleri ve temel sınıfları import ediyoruz.
# Pickle'ın özel sınıfımızı ve modelleri tanıması için bu importlar gereklidir.
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

# ==============================================================================
# CustomLawClassifier Sınıf Tanımı
# Bu tanım, eğitim script'inizdeki ile birebir aynı olmalıdır.
# ==============================================================================
class CustomLawClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.models = []
    def fit(self, X, Y):
        self.models = []
        for i in range(Y.shape[1]):
            y_subset = Y[:, i]; unique_classes = np.unique(y_subset)
            if len(unique_classes) < 2: model = DummyClassifier(strategy="constant", constant=unique_classes[0])
            else: model = clone(self.base_estimator)
            model.fit(X, y_subset); self.models.append(model)
        return self
    def predict(self, X): return np.array([model.predict(X) for model in self.models]).T

# ==============================================================================
# STREAMLIT UYGULAMASI
# ==============================================================================

st.set_page_config(page_title="Hukuki Metin Analizi", layout="wide")
st.title("⚖️ Hukuki Metin Analiz Aracı")

# === Model Yükleyici Fonksiyon ===
@st.cache_resource
def load_all_models():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, "final_models_combined.pkl")
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"🚨 Model dosyası ('final_models_combined.pkl') bulunamadı. Lütfen önce 'train.py' script'ini çalıştırdığınızdan emin olun.")
        return None

# === Modelleri Yükle ve Değişkenlere Ata ===
models_bundle = load_all_models()
if models_bundle is None:
    st.stop()

try:
    # İki farklı kanun modelini ve diğer bileşenleri yükle
    law_model_lr = models_bundle['law_model_lr']
    law_model_rf = models_bundle['law_model_rf']
    damage_model = models_bundle['damage_model']
    vectorizer_laws = models_bundle['vectorizer_laws']
    vectorizer_damage = models_bundle['vectorizer_damage']
    mlb_classes = models_bundle['mlb_classes']
except KeyError as e:
    st.error(f"🚨 Model dosyasında beklenen anahtar bulunamadı: {e}. Lütfen model dosyasının doğru 'train.py' script'i ile oluşturulduğundan emin olun.")
    st.stop()

# === KENAR ÇUBUĞU (SIDEBAR) - Model Seçimi ===
st.sidebar.title("⚙️ Ayarlar")
selected_model_name = st.sidebar.selectbox(
    "Kullanılacak Kanun Tahmin Modelini Seçin:",
    ("Logistic Regression", "Random Forest")
)

# Seçilen isme göre aktif modeli belirle
if selected_model_name == "Logistic Regression":
    active_law_model = law_model_lr
else:
    active_law_model = law_model_rf

st.sidebar.info(f"Şu anda **{selected_model_name}** modeli aktif.")

# === Ana Arayüz ===
st.markdown("Bu uygulama, girilen dava metnine göre seçtiğiniz modeli kullanarak ilgili **kanunları** ve **kamu zararı durumunu** tahmin eder.")
st.markdown("---")

# === Tahmin Fonksiyonu ===
def predict_case(text, law_model, damage_model, law_vec, damage_vec, classes):
    # Kanun tahmini
    X_laws = law_vec.transform([text])
    law_prediction_vector = law_model.predict(X_laws)[0]
    predicted_laws = [classes[i] for i, val in enumerate(law_prediction_vector) if val == 1]
    
    # Kamu Zararı tahmini
    X_damage = damage_vec.transform([text])
    damage_prediction_code = damage_model.predict(X_damage)[0]
    has_public_damage = "VAR" if damage_prediction_code == 1 else "YOK"
    return predicted_laws, has_public_damage

# === Kullanıcı Girdisi ve Sonuç Alanı ===
input_text = st.text_area("Analiz edilecek metni buraya girin:", height=250, placeholder="Örnek: Eşi çalışan personele aile yardımı ödeneği ödenmesi...")

if st.button("🔍 Analiz Et", type="primary"):
    if not input_text.strip():
        st.warning("Lütfen bir metin girin.")
    else:
        with st.spinner(f"**{selected_model_name}** modeli ile tahmin yapılıyor..."):
            laws, damage = predict_case(
                input_text, 
                active_law_model, # Seçilen aktif modeli kullan
                damage_model, 
                vectorizer_laws, 
                vectorizer_damage, 
                mlb_classes
            )
        
        st.success("✅ Tahmin tamamlandı!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📘 Tahmin Edilen Kanunlar")
            if laws:
                for k in laws: st.markdown(f"- {k}")
            else:
                st.warning("İlişkili bir kanun bulunamadı.")
        
        with col2:
            st.subheader("💸 Kamu Zararı Durumu")
            if damage == "VAR": st.error(f"**{damage}**")
            else: st.info(f"**{damage}**")
