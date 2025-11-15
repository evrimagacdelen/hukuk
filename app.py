import streamlit as st
import pickle
import numpy as np
import os

# Gerekli kütüphaneleri ve temel sınıfları import ediyoruz.
# pickle.load() fonksiyonunun özel sınıfımızı ve modelleri tanıması için bu importlar gereklidir.
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

# ==============================================================================
# HATA DÜZELTMESİ: CustomLawClassifier Sınıf Tanımı
# Bu tanım, eğitim script'inizdeki ile birebir aynı olmalıdır.
# Pickle, .pkl dosyasını okurken bu sınıfın yapısını bilmek zorundadır.
# ==============================================================================
class CustomLawClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.models = []
    def fit(self, X, Y):
        self.models = []
        for i in range(Y.shape[1]):
            y_subset = Y[:, i]
            unique_classes = np.unique(y_subset)
            if len(unique_classes) < 2:
                model = DummyClassifier(strategy="constant", constant=unique_classes[0])
            else:
                model = clone(self.base_estimator)
            model.fit(X, y_subset)
            self.models.append(model)
        return self
    def predict(self, X):
        return np.array([model.predict(X) for model in self.models]).T

# ==============================================================================
# STREAMLIT UYGULAMASI
# ==============================================================================

# Sayfa yapılandırması (geniş mod ve başlık)
st.set_page_config(page_title="Hukuki Metin Analizi", layout="wide")

# Başlık ve açıklama
st.title("⚖️ Kamu Zararı ve İlgili Kanun Tahmin Aracı")
st.markdown("Bu uygulama, girilen dava metnine göre ilgili **kanunları** ve **kamu zararı** olup olmadığını tahmin eder.")
st.markdown("---")

# === Model Yükleyici Fonksiyon ===
@st.cache_resource
def load_all_models():
    """
    Tüm modelleri ve vektörleştiricileri, dosyanın tam yolunu bularak güvenli bir şekilde yükler.
    """
    # Bu kod, app.py dosyasının bulunduğu dizini bularak dosya yolunu doğru şekilde oluşturur.
    # Bu sayede "FileNotFoundError" hatasının önüne geçilir.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, "final_models_combined.pkl")
    
    try:
        with open(file_path, "rb") as f:
            models_data = pickle.load(f)
        return models_data
    except FileNotFoundError:
        # Hata durumunda kullanıcıya bilgilendirici bir mesaj gösterilir.
        st.error(f"🚨 Model dosyası belirtilen yolda bulunamadı: {file_path}")
        st.info("Lütfen 'final_models_combined.pkl' dosyasının 'app.py' ile aynı dizinde olduğundan emin olun.")
        return None

# === Modelleri Yükle ve Değişkenlere Ata ===
models_bundle = load_all_models()

# Modellerin başarılı bir şekilde yüklenip yüklenmediğini kontrol et
if models_bundle is None:
    st.stop() # Model yoksa uygulamayı durdur
else:
    try:
        # Doğru anahtarları kullanarak her bir bileşeni değişkene ata
        law_model = models_bundle['law_model']
        damage_model = models_bundle['damage_model']
        vectorizer_laws = models_bundle['vectorizer_laws']
        vectorizer_damage = models_bundle['vectorizer_damage']
        mlb_classes = models_bundle['mlb_classes']
    except KeyError as e:
        st.error(f"🚨 Model dosyasında beklenen anahtar bulunamadı: {e}. Lütfen model dosyasının doğru eğitim script'i ile oluşturulduğundan emin olun.")
        st.stop()


# === Tahmin Fonksiyonu ===
def predict_case(text, law_vec, damage_vec, law_mdl, damage_mdl, classes):
    """
    Verilen metin için hem kanun hem de kamu zararı tahmini yapar.
    Her model kendi özel vektörleştiricisini kullanır.
    """
    # Kanun tahmini için 'vectorizer_laws' kullanılıyor
    X_laws = law_vec.transform([text])
    law_prediction_vector = law_mdl.predict(X_laws)[0]
    predicted_laws = [classes[i] for i, val in enumerate(law_prediction_vector) if val == 1]
    
    # Kamu Zararı tahmini için 'vectorizer_damage' kullanılıyor
    X_damage = damage_vec.transform([text])
    damage_prediction_code = damage_mdl.predict(X_damage)[0]
    has_public_damage = "VAR" if damage_prediction_code == 1 else "YOK"

    return predicted_laws, has_public_damage

# === Kullanıcı Arayüzü (İki Sütunlu Tasarım) ===
col1, col2 = st.columns([2, 1]) # Giriş sütunu daha geniş olsun

with col1:
    st.subheader("📝 Dava Metni")
    input_text = st.text_area(
        "Analiz edilecek metni buraya girin:", 
        height=300, 
        placeholder="Örnek: Eşi çalışan personele aile yardımı ödeneği ödenmesi..."
    )

    # Butona basıldığında tahmin işlemini başlat
    if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Lütfen analiz için bir metin girin.")
        else:
            with st.spinner("Modeller çalışıyor, tahminler yapılıyor..."):
                # Tahminleri yap ve sonuçları session_state'e kaydet (sayfa yenilense de kalır)
                laws, damage = predict_case(
                    input_text, 
                    vectorizer_laws, 
                    vectorizer_damage, 
                    law_model, 
                    damage_model, 
                    mlb_classes
                )
                st.session_state['predicted_laws'] = laws
                st.session_state['predicted_damage'] = damage
                st.session_state['ran_prediction'] = True

with col2:
    st.subheader("📊 Analiz Sonuçları")
    # Eğer daha önce bir tahmin yapıldıysa sonuçları göster
    if 'ran_prediction' in st.session_state:
        st.markdown("##### 📘 Tahmin Edilen İlgili Kanunlar:")
        if st.session_state['predicted_laws']:
            for k in st.session_state['predicted_laws']:
                st.success(f"- {k}")
        else:
            st.warning("⚠️ İlişkili bir kanun bulunamadı.")
        
        st.markdown("---")

        st.markdown("##### 💸 Kamu Zararı Durumu:")
        damage_result = st.session_state['predicted_damage']
        if damage_result == "VAR":
            st.error(f"**{damage_result}**")
        else:
            st.info(f"**{damage_result}**")
    else:
        st.info("Sonuçları görmek için lütfen sol tarafa bir metin girip 'Analiz Et' butonuna tıklayın.")
