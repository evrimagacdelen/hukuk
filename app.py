import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import google.generativeai as genai

# Gerekli kütüphaneleri ve temel sınıfları import ediyoruz.
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

# ==============================================================================
# HATA DÜZELTMESİ: CustomLawClassifier Sınıf Tanımı
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
# DEĞİŞİKLİK 2: GEMINI API AYARLARI (API Anahtarı Koda Eklendi - ÖNERİLMEZ!)
# ==============================================================================
# 🚨 UYARI: API anahtarınızı asla herkese açık kod depolarında paylaşmayın!
# Bu anahtarın çalınması hesabınızın kötüye kullanılmasına neden olabilir.
try:
    # API anahtarını doğrudan buraya yazın.
    API_KEY = "BURAYA_YENİ_VE_GÜVENLİ_API_ANAHTARINIZI_YAPIŞTIRIN"
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gemini API yapılandırılırken bir hata oluştu: {e}")
    st.info("Lütfen kod içerisindeki API_KEY değişkenine geçerli bir anahtar girdiğinizden emin olun.")
    gemini_model = None

# ==============================================================================
# STREAMLIT UYGULAMASI
# ==============================================================================

# Sayfa yapılandırması
st.set_page_config(page_title="Hukuki Metin Analizi", layout="wide")

# Başlık ve açıklama
st.title("⚖️ Gelişmiş Hukuki Metin Analiz Aracı")
st.markdown("Bu uygulama, girilen dava metnine göre ilgili **kanunları**, **kamu zararı** durumunu tahmin eder ve **Gemini AI** ile dava metninin özetini çıkarır.")
st.markdown("---")

# === Model ve Veri Yükleyici Fonksiyonlar ===
@st.cache_resource
def load_all_models():
    """Tüm modelleri ve vektörleştiricileri güvenli bir şekilde yükler."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, "final_models_combined.pkl")
    try:
        with open(file_path, "rb") as f:
            models_data = pickle.load(f)
        return models_data
    except FileNotFoundError:
        st.error(f"🚨 Model dosyası bulunamadı: {file_path}")
        return None

# ==============================================================================
# DEĞİŞİKLİK 1: EXCEL DOSYASINI OTOMATİK OLARAK YÜKLEME
# ==============================================================================
@st.cache_data
def load_excel_data():
    """'SOMUT OLAY-PYHTON.xlsx' dosyasını app.py ile aynı dizinden yükler."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, "SOMUT OLAY-PYHTON.xlsx")
    try:
        df = pd.read_excel(file_path)
        if 'GİRİŞ' not in df.columns or 'Tam Metin' not in df.columns:
            st.error(f"'{file_path}' dosyasında 'GİRİŞ' ve/veya 'Tam Metin' sütunları bulunamadı.")
            return None
        return df
    except FileNotFoundError:
        st.error(f"🚨 Veri dosyası bulunamadı: {file_path}")
        st.info("Lütfen 'SOMUT OLAY-PYHTON.xlsx' dosyasının 'app.py' ile aynı dizinde olduğundan emin olun.")
        return None
    except Exception as e:
        st.error(f"Excel dosyası okunurken bir hata oluştu: {e}")
        return None

# === Modelleri ve Veriyi Yükle ===
models_bundle = load_all_models()
df_data = load_excel_data() # Dosya yükleyici yerine doğrudan fonksiyonu çağırıyoruz.

if models_bundle is None or df_data is None:
    st.warning("Modeller veya veri dosyası yüklenemedi. Lütfen yukarıdaki hata mesajlarını kontrol edin.")
    st.stop() # Modeller veya veri yoksa uygulamayı durdur

try:
    law_model = models_bundle['law_model']
    damage_model = models_bundle['damage_model']
    vectorizer_laws = models_bundle['vectorizer_laws']
    vectorizer_damage = models_bundle['vectorizer_damage']
    mlb_classes = models_bundle['mlb_classes']
except KeyError as e:
    st.error(f"🚨 Model dosyasında beklenen anahtar bulunamadı: {e}.")
    st.stop()

# === Yardımcı Fonksiyonlar ===
def predict_case(text, law_vec, damage_vec, law_mdl, damage_mdl, classes):
    """Verilen metin için hem kanun hem de kamu zararı tahmini yapar."""
    X_laws = law_vec.transform([text])
    law_prediction_vector = law_mdl.predict(X_laws)[0]
    predicted_laws = [classes[i] for i, val in enumerate(law_prediction_vector) if val == 1]
    
    X_damage = damage_vec.transform([text])
    damage_prediction_code = damage_mdl.predict(X_damage)[0]
    has_public_damage = "VAR" if damage_prediction_code == 1 else "YOK"
    return predicted_laws, has_public_damage

def find_full_text(df, input_text):
    """DataFrame'de verilen giriş metnini arar ve karşılık gelen 'Tam Metin'i döndürür."""
    mask = df['GİRİŞ'].str.strip().str.startswith(input_text.strip(), na=False)
    if mask.any():
        return df.loc[mask, 'Tam Metin'].iloc[0]
    return None

def get_gemini_summary(text):
    """Verilen metni Gemini API'sine göndererek bir özet alır."""
    if gemini_model is None:
        return "Gemini modeli yüklenemediği için özet oluşturulamadı."
    try:
        prompt = f"""Aşağıdaki hukuki metni analiz et ve ana konuyu, tarafların temel argümanlarını ve olayın sonucunu (eğer belirtilmişse) vurgulayan kısa ve anlaşılır bir özet çıkar. Özet, hukuki terimlerden arındırılmış ve herkesin anlayabileceği bir dilde olmalıdır.

Metin:
"{text}"

Özet:
"""
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini özetleme sırasında bir hata oluştu: {e}"

# === Kullanıcı Arayüzü ===
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Dava Metni (Giriş Kısmı)")
    input_text = st.text_area(
        "Analiz edilecek metnin başlangıç kısmını buraya girin:", 
        height=300, 
        placeholder="Örnek: Eşi çalışan personele aile yardımı ödeneği ödenmesi..."
    )

    if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Lütfen analiz için bir metin girin.")
        else:
            with st.spinner("Analiz yapılıyor..."):
                laws, damage = predict_case(input_text, vectorizer_laws, vectorizer_damage, law_model, damage_model, mlb_classes)
                st.session_state['predicted_laws'] = laws
                st.session_state['predicted_damage'] = damage
                
                full_text = find_full_text(df_data, input_text)
                
                if full_text:
                    gemini_summary = get_gemini_summary(full_text)
                    st.session_state['gemini_summary'] = gemini_summary
                else:
                    st.session_state['gemini_summary'] = "Girdiğiniz metinle eşleşen bir 'Tam Metin' Excel dosyasında bulunamadı. Özetleme yapılamadı."

                st.session_state['ran_prediction'] = True

with col2:
    st.subheader("📊 Analiz Sonuçları")
    if 'ran_prediction' in st.session_state:
        st.markdown("##### 📘 Tahmin Edilen İlgili Kanunlar:")
        if st.session_state['predicted_laws']:
            for k in st.session_state['predicted_laws']:
                st.success(f"- {k}")
        else:
            st.info("⚠️ İlişkili bir kanun bulunamadı.")
        
        st.markdown("---")
        st.markdown("##### 💸 Kamu Zararı Durumu:")
        damage_result = st.session_state['predicted_damage']
        if damage_result == "VAR":
            st.error(f"**{damage_result}**")
        else:
            st.info(f"**{damage_result}**")
        
        st.markdown("---")
        st.markdown("##### 🤖 Gemini AI Metin Özeti:")
        with st.expander("Özeti Görmek İçin Tıklayın", expanded=True):
            st.info(st.session_state.get('gemini_summary', 'Özet bulunamadı.'))
    else:
        st.info("Sonuçları görmek için lütfen sol tarafa bir metin girip 'Analiz Et' butonuna tıklayın.")
