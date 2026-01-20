import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import plotly.express as px
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier

# ==============================================================================
# 1. BÖLÜM: MODEL İÇİN GEREKLİ SINIF
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
        if not self.models: return np.zeros((X.shape[0], 1))
        preds = [model.predict(X) for model in self.models]
        return np.array(preds).T

# ==============================================================================
# 2. BÖLÜM: MODEL YÜKLEME VE HATA KONTROLÜ
# ==============================================================================
st.set_page_config(page_title="Hukuki Analiz Sistemi", layout="wide")

@st.cache_resource
def load_bundle():
    # Dosya yolu kontrolü
    path = os.path.join(os.path.dirname(__file__), "final_models_combined.pkl")
    if not os.path.exists(path):
        st.error(f"Kritik Hata: '{path}' dosyası sunucuda bulunamadı!")
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception as e:
        st.error(f"Pickle dosyası okunurken hata oluştu: {e}")
        return None

bundle = load_bundle()

# --- DEBUG: Pickle içindeki anahtarları kontrol et ---
if bundle:
    with st.expander("🛠️ Teknik Detaylar (Model Dosyası İçeriği)"):
        st.write("Dosya başarıyla yüklendi.")
        st.write("İçindeki Anahtarlar (Keys):", list(bundle.keys()))
else:
    st.stop() # Bundle yoksa uygulamayı burada durdur

# Modelleri Çıkart (Eğer anahtar isimleri farklıysa bundle.get('BURAYA_YENI_ISIM') yapın)
law_model = bundle.get('law_model_lr')
damage_model = bundle.get('damage_model')
vec_law = bundle.get('vectorizer_laws')
vec_dmg = bundle.get('vectorizer_damage')
classes = bundle.get('mlb_classes')

# Kontrol: Eğer yüklenenlerden biri eksikse kullanıcıyı uyar
missing_items = []
if law_model is None: missing_items.append("law_model_lr")
if damage_model is None: missing_items.append("damage_model")
if vec_law is None: missing_items.append("vectorizer_laws")
if vec_dmg is None: missing_items.append("vectorizer_damage")
if classes is None: missing_items.append("mlb_classes")

if missing_items:
    st.error(f"🚨 Model dosyasında şu anahtarlar eksik: {', '.join(missing_items)}")
    st.info("Lütfen Pickle dosyasını oluştururken kullandığınız anahtar isimleri ile yukarıdakilerin aynı olduğundan emin olun.")

# ==============================================================================
# 3. BÖLÜM: YARDIMCI ANALİZ FONKSİYONU
# ==============================================================================
@st.cache_data
def analyze_excel_data():
    try:
        path = "sorumlu.xlsx"
        if not os.path.exists(path): return None
        df = pd.read_excel(path, sheet_name='VERİ-2-EMİR').fillna('')
        sutun_map = {'Kararların Niteliği': 'Karar_Turu', 'Kamu Zararı Var mı?': 'Kamu_Zarari', 'Kamu Zararının Sorumlusu Kim?': 'Sorumlular', 'Kararın Konusu Nedir?': 'Konu'}
        df.rename(columns=sutun_map, inplace=True)
        return {
            "karar_turu": df['Karar_Turu'].value_counts().reset_index(),
            "kamu_zarari": df['Kamu_Zarari'].str.contains('Var', case=False).map({True:'Zarar Var', False:'Zarar Yok'}).value_counts().reset_index(),
            "konu": df['Konu'].value_counts().reset_index()
        }
    except: return None

# ==============================================================================
# 4. BÖLÜM: UI (KULLANICI ARAYÜZÜ)
# ==============================================================================
tool = st.sidebar.radio("Seçiniz:", ("Sayıştay Karar Destek Sistemi", "Veri Analizi"))

if tool == "Sayıştay Karar Destek Sistemi":
    st.title("⚖️ Sayıştay Karar Destek Sistemi")
    txt = st.text_area("Analiz edilecek metni yazınız:", height=300)
    
    if st.button("🔍 Analizi Başlat", type="primary"):
        if not txt:
            st.warning("Lütfen bir metin giriniz.")
        elif missing_items:
            st.error("Modeller eksik olduğu için tahmin yapılamıyor.")
        else:
            with st.spinner("Tahmin ediliyor..."):
                try:
                    # Tahminler
                    X_l = vec_law.transform([txt])
                    y_l = law_model.predict(X_l)[0]
                    pred_laws = [classes[i] for i, v in enumerate(y_l) if v == 1]
                    
                    X_d = vec_dmg.transform([txt])
                    pred_dmg = "VAR" if damage_model.predict(X_d)[0] == 1 else "YOK"
                    
                    # Sonuçlar
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📚 İlgili Kanunlar")
                        if pred_laws:
                            for l in pred_laws: st.success(l)
                        else: st.info("Eşleşme bulunamadı.")
                    with col2:
                        st.subheader("💰 Kamu Zararı")
                        if pred_dmg == "VAR": st.error("🚨 TESPİT EDİLDİ")
                        else: st.info("✅ TESPİT EDİLMEDİ")
                except Exception as e:
                    st.error(f"Tahmin hatası: {e}")

else:
    st.title("📊 Veri Analizi")
    res = analyze_excel_data()
    if res:
        st.plotly_chart(px.pie(res['karar_turu'], values=res['karar_turu'].columns[1], names=res['karar_turu'].columns[0], title="Karar Türleri Dağılımı"), use_container_width=True)
        st.plotly_chart(px.bar(res['konu'].head(10), x=res['konu'].columns[1], y=res['konu'].columns[0], orientation='h', title="En Sık Konular"), use_container_width=True)
    else:
        st.error("'sorumlu.xlsx' bulunamadı.")
