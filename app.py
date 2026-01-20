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
    def __init__(self, base_estimator=None):
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
        if not self.models: return np.array([])
        return np.array([model.predict(X) for model in self.models]).T

# ==============================================================================
# 2. BÖLÜM: MODEL YÜKLEME (Geliştirilmiş)
# ==============================================================================
@st.cache_resource
def load_bundle():
    path = os.path.join(os.path.dirname(__file__), "final_models_combined.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None

bundle = load_bundle()

# Modelleri güvenli bir şekilde çekelim
if bundle:
    # Anahtar isimleri değişmiş olabilir, alternatifleri kontrol edelim
    law_model = bundle.get('law_model_lr') or bundle.get('law_model')
    damage_model = bundle.get('damage_model')
    vec_law = bundle.get('vectorizer_laws') or bundle.get('vectorizer_law')
    vec_dmg = bundle.get('vectorizer_damage')
    classes = bundle.get('mlb_classes')
else:
    law_model = damage_model = vec_law = vec_dmg = classes = None

# ==============================================================================
# 3. BÖLÜM: ANALİZ FONKSİYONLARI
# ==============================================================================
def cerrahi_analiz_tek_satir(metin):
    BILINEN_UNVANLAR = sorted(['Harcama Yetkilisi', 'Gerçekleştirme Görevlisi', 'Muhasebe Yetkilisi', 'Üst Yönetici', 'Akademik Teşvik Komisyonu', 'Rektör', 'Dekan', 'Başhekim', 'Genel Sekreter', 'Müdür', 'Memur', 'Şef'], key=len, reverse=True)
    if not isinstance(metin, str) or metin in ["YOK", "Kaynakta Yok"]: return []
    roller = set()
    for unvan in BILINEN_UNVANLAR:
        if unvan.lower() in metin.lower():
            roller.add(unvan)
    return list(roller)

@st.cache_data
def analyze_excel_data(script_dir):
    try:
        df = pd.read_excel(os.path.join(script_dir, "sorumlu.xlsx"), sheet_name='VERİ-2-EMİR').fillna('')
        sutun_map = {'Kararların Niteliği': 'Karar_Turu', 'Kamu Zararı Var mı?': 'Kamu_Zarari', 'Kamu Zararının Sorumlusu Kim?': 'Sorumlular', 'Kararın Konusu Nedir?': 'Konu'}
        df.rename(columns=sutun_map, inplace=True)
        return {
            "karar_turu": df['Karar_Turu'].value_counts().reset_index(),
            "kamu_zarari": df['Kamu_Zarari'].str.contains('Var', case=False).map({True:'Zarar Var', False:'Zarar Yok'}).value_counts().reset_index(),
            "konu": df['Konu'].value_counts().reset_index(),
            "sorumlular": pd.DataFrame([{'Unvan': u} for s in df['Sorumlular'] for u in cerrahi_analiz_tek_satir(s)])['Unvan'].value_counts().reset_index() if not df.empty else None
        }
    except: return None

# ==============================================================================
# 4. BÖLÜM: UI
# ==============================================================================
st.set_page_config(page_title="Hukuki Analiz Sistemi", layout="wide")
tool = st.sidebar.radio("Seçiniz:", ("Sayıştay Karar Destek Sistemi", "Veri Analizi"))

if tool == "Sayıştay Karar Destek Sistemi":
    st.title("⚖️ Sayıştay Karar Destek Sistemi")
    
    if bundle is None:
        st.error("🚨 'final_models_combined.pkl' dosyası bulunamadı veya yüklenemedi. Lütfen dosyanın uygulama ile aynı klasörde olduğundan emin olun.")
    else:
        txt = st.text_area("Analiz edilecek metni yazınız:", height=250)
        
        if st.button("🔍 Analizi Başlat", type="primary"):
            if txt:
                try:
                    with st.spinner("Analiz ediliyor..."):
                        # Tahminleri yapmadan önce objelerin varlığını kontrol et
                        if law_model and vec_law:
                            X_l = vec_law.transform([txt])
                            y_l = law_model.predict(X_l)[0]
                            pred_laws = [classes[i] for i, v in enumerate(y_l) if v == 1]
                        else:
                            pred_laws = []

                        if damage_model and vec_dmg:
                            X_d = vec_dmg.transform([txt])
                            pred_dmg = "VAR" if damage_model.predict(X_d)[0] == 1 else "YOK"
                        else:
                            pred_dmg = "Bilinmiyor"

                        # Sonuç Paneli
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("📚 İlgili Kanunlar")
                            if pred_laws:
                                for l in pred_laws: st.success(l)
                            else: st.info("Eşleşen kanun bulunamadı.")
                        with col2:
                            st.subheader("💰 Kamu Zararı")
                            if pred_dmg == "VAR": st.error("🚨 TESPİT EDİLDİ")
                            elif pred_dmg == "YOK": st.info("✅ TESPİT EDİLMEDİ")
                            else: st.warning("Tahmin yapılamadı.")
                except Exception as e:
                    st.error(f"Analiz sırasında bir hata oluştu: {e}")
            else:
                st.warning("Lütfen bir metin giriniz.")

else:
    st.title("📊 Veri Analizi")
    res = analyze_excel_data(os.path.dirname(__file__))
    if res:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.pie(res['karar_turu'], values=res['karar_turu'].columns[1], names=res['karar_turu'].columns[0], title="Karar Türleri Dağılımı", hole=0.4), use_container_width=True)
        with c2: st.plotly_chart(px.pie(res['kamu_zarari'], values=res['kamu_zarari'].columns[1], names=res['kamu_zarari'].columns[0], title="Kamu Zararı Oranı", hole=0.4), use_container_width=True)
        st.plotly_chart(px.bar(res['konu'].head(15), x=res['konu'].columns[1], y=res['konu'].columns[0], orientation='h', title="En Sık Karar Konuları"), use_container_width=True)
    else:
        st.error("Analiz dosyası (sorumlu.xlsx) bulunamadı.")
