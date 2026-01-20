import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import re
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
        if not self.models: 
            return np.zeros((X.shape[0], 1)) # Boşsa sıfır matrisi dön
        # Her alt modelden tahmini al ve birleştir
        preds = [model.predict(X) for model in self.models]
        return np.array(preds).T

# ==============================================================================
# 2. BÖLÜM: YARDIMCI FONKSİYONLAR
# ==============================================================================
def cerrahi_analiz_tek_satir(metin):
    BILINEN_UNVANLAR = sorted(['Harcama Yetkilisi', 'Gerçekleştirme Görevlisi', 'Muhasebe Yetkilisi', 'Üst Yönetici', 'Akademik Teşvik Komisyonu', 'Üniversite Yönetim Kurulu', 'Döner Sermaye Yürütme Kurulu', 'Fakülte Yönetim Kurulu', 'İtiraz Komisyonu', 'Birim Komisyon', 'Jüri', 'Üniversite Senatosu', 'Personel Daire Başkanı', 'Strateji Geliştirme Daire Başkanı', 'İdari ve Mali İşler Daire Başkanı', 'Sağlık Kültür ve Spor Daire Başkanı', 'Döner Sermaye İşletme Müdürü', 'Hastane Başmüdürü', 'Hukuk Müşaviri', 'Fakülte Sekreteri', 'Enstitü Sekreteri', 'Yüksekokul Sekreteri', 'Rektör Yardımcısı', 'Dekan Yardımcısı', 'Başhekim Yardımcısı', 'Müdür Yardımcısı', 'Yüksekokul Müdürü', 'Enstitü Müdürü', 'Merkez Müdürü', 'Şube Müdürü', 'Hastane Müdürü', 'Daire Başkanı', 'Rektör', 'Dekan', 'Başhekim', 'Genel Sekreter', 'Müdür', 'Memur', 'Şef', 'Tekniker', 'Sayman', 'Bilgisayar İşletmeni', 'Öğretim Üyesi', 'Başkan'], key=len, reverse=True)
    if not isinstance(metin, str) or metin in ["YOK", "Kaynakta Yok"]: return []
    roller = set()
    for unvan in BILINEN_UNVANLAR:
        if unvan.lower() in metin.lower():
            rol = unvan
            if any(k in unvan for k in ['Kurulu', 'Komisyonu', 'Senatosu', 'Jüri']): rol += ' Üyesi'
            roller.add(rol)
    return list(roller)

@st.cache_data
def analyze_excel_data(script_dir):
    try:
        path = os.path.join(script_dir, "sorumlu.xlsx")
        if not os.path.exists(path): return None
        df = pd.read_excel(path, sheet_name='VERİ-2-EMİR').fillna('')
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
# 3. BÖLÜM: MODELLERİN YÜKLENMESİ
# ==============================================================================
st.set_page_config(page_title="Hukuki Analiz Sistemi", layout="wide")

@st.cache_resource
def load_bundle():
    path = os.path.join(os.path.dirname(__file__), "final_models_combined.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return None

bundle = load_bundle()

# Global değişkenleri tanımla (Hata almamak için)
law_model = vec_law = damage_model = vec_dmg = classes = None

if bundle:
    # Anahtar isimleri pickle oluşturulurken ne verildiyse o olmalı.
    # Eğer hata alıyorsanız bundle.keys() ile kontrol edebilirsiniz.
    law_model = bundle.get('law_model_lr')
    damage_model = bundle.get('damage_model')
    vec_law = bundle.get('vectorizer_laws')
    vec_dmg = bundle.get('vectorizer_damage')
    classes = bundle.get('mlb_classes')
else:
    st.error("🚨 'final_models_combined.pkl' bulunamadı veya yüklenemedi!")

# ==============================================================================
# 4. BÖLÜM: UI
# ==============================================================================
tool = st.sidebar.radio("Seçiniz:", ("Sayıştay Karar Destek Sistemi", "Veri Analizi"))

if tool == "Sayıştay Karar Destek Sistemi":
    st.title("⚖️ Sayıştay Karar Destek Sistemi")
    txt = st.text_area("Analiz edilecek metni yazınız:", height=300)
    
    if st.button("🔍 Analizi Başlat", type="primary"):
        if not txt:
            st.warning("Lütfen bir metin giriniz.")
        elif not law_model or not vec_law:
            st.error("Model dosyaları eksik veya hatalı yüklendi. Lütfen 'final_models_combined.pkl' dosyasını kontrol edin.")
        else:
            with st.spinner("Analiz ediliyor..."):
                try:
                    # KANUN TAHMİNİ
                    X_l = vec_law.transform([txt])
                    y_l_pred = law_model.predict(X_l)
                    
                    # Güvenli index erişimi
                    pred_laws = []
                    if len(y_l_pred) > 0 and classes is not None:
                        first_pred = y_l_pred[0]
                        pred_laws = [classes[i] for i, v in enumerate(first_pred) if v == 1]
                    
                    # KAMU ZARARI TAHMİNİ
                    pred_dmg = "Bilinmiyor"
                    if damage_model and vec_dmg:
                        X_d = vec_dmg.transform([txt])
                        pred_dmg = "VAR" if damage_model.predict(X_d)[0] == 1 else "YOK"
                    
                    # SONUÇ GÖSTERİMİ
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("📚 İlgili Kanunlar")
                        if pred_laws:
                            for l in pred_laws: st.success(f"📍 {l}")
                        else: st.info("Eşleşen kanun bulunamadı.")
                    with c2:
                        st.subheader("💰 Kamu Zararı Durumu")
                        if pred_dmg == "VAR": st.error("🚨 KAMU ZARARI TESPİT EDİLDİ")
                        elif pred_dmg == "YOK": st.info("✅ KAMU ZARARI TESPİT EDİLMEDİ")
                        else: st.warning("Tahmin yapılamadı.")
                except Exception as e:
                    st.error(f"Analiz sırasında bir teknik hata oluştu: {e}")

else:
    st.title("📊 Veri Analizi")
    res = analyze_excel_data(os.path.dirname(__file__))
    if res:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.pie(res['karar_turu'], values=res['karar_turu'].columns[1], names=res['karar_turu'].columns[0], title="Karar Türleri Dağılımı", hole=0.4), use_container_width=True)
        with c2: st.plotly_chart(px.pie(res['kamu_zarari'], values=res['kamu_zarari'].columns[1], names=res['kamu_zarari'].columns[0], title="Kamu Zararı Oranı", hole=0.4), use_container_width=True)
        st.plotly_chart(px.bar(res['konu'].head(15), x=res['konu'].columns[1], y=res['konu'].columns[0], orientation='h', title="En Sık Karar Konuları"), use_container_width=True)
        if res['sorumlular'] is not None:
            st.plotly_chart(px.bar(res['sorumlular'].head(15), x=res['sorumlular'].columns[1], y=res['sorumlular'].columns[0], orientation='h', title="Sorumlu Unvanlar"), use_container_width=True)
    else:
        st.error("'sorumlu.xlsx' dosyası bulunamadı.")
