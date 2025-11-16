import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import google.generativeai as genai
import re
import plotly.express as px
import io
import traceback

# ==============================================================================
# BÖLÜM 1: TAHMİN MODELİ İÇİN GEREKLİ SINIF VE FONKSİYONLAR
# ==============================================================================
# ... (Bu bölümde değişiklik yok) ...
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier

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
# BÖLÜM 2: GÖRSELLEŞTİRME VE ANALİZ İÇİN FONKSİYONLAR
# ==============================================================================
# ... (Bu bölümde değişiklik yok) ...
def cerrahi_analiz_tek_satir(metin):
    BILINEN_UNVANLAR = sorted(['Harcama Yetkilisi', 'Gerçekleştirme Görevlisi', 'Muhasebe Yetkilisi', 'Üst Yönetici', 'Akademik Teşvik Komisyonu', 'Üniversite Yönetim Kurulu', 'Döner Sermaye Yürütme Kurulu', 'Fakülte Yönetim Kurulu', 'İtiraz Komisyonu', 'Birim Komisyon', 'Jüri', 'Üniversite Senatosu', 'Personel Daire Başkanı', 'Strateji Geliştirme Daire Başkanı', 'İdari ve Mali İşler Daire Başkanı', 'Sağlık Kültür ve Spor Daire Başkanı', 'Döner Sermaye İşletme Müdürü', 'Hastane Başmüdürü', 'Hukuk Müşaviri', 'Fakülte Sekreteri', 'Enstitü Sekreteri', 'Yüksekokul Sekreteri', 'Rektör Yardımcısı', 'Dekan Yardımcısı', 'Başhekim Yardımcısı', 'Müdür Yardımcısı', 'Yüksekokul Müdürü', 'Enstitü Müdürü', 'Merkez Müdürü', 'Şube Müdürü', 'Hastane Müdürü', 'Daire Başkanı', 'Rektör', 'Dekan', 'Başhekim', 'Genel Sekreter', 'Müdür', 'Memur', 'Şef', 'Tekniker', 'Sayman', 'Bilgisayar İşletmeni', 'Öğretim Üyesi', 'Başkan'], key=len, reverse=True)
    AKADEMIK_DESENLER = {'Prof. Dr.': r'prof\s*\.\s*dr', 'Doç. Dr.': r'doç\s*\.\s*dr', 'Yrd. Doç. Dr.': r'yrd\s*\.\s*doç\s*\.\s*dr', 'Dr. Öğr. Üyesi': r'dr\s*\.\s*öğr\s*\.\s*üyesi', 'Öğr. Gör.': r'öğr\s*\.\s*gör', 'Dr.': r'\bdr\b'}
    NORM_MAP = {'hy': 'Harcama Yetkilisi', 'gg': 'Gerçekleştirme Görevlisi', 'dekan v.': 'Dekan Vekili', 'dekan v': 'Dekan Vekili', 'rektör yrd.': 'Rektör Yardımcısı', 'rektör yrd': 'Rektör Yardımcısı', 'müdür v.': 'Müdür Vekili', 'müdür v': 'Müdür Vekili', 'müdür yrd.': 'Müdür Yardımcısı', 'müdür yrd': 'Müdür Yardımcısı', 'fakülte sekreter v.': 'Fakülte Sekreteri Vekili', 'fakülte sekreter v': 'Fakülte Sekreteri Vekili', 'fakülte sekreterv': 'Fakülte Sekreteri Vekili', 'fakül. sekr. vekili': 'Fakülte Sekreteri Vekili', 'yüksekokul sekreter v.': 'Yüksekokul Sekreteri Vekili', 'yüksekokul sekreter v': 'Yüksekokul Sekreteri Vekili', 'yüksekokul sek. v': 'Yüksekokul Sekreteri Vekili', 'genel sekreter v.': 'Genel Sekreter Vekili', 'genel sekreter v': 'Genel Sekreter Vekili', 'döner ser. işl. md. v.': 'Döner Sermaye İşletme Müdürü Vekili', 'işletme müd. v.': 'İşletme Müdürü Vekili', 'hastane md. yrd': 'Hastane Müdür Yardımcısı', 'has. baş müd.': 'Hastane Başmüdürü', 'üyk': 'Üniversite Yönetim Kurulu', 'dsyk': 'Döner Sermaye Yürütme Kurulu'}
    if not isinstance(metin, str) or metin in ["YOK", "Kaynakta Yok"]: return []
    anlamsız_ifadeler = ['zararın', 'tahsil edildiği', 'ilişik kalmadı', 'kastedilmektedir', 'implicit', 'münferiden sorumlu']
    if any(ifade in metin.lower() for ifade in anlamsız_ifadeler): return []
    roller_bu_satirda, kalan_metin = set(), metin
    for standart_ad, desen in AKADEMIK_DESENLER.items():
        if re.search(desen, kalan_metin, re.IGNORECASE):
            roller_bu_satirda.add(standart_ad); kalan_metin = re.sub(desen, '', kalan_metin, flags=re.IGNORECASE)
    for unvan in BILINEN_UNVANLAR:
        if unvan.lower() in kalan_metin.lower():
            rol = unvan
            if any(k in unvan for k in ['Kurulu', 'Komisyonu', 'Senatosu', 'Jüri']): rol += ' Üyesi'
            roller_bu_satirda.add(rol); kalan_metin = re.sub(re.escape(unvan), '', kalan_metin, flags=re.IGNORECASE)
    potensiyel_roller = re.split(r'[,/()]|\s+ve\s+|\s+ile\s+', kalan_metin)
    for parca in potensiyel_roller:
        temiz_parca = parca.strip().lower()
        if temiz_parca in NORM_MAP: roller_bu_satirda.add(NORM_MAP[temiz_parca])
        elif 'vekili' in temiz_parca or temiz_parca.endswith((' v', ' v.')):
            if 'dekan' in temiz_parca: roller_bu_satirda.add('Dekan Vekili')
            elif 'rektör' in temiz_parca: roller_bu_satirda.add('Rektör Vekili')
    return list(roller_bu_satirda)

def create_plotly_pie(df, title):
    if df is None or df.empty:
        return None
    fig = px.pie(df, values=df.columns[1], names=df.columns[0], title=title, hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(height=400, title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white' if st.get_option("theme.base") == "dark" else "black"))
    fig.update_traces(hovertemplate="<b>%{label}</b><br>Sayı: %{value}<br>Yüzde: %{percent}")
    return fig

def create_plotly_bar(df, title, top_n=15):
    if df is None or df.empty:
        return None
    data_to_plot = df.head(top_n)
    fig = px.bar(data_to_plot, x=df.columns[1], y=df.columns[0], orientation='h', title=title, 
                 labels={df.columns[1]: '', df.columns[0]: ''}, 
                 color=df.columns[1], color_continuous_scale=px.colors.sequential.Teal, text=df.columns[1])
    fig.update_layout(height=500, title_x=0.5, yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white' if st.get_option("theme.base") == "dark" else "black"))
    return fig

@st.cache_data
def analyze_and_prepare_data(script_dir):
    try:
        dosya_adi = os.path.join(script_dir, "sorumlu.xlsx")
        df = pd.read_excel(dosya_adi, sheet_name='VERİ-2-EMİR', header=0, dtype=str).fillna('')
        sutun_map = {'Kararların Niteliği': 'Karar_Turu', 'Kamu Zararı Var mı?': 'Kamu_Zarari_Durumu', 'Kamu Zararının Sorumlusu Kim?': 'Sorumlular_Metni', 'Kararda Hangi Kanunlara ve Kanun Maddelerine Atıf Yapılmıştır?': 'Kanun_Maddeleri', 'Kararın Konusu Nedir?': 'Karar_Konusu', 'Azınlık Oyu': 'Azinlik_Oyu', 'Daire ilk kararında ısrar etmiş mi?': 'Israr_Durumu'}
        df.rename(columns=sutun_map, inplace=True)
        
        df['Azinlik_Oyu_Temiz'] = df['Azinlik_Oyu'].apply(lambda x: "Var" if str(x).strip().lower() == 'var' else "Yok")
        df['_KamuZarariVar'] = df['Kamu_Zarari_Durumu'].str.contains('Var|Zarar Oluştu', case=False, na=False)
        
        df_karar_turu = df['Karar_Turu'].value_counts().reset_index()
        df_karar_turu.columns = ['Karar Türü', 'Frekans']
        
        df_azinlik_oyu = df['Azinlik_Oyu_Temiz'].value_counts().reset_index()
        df_azinlik_oyu.columns = ['Azınlık Oyu', 'Frekans']
        
        df_karar_konusu = df['Karar_Konusu'].value_counts().reset_index()
        df_karar_konusu.columns = ['Karar Konusu', 'Frekans']

        df_kamu_zarari = df['_KamuZarariVar'].value_counts().rename({True: 'Kamu Zararı Var', False: 'Kamu Zararı Yok'}).reset_index()
        df_kamu_zarari.columns = ['Kamu Zararı', 'Frekans']

        analysis_results = {
            "karar_turu": df_karar_turu,
            "azinlik_oyu": df_azinlik_oyu,
            "karar_konusu": df_karar_konusu,
            "kamu_zarari": df_kamu_zarari,
            "sorumlu_sayilari": None
        }
        
        analiz_listesi = []
        for _, satir in df.dropna(subset=['Sorumlular_Metni']).iterrows():
            unvanlar = cerrahi_analiz_tek_satir(satir['Sorumlular_Metni'])
            for unvan in unvanlar:
                analiz_listesi.append({'Unvan': unvan})
        if analiz_listesi:
            df_sorumlu_sayilari = pd.DataFrame(analiz_listesi)['Unvan'].value_counts().reset_index()
            df_sorumlu_sayilari.columns = ['Unvan', 'Frekans']
            analysis_results["sorumlu_sayilari"] = df_sorumlu_sayilari
            
        return analysis_results
    except Exception as e:
        st.error(f"Veri analizi sırasında bir hata oluştu: {e}")
        st.code(traceback.format_exc())
        return None

# ==============================================================================
# BÖLÜM 3: GENEL UYGULAMA YAPISI VE AYARLAR
# ==============================================================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    # --- DEĞİŞİKLİK: GEÇERLİ GEMINI MODELİ KULLANILDI ---
    gemini_model = genai.GenerativeModel('models/gemini-1.5-pro-latest') 
except Exception as e:
    st.error(f"Gemini API anahtarı yüklenirken bir hata oluştu: {e}")
    gemini_model = None

st.set_page_config(page_title="Hukuki Metin Analizi", layout="wide")

@st.cache_resource
def load_all_models():
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "final_models_combined.pkl")
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"🚨 Tahmin modeli dosyası bulunamadı: '{file_path}'. Lütfen GitHub deponuza yükleyin.")
        return None

@st.cache_data
def load_excel_data():
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SOMUT OLAY-PYHTON.xlsx")
    try:
        df = pd.read_excel(file_path)
        if 'GİRİŞ' not in df.columns or 'Tam Metin' not in df.columns:
            st.error(f"'{os.path.basename(file_path)}' dosyasında 'GİRİŞ' ve/veya 'Tam Metin' sütunları bulunamadı.")
            return None
        return df
    except FileNotFoundError:
        st.error(f"🚨 Özetleme için veri dosyası bulunamadı: '{file_path}'. Lütfen GitHub deponuza yükleyin.")
        return None

models_bundle = load_all_models()
df_data = load_excel_data()

def predict_case(text, law_vec, damage_vec, law_mdl, damage_mdl, classes):
    X_laws = law_vec.transform([text])
    law_prediction_vector = law_mdl.predict(X_laws)[0]
    predicted_laws = [classes[i] for i, val in enumerate(law_prediction_vector) if val == 1]
    X_damage = damage_vec.transform([text])
    damage_prediction_code = damage_mdl.predict(X_damage)[0]
    has_public_damage = "VAR" if damage_prediction_code == 1 else "YOK"
    return predicted_laws, has_public_damage

def find_full_text(df, input_text):
    if df is None or not input_text or not input_text.strip(): return None
    mask = df['GİRİŞ'].str.strip().str.startswith(input_text.strip(), na=False)
    return df.loc[mask, 'Tam Metin'].iloc[0] if mask.any() else None

def get_gemini_summary(text):
    if gemini_model is None: 
        return "Gemini modeli yüklenemediği için özet oluşturulamadı."
    try:
        prompt = f"""Aşağıdaki hukuki metni analiz et ve ana konuyu, tarafların temel argümanlarını ve olayın sonucunu (eğer belirtilmişse) vurgulayan kısa ve anlaşılır bir özet çıkar. Özet, hukuki terimlerden arındırılmış ve herkesin anlayabileceği bir dilde olmalıdır. Metin: "{text}" Özet: """
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini özetleme sırasında bir hata oluştu: {e}"

# ==============================================================================
# BÖLÜM 4: KULLANICI ARAYÜZÜ (STREAMLIT UI)
# ==============================================================================

st.sidebar.title("⚖️ Analiz Platformu")
selected_tool = st.sidebar.radio("Lütfen bir analiz aracını seçin:", 
                                   ("Bireysel Dava Metni Analizi", "Toplu Veri Analizi ve Raporlama"))
st.sidebar.markdown("---")
st.sidebar.info("Bu uygulama, hukuki metinleri analiz etmek ve kapsamlı raporlar oluşturmak için tasarlanmıştır.")

if selected_tool == "Bireysel Dava Metni Analizi":
    
    st.title(selected_tool)
    st.markdown("Girilen dava metninin giriş kısmına göre ilgili **kanunları**, **kamu zararı** durumunu tahmin eder ve metnin tamamını bularak **Gemini AI** ile özetler.")
    if models_bundle is None or df_data is None:
        st.warning("Bireysel analiz aracı için gerekli model veya veri dosyaları yüklenemedi.")
    else:
        law_model, damage_model, vectorizer_laws, vectorizer_damage, mlb_classes = (
            models_bundle['law_model'], models_bundle['damage_model'], 
            models_bundle['vectorizer_laws'], models_bundle['vectorizer_damage'], 
            models_bundle['mlb_classes']
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            input_text = st.text_area("Analiz edilecek metnin başlangıcını girin:", height=250, placeholder="Örnek: Eşi çalışan personele aile yardımı ödeneği ödenmesi...")
            if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
                if input_text.strip():
                    with st.spinner("Analiz yapılıyor..."):
                        st.session_state.laws, st.session_state.damage = predict_case(input_text, vectorizer_laws, vectorizer_damage, law_model, damage_model, mlb_classes)
                        full_text = find_full_text(df_data, input_text)
                        st.session_state.summary = get_gemini_summary(full_text) if full_text else "Girdiğiniz metinle eşleşen bir 'Tam Metin' bulunamadı."
                        st.session_state.ran_prediction = True
                else:
                    st.warning("Lütfen analiz için bir metin girin.")
        with col2:
            st.subheader("📊 Analiz Sonuçları")
            if st.session_state.get('ran_prediction', False):
                st.markdown("##### 📘 İlgili Kanunlar:")
                if st.session_state.laws:
                    for k in st.session_state.laws: st.success(f"- {k}")
                else:
                    st.info("İlişkili bir kanun bulunamadı.")
                st.markdown("---")
                st.markdown("##### 💸 Kamu Zararı Durumu:")
                if st.session_state.damage == "VAR":
                    st.error(f"**{st.session_state.damage}**")
                else:
                    st.info(f"**{st.session_state.damage}**")
                st.markdown("---")
                st.markdown("##### 🤖 Gemini AI Metin Özeti:")
                with st.expander("Özeti Göster", expanded=True):
                    st.info(st.session_state.summary)
            else:
                st.info("Sonuçları görmek için bir metin girip 'Analiz Et' butonuna tıklayın.")

elif selected_tool == "Toplu Veri Analizi ve Raporlama":
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    results = analyze_and_prepare_data(script_dir)

    if results:
        st.header("📊 Analiz Sonuçları ve Görseller")
        
        st.markdown("#### Karar Türü Dağılımı")
        col1, col2 = st.columns([2, 1.2])
        with col1:
            fig_karar_turu = create_plotly_pie(results['karar_turu'], "Karar Türü Dağılımı")
            if fig_karar_turu: st.plotly_chart(fig_karar_turu, use_container_width=True)
        with col2:
            st.table(results['karar_turu'])

        st.markdown("#### Kamu Zararı Dağılımı")
        col1, col2 = st.columns([2, 1.2])
        with col1:
            fig_kamu_zarari = create_plotly_pie(results['kamu_zarari'], "Kamu Zararı Dağılımı")
            if fig_kamu_zarari: st.plotly_chart(fig_kamu_zarari, use_container_width=True)
        with col2:
            st.table(results['kamu_zarari'])
        
        st.markdown("#### Azınlık Oyu Dağılımı")
        col1, col2 = st.columns([2, 1.2])
        with col1:
            fig_azinlik_oyu = create_plotly_pie(results['azinlik_oyu'], "Azınlık Oyu Dağılımı")
            if fig_azinlik_oyu: st.plotly_chart(fig_azinlik_oyu, use_container_width=True)
        with col2:
            st.table(results['azinlik_oyu'])
        
        st.markdown("---")
        st.markdown("#### Karar Konuları")
        col1, col2 = st.columns([2, 1.2])
        with col1:
            fig_konu = create_plotly_bar(results['karar_konusu'], "Karar Konuları")
            if fig_konu: st.plotly_chart(fig_konu, use_container_width=True)
        with col2:
            st.table(results['karar_konusu'].head(15))

        st.markdown("---")
        st.markdown("#### Sorumlu Unvanlar")
        col1, col2 = st.columns([2, 1.2])
        with col1:
            if results['sorumlu_sayilari'] is not None:
                fig_sorumlu = create_plotly_bar(results['sorumlu_sayilari'], "Sorumlu Unvanlar")
                if fig_sorumlu: st.plotly_chart(fig_sorumlu, use_container_width=True)
            else:
                st.info("Sorumlu unvan analizi için veri bulunamadı.")
        with col2:
            if results['sorumlu_sayilari'] is not None:
                st.table(results['sorumlu_sayilari'].head(15))
                
    else:
        st.error("Analiz verileri yüklenemedi. Lütfen 'sorumlu.xlsx' dosyasının formatını ve içeriğini kontrol edin.")
