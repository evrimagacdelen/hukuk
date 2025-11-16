import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import google.generativeai as genai
import re
import matplotlib
matplotlib.use('Agg') # Streamlit Cloud üzerinde uyumluluk için
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image
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
# BÖLÜM 2: EXCEL RAPORLAMA VE GÖRSELLEŞTİRME İÇİN FONKSİYONLAR
# ==============================================================================
def cerrahi_analiz_tek_satir(metin):
    # ... (Bu fonksiyonda değişiklik yok) ...
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
    potansiyel_roller = re.split(r'[,/()]|\s+ve\s+|\s+ile\s+', kalan_metin)
    for parca in potansiyel_roller:
        temiz_parca = parca.strip().lower()
        if temiz_parca in NORM_MAP: roller_bu_satirda.add(NORM_MAP[temiz_parca])
        elif 'vekili' in temiz_parca or temiz_parca.endswith((' v', ' v.')):
            if 'dekan' in temiz_parca: roller_bu_satirda.add('Dekan Vekili')
            elif 'rektör' in temiz_parca: roller_bu_satirda.add('Rektör Vekili')
    return list(roller_bu_satirda)

def create_pie_chart(data, title):
    if data.empty:
        return None, False
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, 
           wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 10})
    ax.set_title(title, fontsize=14, pad=20, weight='bold')
    plt.axis('equal')
    return fig, True

def create_bar_plot(data, title, top_n=15):
    if data.empty:
        return None, False
    data_to_plot = data.head(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(data_to_plot.index, data_to_plot.values, color='skyblue')
    ax.set_title(title, fontsize=16, pad=20, weight='bold')
    ax.set_xlabel("Karar Sayısı")
    plt.tight_layout()
    return fig, True

def analyze_and_prepare_data(script_dir):
    try:
        dosya_adi = os.path.join(script_dir, "sorumlu.xlsx")
        df = pd.read_excel(dosya_adi, sheet_name='VERİ-2-EMİR', header=0, dtype=str).fillna('')
        st.info(f"'{os.path.basename(dosya_adi)}' dosyasından {len(df)} satır veri bulundu.")
        sutun_map = {'Kararların Niteliği': 'Karar_Turu', 'Kamu Zararı Var mı?': 'Kamu_Zarari_Durumu', 'Kamu Zararının Sorumlusu Kim?': 'Sorumlular_Metni', 'Kararda Hangi Kanunlara ve Kanun Maddelerine Atıf Yapılmıştır?': 'Kanun_Maddeleri', 'Kararın Konusu Nedir?': 'Karar_Konusu', 'Azınlık Oyu': 'Azinlik_Oyu', 'Daire ilk kararında ısrar etmiş mi?': 'Israr_Durumu'}
        df.rename(columns=sutun_map, inplace=True)
        df['Azinlik_Oyu'] = df['Azinlik_Oyu'].str.strip()
        df['Israr_Durumu'] = df['Israr_Durumu'].str.strip()
        df['_KamuZarariVar'] = df['Kamu_Zarari_Durumu'].str.contains('Var|Zarar Oluştu', case=False, na=False)
        df['_AzinlikOyuVar'] = df['Azinlik_Oyu'].str.upper() == 'VAR'
        df['_IsrarVar'] = df['Israr_Durumu'] != ''
        st.info("Veri temizlendi ve yardımcı analiz sütunları oluşturuldu.")
        
        analysis_results = {
            "karar_turu": df['Karar_Turu'].value_counts(),
            "azinlik_oyu": df['Azinlik_Oyu'].value_counts(),
            "karar_konusu": df['Karar_Konusu'].value_counts(),
            "kamu_zarari": df['_KamuZarariVar'].value_counts().rename({True: 'Kamu Zararı Var', False: 'Kamu Zararı Yok'}),
            "unvan_analizi": None
        }
        
        analiz_listesi = []
        for _, satir in df.dropna(subset=['Sorumlular_Metni']).iterrows():
            unvanlar = cerrahi_analiz_tek_satir(satir['Sorumlular_Metni'])
            for unvan in unvanlar:
                analiz_listesi.append({'Unvan': unvan, 'Zarar_Durumu': satir['_KamuZarariVar']})
        if analiz_listesi:
            ozet_tablo_unvan = pd.DataFrame(analiz_listesi).groupby('Unvan')['Zarar_Durumu'].value_counts().unstack(fill_value=0).rename(columns={True:'Kamu Zararı Var', False:'Kamu Zararı Yok'})
            if 'Kamu Zararı Var' not in ozet_tablo_unvan: ozet_tablo_unvan['Kamu Zararı Var'] = 0
            if 'Kamu Zararı Yok' not in ozet_tablo_unvan: ozet_tablo_unvan['Kamu Zararı Yok'] = 0
            ozet_tablo_unvan['Toplam'] = ozet_tablo_unvan.sum(axis=1)
            ozet_tablo_unvan['KZ Oranı %'] = ((ozet_tablo_unvan['Kamu Zararı Var'] / ozet_tablo_unvan['Toplam']) * 100).round(1)
            analysis_results["unvan_analizi"] = ozet_tablo_unvan.sort_values(by='Toplam', ascending=False)
            
        return analysis_results
    except Exception as e:
        st.error(f"Veri analizi sırasında bir hata oluştu: {e}")
        st.code(traceback.format_exc())
        return None

def generate_excel_report(analysis_results):
    chart_files_to_delete = []
    try:
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            st.info("İndirilebilir Excel raporu oluşturuluyor...")
            
            # SEKME 1: GENEL ÖZETLER
            analysis_results['karar_turu'].to_excel(writer, sheet_name='Genel_Ozetler', header=['Sayı'], startrow=1)
            analysis_results['kamu_zarari'].to_excel(writer, sheet_name='Genel_Ozetler', header=['Sayı'], startcol=3)
            analysis_results['azinlik_oyu'].to_excel(writer, sheet_name='Genel_Ozetler', header=['Sayı'], startcol=6)

            # SEKME 2: KARAR KONUSU DETAYLARI
            analysis_results['karar_konusu'].to_excel(writer, sheet_name='Karar_Konusu_Detaylari', header=['Sayı'], startrow=1)

            # SEKME 3: UNVAN ANALİZİ (Sütunlar ayarlanarak yazılır)
            if analysis_results['unvan_analizi'] is not None:
                df_unvan_for_excel = analysis_results['unvan_analizi'].drop(columns=['Toplam', 'Kamu Zararı Yok', 'KZ Oranı %'], errors='ignore')
                df_unvan_for_excel.to_excel(writer, sheet_name='Unvan_Kamu_Zarari_Analizi')
        
        return output_buffer.getvalue()

    except Exception as e:
        st.error(f"Excel raporu oluşturulurken bir hata oluştu: {e}")
        st.code(traceback.format_exc())
        return None
    finally:
        st.info("Geçici dosyalar temizleniyor...")
        for f in chart_files_to_delete:
            if os.path.exists(f):
                os.remove(f)

# ==============================================================================
# BÖLÜM 3: GENEL UYGULAMA YAPISI VE AYARLAR
# ==============================================================================
# ... (Bu bölümde değişiklik yok) ...
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
except Exception as e:
    st.error(f"Gemini API anahtarı yüklenirken bir hata oluştu: {e}")
    gemini_model = None

st.set_page_config(page_title="Hukuki Metin Analizi", layout="wide")

st.title("⚖️ Gelişmiş Hukuki Analiz Platformu")
st.markdown("Bu platform, iki ana araç sunar: **Bireysel Metin Analizi** ve **Toplu Veri Raporlama**.")
st.markdown("---")

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
    if gemini_model is None: return "Gemini modeli yüklenemediği için özet oluşturulamadı."
    try:
        prompt = f"""Aşağıdaki hukuki metni analiz et ve ana konuyu, tarafların temel argümanlarını ve olayın sonucunu (eğer belirtilmişse) vurgulayan kısa ve anlaşılır bir özet çıkar. Özet, hukuki terimlerden arındırılmış ve herkesin anlayabileceği bir dilde olmalıdır. Metin: "{text}" Özet: """
        response = genai.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini özetleme sırasında bir hata oluştu: {e}"

# ==============================================================================
# BÖLÜM 4: KULLANICI ARAYÜZÜ (STREAMLIT UI) - SİDEBAR YAPISI
# ==============================================================================

# Sidebar'da araç seçimi
selected_tool = st.sidebar.radio("Lütfen bir analiz aracını seçin:", 
                                   ("Bireysel Dava Metni Analizi", "Toplu Veri Analizi ve Raporlama"))

st.sidebar.markdown("---")
st.sidebar.title("ℹ️ Hakkında")
st.sidebar.info("Bu uygulama, hukuki metinleri analiz etmek, kanunları tahmin etmek ve kapsamlı raporlar oluşturmak için tasarlanmıştır.")

# Ana içeriği seçilen araca göre dinamik olarak göster
if selected_tool == "Bireysel Dava Metni Analizi":
    
    st.header("1. Bireysel Dava Metni Analizi")
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
                st.error(f"**{st.session_state.damage}**") if st.session_state.damage == "VAR" else st.info(f"**{st.session_state.damage}**")
                
                st.markdown("---")
                st.markdown("##### 🤖 Gemini AI Metin Özeti:")
                with st.expander("Özeti Göster", expanded=True):
                    st.info(st.session_state.summary)
            else:
                st.info("Sonuçları görmek için bir metin girip 'Analiz Et' butonuna tıklayın.")

elif selected_tool == "Toplu Veri Analizi ve Raporlama":
    
    st.header("2. Toplu Veri Analizi ve Raporlama")
    st.markdown("`sorumlu.xlsx` dosyasını kullanarak kapsamlı bir analiz yapar, sonuçları aşağıda gösterir ve tam raporu indirilebilir bir Excel dosyası olarak sunar.")

    if st.button("📊 Kapsamlı Analiz Yap ve Göster", use_container_width=True):
        with st.spinner("Analiz yapılıyor ve görseller hazırlanıyor..."):
            script_dir = os.path.dirname(os.path.realpath(__file__))
            analysis_data = analyze_and_prepare_data(script_dir)
            if analysis_data:
                st.session_state.analysis_results = analysis_data
                report_file = generate_excel_report(analysis_data)
                if report_file:
                    st.session_state.report_data = report_file
                    st.success("✅ Analiz tamamlandı! Sonuçları aşağıda görebilir ve tam raporu indirebilirsiniz.")

    if 'report_data' in st.session_state:
        st.download_button(
            label="📥 Tam Analiz Raporunu İndir (.xlsx)",
            data=st.session_state.report_data,
            file_name="Vaaaov_Analiz_Raporu.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if 'analysis_results' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Analiz Sonuçları ve Görseller")
        
        results = st.session_state.analysis_results
        
        # Genel Dağılımlar
        st.markdown("#### Genel Karar Dağılımları")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Karar Türü Dağılımı**")
            fig, success = create_pie_chart(results['karar_turu'], "Karar Türü")
            if success: st.pyplot(fig)
        with col2:
            st.write("**Kamu Zararı Dağılımı**")
            fig, success = create_pie_chart(results['kamu_zarari'], "Kamu Zararı")
            if success: st.pyplot(fig)
        with col3:
            st.write("**Azınlık Oyu Dağılımı**")
            fig, success = create_pie_chart(results['azinlik_oyu'], "Azınlık Oyu")
            if success: st.pyplot(fig)

        # Karar Konusu Analizi - YENİ BÖLÜM
        st.markdown("---")
        st.markdown("#### En Sık Görülen Karar Konuları")
        fig, success = create_bar_plot(results['karar_konusu'], "En Sık Görülen 15 Karar Konusu", top_n=15)
        if success:
            st.pyplot(fig)
        
        with st.expander("Tüm Karar Konularını ve Sayılarını Gör"):
            st.dataframe(results['karar_konusu'])

        # Unvan Analizi
        st.markdown("---")
        st.markdown("#### Unvanlara Göre Kamu Zararı Analizi")
        if results['unvan_analizi'] is not None:
            st.dataframe(results['unvan_analizi'])
        else:
            st.info("Unvan analizi için veri bulunamadı.")
