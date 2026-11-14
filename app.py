import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Enterprise Churn AI | Müşteri Kayıp Tahmini",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# SADE, RENKLİ VE AÇIKLAYICI KURUMSAL UI TASARIMI
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 8px 20px rgba(49, 46, 129, 0.25);
    }
    .main-title {font-size: 2.1rem; font-weight: 800; margin: 0 0 8px 0; color: #FFFFFF;}
    .main-sub {font-size: 0.98rem; margin: 0 0 16px 0; color: #E0E7FF; font-weight: 400; line-height: 1.5;}
    .contact-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        padding-top: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
    .pill-badge {
        background: rgba(255, 255, 255, 0.15);
        color: #FFFFFF !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-decoration: none !important;
        border: 1px solid rgba(255, 255, 255, 0.25);
        transition: background 0.3s;
    }
    .pill-badge:hover {background: rgba(255, 255, 255, 0.25);}
    
    /* Rehber Bilgi Kutuları */
    .guide-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 8px;
        color: #334155;
        font-size: 0.92rem;
        margin-bottom: 18px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    
    /* Renkli KPI Metrik Kartları */
    .kpi-container {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        flex: 1;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: left;
    }
    .kpi-card.blue {background: #EFF6FF; border-top: 4px solid #3B82F6;}
    .kpi-card.green {background: #F0FDF4; border-top: 4px solid #10B981;}
    .kpi-card.red {background: #FEF2F2; border-top: 4px solid #EF4444;}
    .kpi-card.purple {background: #FAF5FF; border-top: 4px solid #8B5CF6;}
    
    .kpi-title {font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 8px; text-transform: uppercase;}
    .kpi-value {font-size: 1.8rem; font-weight: 800; color: #0F172A; margin: 0;}
    .kpi-icon {font-size: 1.4rem; float: right; opacity: 0.7;}
</style>
""", unsafe_allow_html=True)

# Üst Başlık (Hero Section)
st.markdown("""
<div class="main-header">
    <div class="main-title">📊 Kurumsal Müşteri Analitiği (RFM) & AI Kayıp (Churn) Tahmin Sistemi</div>
    <div class="main-sub">
        Bu platform, müşteri tabanını RFM metriklerine göre segmente eder ve <b>Random Forest Makine Öğrenmesi</b> modeliyle hangi müşterilerin şirketi terk etme (Churn) riski taşıdığını önceden tahmin ederek otonom aksiyonlar üretir.
    </div>
    <div class="contact-bar">
        <span class="pill-badge">👨‍💻 Lead Data Scientist: Olgun Dağdelen</span>
        <a class="pill-badge" href="mailto:olgundagdelen46@gmail.com">📧 olgundagdelen46@gmail.com</a>
        <a class="pill-badge" href="tel:+905439322372">📞 +90 543 932 23 72</a>
        <a class="pill-badge" href="https://wa.me/905439322372" target="_blank">💬 WhatsApp Doğrudan İletişim</a>
        <span class="pill-badge">🎓 Danışman: Öğr. Gör. Ümit Bulut</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. VERİ ÜRETİMİ VE MAKİNE ÖĞRENMESİ (AI) BEYNİ
# ==============================================================================
@st.cache_data
def generate_and_train_crm_data(n_customers=2000):
    np.random.seed(42)
    
    # Gerçekçi sentetik müşteri (CRM) verisi oluşturma
    df = pd.DataFrame({
        "Musteri_ID": [f"CRM-{10000+i}" for i in range(n_customers)],
        "Yenilik_Gun_R": np.random.randint(1, 365, size=n_customers),
        "Siklik_Adet_F": np.random.randint(1, 50, size=n_customers),
        "Parasal_Deger_M_TL": np.random.randint(500, 25000, size=n_customers),
        "Musteri_Yasi_Ay": np.random.randint(1, 72, size=n_customers),
        "Son_1_Ay_Destek_Talebi": np.random.poisson(lam=1.5, size=n_customers)
    })
    
    # RFM Skorlaması ve Segmentasyon
    df['R_Skor'] = pd.qcut(df['Yenilik_Gun_R'], 4, labels=[4, 3, 2, 1])
    df['F_Skor'] = pd.qcut(df['Siklik_Adet_F'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    df['M_Skor'] = pd.qcut(df['Parasal_Deger_M_TL'], 4, labels=[1, 2, 3, 4])
    df['RFM_Skoru'] = df['R_Skor'].astype(str) + df['F_Skor'].astype(str)
    
    seg_map = {
        r'[1-2][1-2]': '💤 Uykuya Dalanlar (Hibernating)',
        r'[1-2][3-4]': '⚠️ Kaybedilmemesi Gerekenler (At Risk)',
        r'[3-4][1-2]': '🌱 Yeni/Potansiyel Müşteriler',
        r'3[3-4]': '💎 Sadık Müşteriler (Loyal)',
        r'4[3-4]': '🏆 Şampiyonlar (Champions)'
    }
    df['Segment'] = df['R_Skor'].astype(str) + df['F_Skor'].astype(str)
    df['Segment'] = df['Segment'].replace(seg_map, regex=True)
    
    # Churn (Kayıp) Etiketini Mantıklı Şekilde Belirleme
    # Çok şikayet eden, uzun süredir gelmeyen ve şirkette yeni olanlar daha çabuk ayrılır (Churn = 1)
    churn_prob = (
        (df['Yenilik_Gun_R'] / 365) * 0.4 +
        (df['Son_1_Ay_Destek_Talebi'] / 5) * 0.35 -
        (df['Musteri_Yasi_Ay'] / 72) * 0.25
    )
    df['Gercek_Churn'] = (churn_prob + np.random.normal(0, 0.1, n_customers) > 0.45).astype(int)
    
    # Makine Öğrenmesi (Random Forest) Model Eğitimi
    features = ["Yenilik_Gun_R", "Siklik_Adet_F", "Parasal_Deger_M_TL", "Musteri_Yasi_Ay", "Son_1_Ay_Destek_Talebi"]
    X = df[features]
    y = df["Gercek_Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return df, model, accuracy, features

# Veriyi ve Modeli Arka Planda Yükle
df_crm, rf_model, model_acc, feature_cols = generate_and_train_crm_data()

# ==============================================================================
# 2. RENKLİ KPI KARTLARI (ÜST BÖLÜM)
# ==============================================================================
toplam_musteri = len(df_crm)
toplam_ciro = df_crm["Parasal_Deger_M_TL"].sum()
churn_orani = (df_crm["Gercek_Churn"].mean()) * 100

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card blue">
        <div class="kpi-icon">👥</div>
        <div class="kpi-title">Toplam Aktif Müşteri</div>
        <div class="kpi-value">{toplam_musteri:,}</div>
    </div>
    <div class="kpi-card green">
        <div class="kpi-icon">💰</div>
        <div class="kpi-title">Toplam Ciro Hacmi</div>
        <div class="kpi-value">₺{toplam_ciro:,.0f}</div>
    </div>
    <div class="kpi-card red">
        <div class="kpi-icon">🚨</div>
        <div class="kpi-title">Ortalama Churn (Kayıp) Oranı</div>
        <div class="kpi-value">%{churn_orani:.1f}</div>
    </div>
    <div class="kpi-card purple">
        <div class="kpi-icon">🤖</div>
        <div class="kpi-title">AI Model Doğruluk Başarısı</div>
        <div class="kpi-value">%{model_acc*100:.1f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. YAN MENÜ (SIDEBAR) BİLGİ VE İNDİRME ALANI
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="background:#F8FAFC; border-left:4px solid #4338CA; padding:14px; border-radius:8px; margin-bottom:15px;">
        <b style="color:#312E81;">👨‍💻 Geliştirici Ekip</b><br>
        <span style="font-size:1.0rem; font-weight:700; color:#0F172A;">Olgun Dağdelen</span><br>
        <small>Ahmet Emek | İbrahim Taştekin | Emir Kıdrez</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📥 Veri Setini İndir")
    csv = df_crm.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📊 Temizlenmiş CRM Verisini İndir (.CSV)",
        data=csv,
        file_name='kurumsal_crm_churn_verisi.csv',
        mime='text/csv',
        use_container_width=True
    )
    
    st.divider()
    st.caption("Bu proje Random Forest Sınıflandırma algoritması kullanılarak geliştirilmiştir.")

# ==============================================================================
# 4. FONKSİYONEL VE AÇIKLAMALI SEKMELER
# ==============================================================================
tabs = st.tabs([
    "📈 RFM Segmentasyon & Grafik Analizi",
    "🤖 Canlı AI Müşteri Kaybı (Churn) Simülatörü",
    "🎯 Riskli Müşteriler & Otonom Aksiyon Listesi"
])

# ------------------------------------------------------------------------------
# SEKME 1: RFM SEGMENTASYON VE RENKLİ GRAFİKLER
# ------------------------------------------------------------------------------
with tabs[0]:
    st.markdown("""
    <div class="guide-box">
        <b>ℹ️ Bu Bölüm Ne İşe Yarar? (RFM Analizi)</b><br>
        Müşterileri <b>R</b>ecency (Yenilik), <b>F</b>requency (Sıklık) ve <b>M</b>onetary (Parasal Değer) metriklerine göre gruplar. Şirkete en çok para kazandıran <i>"Şampiyonları"</i> ve uzun süredir uğramayan <i>"Uykuya Dalanları"</i> canlı grafiklerle görselleştirir.
    </div>
    """, unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_pie = px.pie(
            df_crm, names='Segment', title='Müşteri Segment Dağılımı (%)',
            hole=0.45, color='Segment',
            color_discrete_map={
                '🏆 Şampiyonlar (Champions)': '#10B981',
                '💎 Sadık Müşteriler (Loyal)': '#3B82F6',
                '🌱 Yeni/Potansiyel Müşteriler': '#8B5CF6',
                '⚠️ Kaybedilmemesi Gerekenler (At Risk)': '#F59E0B',
                '💤 Uykuya Dalanlar (Hibernating)': '#EF4444'
            }
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_g2:
        segment_ciro = df_crm.groupby('Segment')['Parasal_Deger_M_TL'].mean().reset_index()
        fig_bar = px.bar(
            segment_ciro, x='Segment', y='Parasal_Deger_M_TL',
            title='Segmentlere Göre Ortalama Bırakılan Ciro (TL)',
            color='Segment', text_auto='.0f',
            color_discrete_map={
                '🏆 Şampiyonlar (Champions)': '#10B981',
                '💎 Sadık Müşteriler (Loyal)': '#3B82F6',
                '🌱 Yeni/Potansiyel Müşteriler': '#8B5CF6',
                '⚠️ Kaybedilmemesi Gerekenler (At Risk)': '#F59E0B',
                '💤 Uykuya Dalanlar (Hibernating)': '#EF4444'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------------------------------------------
# SEKME 2: CANLI CHURN SİMÜLATÖRÜ
# ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("""
    <div class="guide-box">
        <b>ℹ️ Bu Bölüm Ne İşe Yarar? (AI Churn Tahmincisi)</b><br>
        Yapay zekanın nasıl çalıştığını test etmenizi sağlar. Sol tarafta bir müşterinin özelliklerini (fatura, şikayet sayısı vb.) değiştirdikçe, sağ tarafta modelin o müşterinin <b>şirketi terk etme ihtimalini (%)</b> nasıl hesapladığını canlı görebilirsiniz.
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1.2])
    
    with sim_col1:
        st.markdown("#### ⚙️ Müşteri Özelliklerini Belirleyin")
        sim_R = st.slider("Son Alışverişten Geçen Gün (Recency):", 1, 365, 120)
        sim_F = st.slider("Toplam Alışveriş Sayısı (Frequency):", 1, 50, 10)
        sim_M = st.number_input("Toplam Harcama / Fatura (TL):", 500, 25000, 4500, step=500)
        sim_Yas = st.slider("Şirketteki Üyelik Yaşı (Ay):", 1, 72, 24)
        sim_Destek = st.slider("Son 1 Ayda Açtığı Şikayet/Destek Sayısı:", 0, 10, 2)
        
        sim_veri = pd.DataFrame([[sim_R, sim_F, sim_M, sim_Yas, sim_Destek]], columns=feature_cols)
        churn_olasilik = rf_model.predict_proba(sim_veri)[0][1] * 100

    with sim_col2:
        st.markdown("#### 🤖 Yapay Zeka (Random Forest) Kararı")
        
        if churn_olasilik >= 60:
            st.error(f"### 🚨 YÜKSEK RİSK: Şirketi Terk Etme İhtimali %{churn_olasilik:.1f}")
            st.warning("**Yapay Zeka Yorumu:** Bu müşteri uzun süredir işlem yapmamış ve son dönemde çok fazla destek talebi açmış. Derhal iletişime geçilmeli ve memnuniyet anketi yapılmalı!")
        elif churn_olasilik >= 35:
            st.warning(f"### ⚠️ ORTA RİSK: Şirketi Terk Etme İhtimali %{churn_olasilik:.1f}")
            st.info("**Yapay Zeka Yorumu:** Müşteri kararsız bölgede. Ufak bir sadakat indirimi veya çapraz kampanya tanımlanarak platforma tekrar ısındırılabilir.")
        else:
            st.success(f"### ✅ GÜVENLİ: Şirketi Terk Etme İhtimali %{churn_olasilik:.1f}")
            st.success("**Yapay Zeka Yorumu:** Müşterinin platforma bağlılığı oldukça yüksek. Herhangi bir müdahaleye gerek yok, sadık bir profil sergiliyor.")

        # Özellik Önemi (Feature Importance) Grafiği
        imp_df = pd.DataFrame({'Özellik': ["Yenilik (Gün)", "Sıklık (Adet)", "Harcama (TL)", "Üyelik Yaşı (Ay)", "Şikayet Sayısı"], 'Önem Yüzdesi': rf_model.feature_importances_ * 100})
        fig_imp = px.bar(imp_df.sort_values('Önem Yüzdesi'), x='Önem Yüzdesi', y='Özellik', orientation='h', title="Model Karar Verirken En Çok Neye Baktı?")
        fig_imp.update_layout(height=280)
        st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------------------------------------------
# SEKME 3: RİSKLİ MÜŞTERİLER VE OTONOM AKSİYON LİSTESİ
# ------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("""
    <div class="guide-box">
        <b>ℹ️ Bu Bölüm Ne İşe Yarar? (Otonom Aksiyon Motoru)</b><br>
        Tüm CRM veri tabanını yapay zekadan geçirir ve <b>ayrılma (churn) riski en yüksek olan</b> müşterileri listeler. Üstelik boş durmaz, her müşteri için <i>"%15 İndirim Yap"</i> veya <i>"VIP Müşteri Temsilcisi Ata"</i> gibi pazarlama aksiyonları önerir.
    </div>
    """, unsafe_allow_html=True)

    # Tüm veriyi modelden geçirip olasılıkları hesapla
    df_crm['Tahmini_Kacis_Ihtimali_Yuzde'] = rf_model.predict_proba(df_crm[feature_cols])[:, 1] * 100
    
    # Sadece riski yüksek olanları filtrele (Örn: %60'tan büyük olanlar)
    riskli_musteriler = df_crm[df_crm['Tahmini_Kacis_Ihtimali_Yuzde'] >= 60].sort_values('Tahmini_Kacis_Ihtimali_Yuzde', ascending=False).head(15).copy()
    
    # Her riskli müşteri için AI Aksiyonu belirle
    def aksiyon_belirle(row):
        if row['Parasal_Deger_M_TL'] > 15000:
            return "🎯 VIP Temsilci Ataması Yap ve Ücretsiz Üst Paket Tanımla"
        elif row['Son_1_Ay_Destek_Talebi'] >= 4:
            return "📞 Acil Arama Yap ve Şikayetini Çöz (Öncelikli Bilet)"
        elif row['Musteri_Yasi_Ay'] > 36:
            return "🎁 Vefa Kampanyası: %25 Sadakat İndirimi Kuponu Yolla"
        else:
            return "✉️ Otomatik Geri Kazanım (Win-back) E-Postası Gönder"

    riskli_musteriler['AI_Otonom_Aksiyon_Onerisi'] = riskli_musteriler.apply(aksiyon_belirle, axis=1)
    
    gosterilecek_kolonlar = ['Musteri_ID', 'Tahmini_Kacis_Ihtimali_Yuzde', 'Segment', 'Parasal_Deger_M_TL', 'Son_1_Ay_Destek_Talebi', 'AI_Otonom_Aksiyon_Onerisi']
    
    st.markdown("#### 🚨 Acil Müdahale Gerektiren İlk 15 Müşteri")
    # Tabloyu formatlayarak şık gösterme
    st.dataframe(
        riskli_musteriler[gosterilecek_kolonlar].style.background_gradient(subset=['Tahmini_Kacis_Ihtimali_Yuzde'], cmap='Reds')
                                                      .format({'Tahmini_Kacis_Ihtimali_Yuzde': '{:.1f}%', 'Parasal_Deger_M_TL': '₺{:,.0f}'}),
        use_container_width=True,
        height=400
    )
