import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Kurumsal Müşteri Zekası & AI Churn Paneli", layout="wide")

st.title("📊 Kurumsal Müşteri Analitiği (RFM) & AI Kayıp (Churn) Tahmin Sistemi")
st.markdown("Bu platform, müşteri tabanını **RFM metriklerine** göre segmente eder ve **Random Forest Makine Öğrenmesi** modeliyle hangi müşterilerin şirketi terk etme (Churn) riski taşıdığını tahmin ederek otonom aksiyonlar üretir.")
st.divider()

@st.cache_data
def load_data():
    return pd.read_csv('musteri_verisi.csv')

df = load_data()

@st.cache_resource
def train_model(data):
    X = data[['Recency', 'Frequency', 'Monetary', 'SupportTickets']]
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

model, model_acc = train_model(df)

# Üst KPI Metrik Kartları
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Toplam Aktif Müşteri", f"{len(df):,}")
col2.metric("💰 Toplam Ciro Hacmi", f"₺{df['Monetary'].sum():,.0f}")
col3.metric("🚨 Ortalama Churn (Kayıp) Oranı", f"%{df['Churn'].mean()*100:.1f}")
col4.metric("🤖 AI Model Doğruluk Başarısı", f"%{model_acc*100:.1f}")

st.divider()

# Sekmeli (Tab) Kurumsal Mimari
tab1, tab2, tab3 = st.tabs(["📈 RFM Segmentasyon & Grafik Analizi", "🔮 Canlı AI Müşteri Kaybı (Churn) Simülatörü", "🎯 Riskli Müşteriler & Aksiyon Listesi"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Müşteri Segment Dağılımı")
        fig_pie = px.pie(df, names='Segment', hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("Segmentlere Göre Ortalama Ciro (TL)")
        seg_rev = df.groupby('Segment', as_index=False)['Monetary'].mean()
        fig_bar = px.bar(seg_rev, x='Segment', y='Monetary', color='Segment', text_auto='.2s')
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("🤖 Tekil Müşteri Terk Etme (Churn) Riski Hesaplama")
    st.info("Aşağıdaki parametreleri değiştirerek herhangi bir müşterinin şirketi terk etme olasılığını canlı olarak test edin.")
    
    sc1, sc2 = st.columns(2)
    with sc1:
        in_recency = st.slider("📅 Son Alışverişten Geçen Süre (Gün - Recency)", 1, 365, 120)
        in_freq = st.slider("🛒 Toplam Sipariş Sayısı (Frequency)", 1, 50, 8)
    with sc2:
        in_monetary = st.number_input("💳 Toplam Harcama Tutarı (TL - Monetary)", min_value=100, max_value=100000, value=4500, step=500)
        in_tickets = st.slider("📞 Müşteri Hizmetleri Şikayet/Destek Kaydı Sayısı", 0, 10, 3)
        
    if st.button("🚀 Yapay Zeka İle Risk Analizi Yap"):
        input_data = pd.DataFrame([[in_recency, in_freq, in_monetary, in_tickets]], 
                                  columns=['Recency', 'Frequency', 'Monetary', 'SupportTickets'])
        prob = model.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        if prob > 0.65:
            st.error(f"🚨 **YÜKSEK RİSK! Müşterinin Terk Etme Olasılığı: %{prob*100:.1f}**")
            st.warning("💡 **Otonom Karar Önerisi:** Bu müşteriye anında **%20 Özel Geri Kazanım İndirim Kodu** SMS'i gönderilmeli ve müşteri temsilcisi tarafından aranmalıdır.")
        elif prob > 0.35:
            st.warning(f"⚠️ **ORTA RİSK! Müşterinin Terk Etme Olasılığı: %{prob*100:.1f}**")
            st.info("💡 **Otonom Karar Önerisi:** Müşteriye yeni sezon ürünlerini tanıtan hatırlatma e-postası ve kargo bedava kuponu tanımlanmalıdır.")
        else:
            st.success(f"✅ **GÜVENLİ (SADIK MÜŞTERİ)! Terk Etme Olasılığı Sadece: %{prob*100:.1f}**")
            st.info("💡 **Otonom Karar Önerisi:** İndirim vermeye gerek yok. VIP sadakat programına dahil edilerek çapraz satış (Cross-Sell) yapılabilir.")

with tab3:
    st.subheader("🚨 Acil Müdahale Gereken Yüksek Ciroya Sahip Riskli Müşteriler")
    risk_df = df[(df['Churn'] == 1) & (df['Monetary'] > 15000)].sort_values(by='Monetary', ascending=False)
    st.dataframe(risk_df[['CustomerID', 'Segment', 'Recency', 'Frequency', 'Monetary', 'SupportTickets']].head(20), use_container_width=True, hide_index=True)
