import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import datetime

st.set_page_config(
    page_title="Kurumsal Müşteri Zekası & AI Churn Platformu v3.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# RENKLİ, SADE VE AÇIKLAYICI KURUMSAL UI TASARIMI (v3.0)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #4338CA 50%, #0EA5E9 100%);
        padding: 26px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 12px 28px rgba(67, 56, 202, 0.25);
    }
    .hero-title {font-size: 2.1rem; font-weight: 800; margin: 0 0 8px 0; color: #FFFFFF;}
    .hero-sub {font-size: 0.96rem; margin: 0 0 16px 0; color: #E0F2FE; line-height: 1.5;}
    .contact-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding-top: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.22);
    }
    .badge-item {
        background: rgba(255, 255, 255, 0.16);
        color: #FFFFFF !important;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.83rem;
        font-weight: 600;
        text-decoration: none !important;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }

    /* 6 Renkli KPI Kartı */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 14px;
        margin-bottom: 22px;
    }
    .kpi-box {
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.04);
    }
    .kpi-box.c1 {background: linear-gradient(135deg, #EFF6FF, #DBEAFE); border-top: 4px solid #2563EB;}
    .kpi-box.c2 {background: linear-gradient(135deg, #ECFDF5, #D1FAE5); border-top: 4px solid #059669;}
    .kpi-box.c3 {background: linear-gradient(135deg, #FEF2F2, #FEE2E2); border-top: 4px solid #DC2626;}
    .kpi-box.c4 {background: linear-gradient(135deg, #FFFBEB, #FEF3C7); border-top: 4px solid #D97706;}
    .kpi-box.c5 {background: linear-gradient(135deg, #F5F3FF, #EDE9FE); border-top: 4px solid #7C3AED;}
    .kpi-box.c6 {background: linear-gradient(135deg, #FDF2F8, #FCE7F3); border-top: 4px solid #DB2777;}

    .kpi-lbl {font-size: 0.75rem; font-weight: 700; color: #334155; text-transform: uppercase; margin-bottom: 4px;}
    .kpi-val {font-size: 1.5rem; font-weight: 800; color: #0F172A; margin-bottom: 4px;}
    .kpi-desc {font-size: 0.72rem; color: #475569; font-weight: 600;}

    /* Araç Açıklama Kutuları */
    .explain-card {
        background: linear-gradient(90deg, #F0F9FF 0%, #F8FAFC 100%);
        border: 1px solid #BAE6FD;
        border-left: 5px solid #0284C7;
        padding: 12px 16px;
        border-radius: 10px;
        color: #0F172A;
        font-size: 0.9rem;
        margin-bottom: 16px;
    }
    .tool-tip {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #6366F1;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 0.81rem;
        color: #334155;
        margin-bottom: 10px;
    }
    .side-note {
        font-size: 0.75rem;
        color: #475569;
        background: #F1F5F9;
        padding: 6px 10px;
        border-radius: 6px;
        border-left: 3px solid #4F46E5;
        margin-top: -6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Üst Başlık (Hero Banner)
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">📊 Kurumsal Müşteri Analitiği (RFM), CLV & AI Kayıp (Churn) Tahmin Sistemi (v3.0)</div>
    <div class="hero-sub">
        Müşteri tabanını <b>RFM ve Yaşam Boyu Değer (CLV)</b> metriklerine göre segmente eden, <b>Random Forest Makine Öğrenmesi</b> ile şirketi terk etme (Churn) riski taşıyan müşterileri önceden yakalayıp <b>Bütçe/ROI Simülasyonu ve Otonom Pazarlama Aksiyonları</b> üreten karar destek platformu.
    </div>
    <div class="contact-row">
        <span class="badge-item">👨‍💻 Lead AI Developer: Olgun Dağdelen</span>
        <a class="badge-item" href="mailto:olgundagdelen46@gmail.com">📧 olgundagdelen46@gmail.com</a>
        <a class="badge-item" href="tel:+905439322372">📞 +90 543 932 23 72</a>
        <a class="badge-item" href="https://wa.me/905439322372" target="_blank">💬 WhatsApp İletişim</a>
        <span class="badge-item">🎓 Danışman: Öğr. Gör. Ümit Bulut</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. VERİ ÜRETİMİ, ÖZELLİK MÜHENDİSLİĞİ VE RANDOM FOREST MODEL EĞİTİMİ
# ==============================================================================
@st.cache_data
def build_crm_and_train_model(n_customers=2000):
    np.random.seed(42)

    paketler = ["Başlangıç (Basic)", "Kurumsal (Pro)", "VIP Enterprise"]
    sozlesmeler = ["Aylık Taahhütsüz", "1 Yıllık Sözleşme", "2 Yıllık Sözleşme"]
    odeme_tipleri = ["Kredi Kartı (Otomatik)", "Banka Havalesi / EFT", "Kurumsal Fatura"]

    df = pd.DataFrame({
        "Musteri_ID": [f"CRM-{10000+i}" for i in range(n_customers)],
        "Abonelik_Paketi": np.random.choice(paketler, size=n_customers, p=[0.42, 0.38, 0.20]),
        "Sozlesme_Tipi": np.random.choice(sozlesmeler, size=n_customers, p=[0.50, 0.30, 0.20]),
        "Odeme_Yontemi": np.random.choice(odeme_tipleri, size=n_customers, p=[0.55, 0.25, 0.20]),
        "Yenilik_Gun_R": np.random.randint(1, 365, size=n_customers),
        "Siklik_Adet_F": np.random.randint(1, 50, size=n_customers),
        "Parasal_Deger_M_TL": np.random.randint(800, 35000, size=n_customers),
        "Musteri_Yasi_Ay": np.random.randint(1, 72, size=n_customers),
        "Son_1_Ay_Destek_Talebi": np.random.poisson(lam=1.6, size=n_customers),
        "Memnuniyet_Skoru_NPS": np.random.randint(1, 11, size=n_customers)
    })

    # CLV (Müşteri Yaşam Boyu Değeri) Hesaplama
    df["Aylik_Ortalama_Ciro_TL"] = np.round(df["Parasal_Deger_M_TL"] / np.maximum(1, df["Musteri_Yasi_Ay"] / 6), 2)
    df["CLV_Yasam_Boyu_Deger_TL"] = np.round(df["Parasal_Deger_M_TL"] * (1 + (df["Siklik_Adet_F"] / 50.0)), 0)

    # RFM Skorlama ve 5 Renkli Segment
    df["R_Skor"] = pd.qcut(df["Yenilik_Gun_R"], 4, labels=[4, 3, 2, 1]).astype(int)
    df["F_Skor"] = pd.qcut(df["Siklik_Adet_F"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    df["M_Skor"] = pd.qcut(df["Parasal_Deger_M_TL"], 4, labels=[1, 2, 3, 4]).astype(int)

    seg_map = {
        r"[1-2][1-2]": "💤 Uykuya Dalanlar (Hibernating)",
        r"[1-2][3-4]": "⚠️ Kaybedilmemesi Gerekenler (Cant Lose)",
        r"[3-4][1-2]": "🌱 Potansiyel / Yeni Müşteriler",
        r"3[3-4]": "💎 Sadık Müşteriler (Loyal)",
        r"4[3-4]": "🏆 Şampiyonlar (Champions)"
    }
    df["Segment"] = (df["R_Skor"].astype(str) + df["F_Skor"].astype(str)).replace(seg_map, regex=True)

    # Gerçekçi Churn (Kayıp) Oluşturma
    sozlesme_etkisi = df["Sozlesme_Tipi"].map({"Aylık Taahhütsüz": 0.18, "1 Yıllık Sözleşme": 0.05, "2 Yıllık Sözleşme": -0.08})
    churn_logit = (
        (df["Yenilik_Gun_R"] / 365.0) * 0.42 +
        (df["Son_1_Ay_Destek_Talebi"] / 6.0) * 0.35 -
        (df["Memnuniyet_Skoru_NPS"] / 10.0) * 0.25 -
        (df["Musteri_Yasi_Ay"] / 72.0) * 0.20 +
        sozlesme_etkisi
    )
    df["Gercek_Churn"] = (churn_logit + np.random.normal(0, 0.08, n_customers) > 0.32).astype(int)

    feature_cols = [
        "Yenilik_Gun_R", "Siklik_Adet_F", "Parasal_Deger_M_TL",
        "Musteri_Yasi_Ay", "Son_1_Ay_Destek_Talebi", "Memnuniyet_Skoru_NPS"
    ]
    X = df[feature_cols]
    y = df["Gercek_Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred) * 100,
        "precision": precision_score(y_test, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_test, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_test, y_pred, zero_division=0) * 100,
        "cm": confusion_matrix(y_test, y_pred)
    }

    df["Churn_Riski_Yuzde"] = np.round(rf_model.predict_proba(X)[:, 1] * 100, 1)
    return df, rf_model, metrics, feature_cols

df_all, rf_model, model_metrics, feature_cols = build_crm_and_train_model()

# ==============================================================================
# 2. YAN MENÜ (SIDEBAR) - FİLTRELER VE NE İŞE YARADIKLARI
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#EEF2FF,#E0E7FF); border-left:4px solid #4338CA; padding:14px; border-radius:10px; margin-bottom:14px;">
        <b style="color:#312E81;">👨‍💻 Proje Lideri & Mimarı</b><br>
        <span style="font-size:1.05rem; font-weight:800; color:#0F172A;">Olgun Dağdelen</span><br>
        <small style="color:#334155;">Ekip: Ahmet Emek, İbrahim Taştekin, Emir Kıdrez</small><br>
        <small style="color:#4338CA; font-weight:600;">🎓 Danışman: Öğr. Gör. Ümit Bulut</small>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🎛️ Dinamik CRM Filtreleri")
    st.caption("Aşağıdaki araçlarla müşteri veri tabanını filtreleyerek tüm grafikleri canlı güncelleyebilirsiniz:")

    secili_paket = st.multiselect(
        "📦 1. Abonelik Paketi Filtresi:",
        options=df_all["Abonelik_Paketi"].unique(),
        default=df_all["Abonelik_Paketi"].unique()
    )
    st.markdown('<div class="side-note">ℹ️ <b>Ne İşe Yarar?</b> Sadece seçtiğiniz abonelik paketindeki (Basic, Pro, VIP) müşterilerin analizini gösterir.</div>', unsafe_allow_html=True)

    secili_sozlesme = st.multiselect(
        "📝 2. Sözleşme Tipi Filtresi:",
        options=df_all["Sozlesme_Tipi"].unique(),
        default=df_all["Sozlesme_Tipi"].unique()
    )
    st.markdown('<div class="side-note">ℹ️ <b>Ne İşe Yarar?</b> Aylık taahhütsüz veya yıllık sözleşmeli müşterilerin kayıp (churn) farkını incelemenizi sağlar.</div>', unsafe_allow_html=True)

    risk_esigi = st.slider("🚨 3. Kritik Churn Risk Eşiği (%):", 30, 85, 60, step=5)
    st.markdown('<div class="side-note">ℹ️ <b>Ne İşe Yarar?</b> Bu yüzdelik oranın üzerinde ayrılma ihtimali olan müşterileri "Acil Müdahale Listesine" alır.</div>', unsafe_allow_html=True)

    df_crm = df_all[
        (df_all["Abonelik_Paketi"].isin(secili_paket)) &
        (df_all["Sozlesme_Tipi"].isin(secili_sozlesme))
    ].copy()

    st.divider()
    csv_bytes = df_crm.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Filtrelenmiş CRM Verisini İndir (.CSV)",
        data=csv_bytes,
        file_name="kurumsal_crm_churn_analizi.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.caption("🔹 Tablodaki tüm müşteri skorlarını Excel/CSV olarak bilgisayarınıza indirir.")

# ==============================================================================
# 3. ÜST RENKLİ KPI KARTLARI (6 TEMEL GÖSTERGE)
# ==============================================================================
st.markdown('<div class="tool-tip">ℹ️ <b>Üst Özet Kartları Ne İşe Yarar?</b> Şirketin toplam müşteri sayısını, toplam cirosunu, kayıp (churn) oranını, risk altındaki parayı ve yapay zeka modelinin başarı yüzdesini tek bakışta özetler.</div>', unsafe_allow_html=True)

toplam_musteri = len(df_crm)
toplam_ciro = df_crm["Parasal_Deger_M_TL"].sum()
ort_churn = df_crm["Gercek_Churn"].mean() * 100
riskli_df = df_crm[df_crm["Churn_Riski_Yuzde"] >= risk_esigi]
riskteki_ciro = riskli_df["Parasal_Deger_M_TL"].sum()
ort_clv = df_crm["CLV_Yasam_Boyu_Deger_TL"].mean()

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-box c1">
        <div class="kpi-lbl">👥 Toplam Aktif Müşteri</div>
        <div class="kpi-val">{toplam_musteri:,}</div>
        <div class="kpi-desc">Filtrelenen CRM Havuzu</div>
    </div>
    <div class="kpi-box c2">
        <div class="kpi-lbl">💰 Toplam Ciro Hacmi</div>
        <div class="kpi-val">₺{toplam_ciro:,.0f}</div>
        <div class="kpi-desc">Kümülatif Gelir</div>
    </div>
    <div class="kpi-box c3">
        <div class="kpi-lbl">🚨 Ortalama Churn Oranı</div>
        <div class="kpi-val">%{ort_churn:.1f}</div>
        <div class="kpi-desc">{len(riskli_df)} Kritik Riskli Müşteri</div>
    </div>
    <div class="kpi-box c4">
        <div class="kpi-lbl">⚠️ Risk Altındaki Ciro</div>
        <div class="kpi-val">₺{riskteki_ciro/1000000:.2f}M</div>
        <div class="kpi-desc">Kaybedilme Tehlikesi Olan Gelir</div>
    </div>
    <div class="kpi-box c5">
        <div class="kpi-lbl">💎 Ort. Müşteri Değeri (CLV)</div>
        <div class="kpi-val">₺{ort_clv:,.0f}</div>
        <div class="kpi-desc">Yaşam Boyu Bırakılan Değer</div>
    </div>
    <div class="kpi-box c6">
        <div class="kpi-lbl">🤖 AI Model Doğruluğu</div>
        <div class="kpi-val">%{model_metrics['accuracy']:.1f}</div>
        <div class="kpi-desc">F1 Skoru: %{model_metrics['f1']:.1f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. 5 RENKLİ VE AÇIKLAMALI KURUMSAL SEKME
# ==============================================================================
SEGMENT_COLORS = {
    "🏆 Şampiyonlar (Champions)": "#10B981",
    "💎 Sadık Müşteriler (Loyal)": "#3B82F6",
    "🌱 Potansiyel / Yeni Müşteriler": "#8B5CF6",
    "⚠️ Kaybedilmemesi Gerekenler (Cant Lose)": "#F59E0B",
    "💤 Uykuya Dalanlar (Hibernating)": "#EF4444"
}

tabs = st.tabs([
    "📈 1. RFM Segmentasyon & 3B Uzay",
    "🤖 2. Canlı AI Churn Simülatörü",
    "💰 3. CLV & Kurtarma Bütçesi (ROI) Analizi",
    "🔬 4. AI Model Performansı & Kohort",
    "🎯 5. Riskli Müşteriler & Aksiyon Listesi"
])

# ------------------------------------------------------------------------------
# SEKME 1: RFM SEGMENTASYON VE GRAFİK ANALİZİ
# ------------------------------------------------------------------------------
with tabs[0]:
    st.markdown("""
    <div class="explain-card">
        <b>💡 Bu Sekme Ne İşe Yarar? (RFM Müşteri Segmentasyonu)</b><br>
        Müşterileri <b>Yenilik (Recency - Son geliş günü)</b>, <b>Sıklık (Frequency - İşlem adedi)</b> ve <b>Parasal Değer (Monetary - Toplam harcama)</b> metriklerine göre 5 renkli gruba ayırır. Hangi grubun şirkete en çok para kazandırdığını ve hangi grubun uykuya daldığını gösterir.
    </div>
    """, unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="tool-tip">ℹ️ <b>Pasta Grafiği Ne Gösterir?</b> Tüm müşterilerin yüzde kaçının Şampiyon, Sadık veya Riskli grupta olduğunu oranlar.</div>', unsafe_allow_html=True)
        fig_pie = px.pie(
            df_crm, names="Segment", hole=0.45, color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            title="🍩 Müşteri Segment Dağılımı (%)"
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with g2:
        st.markdown('<div class="tool-tip">ℹ️ <b>Sütun Grafiği Ne Gösterir?</b> Her bir müşteri grubunun şirkete bıraktığı ortalama ciroyu (TL) karşılaştırır.</div>', unsafe_allow_html=True)
        seg_avg = df_crm.groupby("Segment", as_index=False)["Parasal_Deger_M_TL"].mean().sort_values("Parasal_Deger_M_TL", ascending=False)
        fig_bar = px.bar(
            seg_avg, x="Segment", y="Parasal_Deger_M_TL", color="Segment",
            text_auto=".0f", color_discrete_map=SEGMENT_COLORS,
            title="📊 Segmentlere Göre Ortalama Ciro (TL)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.markdown('<div class="tool-tip">ℹ️ <b>3 Boyutlu Kümeleme Grafiği Ne İşe Yarar?</b> Müşterilerin Yenilik (X), Sıklık (Y) ve Harcama (Z) uzayında nasıl kümelendiğini fareyle döndürerek incelemenizi sağlar.</div>', unsafe_allow_html=True)
        fig_3d = px.scatter_3d(
            df_crm.head(500), x="Yenilik_Gun_R", y="Siklik_Adet_F", z="Parasal_Deger_M_TL",
            color="Segment", size="Churn_Riski_Yuzde", color_discrete_map=SEGMENT_COLORS,
            hover_data=["Musteri_ID", "Abonelik_Paketi", "Churn_Riski_Yuzde"],
            title="🧊 3 Boyutlu RFM Müşteri Uzayı (Top Büyüklüğü = Churn Riski)"
        )
        fig_3d.update_layout(height=430)
        st.plotly_chart(fig_3d, use_container_width=True)

    with g4:
        st.markdown('<div class="tool-tip">ℹ️ <b>Hiyerarşik Ciro Haritası (Treemap) Ne Gösterir?</b> Paketlerin ve müşteri segmentlerinin toplam ciro içindeki hacmini kutu büyüklükleriyle gösterir.</div>', unsafe_allow_html=True)
        fig_tree = px.treemap(
            df_crm, path=["Abonelik_Paketi", "Segment"], values="Parasal_Deger_M_TL",
            color="Churn_Riski_Yuzde", color_continuous_scale="RdYlGn_r",
            title="🗺️ Paket ve Segmentlere Göre Ciro & Churn Riski Haritası (Kırmızı = Yüksek Risk)"
        )
        fig_tree.update_layout(height=430)
        st.plotly_chart(fig_tree, use_container_width=True)

# ------------------------------------------------------------------------------
# SEKME 2: CANLI AI MÜŞTERİ KAYBI (CHURN) SİMÜLATÖRÜ
# ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("""
    <div class="explain-card">
        <b>💡 Bu Sekme Ne İşe Yarar? (Canlı AI Müşteri Kaybı Simülatörü)</b><br>
        Sol taraftaki kaydırma çubuklarıyla hayali veya gerçek bir müşterinin değerlerini değiştirdiğinizde, arka plandaki <b>Random Forest</b> yapay zeka modeli saniyesinde o müşterinin <b>şirketi terk etme ihtimalini (%)</b> hesaplar ve sağdaki hız kadranında (Gauge) gösterir.
    </div>
    """, unsafe_allow_html=True)

    s_left, s_right = st.columns([1, 1.25])
    with s_left:
        st.markdown("#### 🎛️ Test Edilecek Müşteri Profilini Girin")
        st.markdown('<div class="tool-tip">ℹ️ Çubukları sağa-sola kaydırarak yapay zekanın tepkisini canlı test edin:</div>', unsafe_allow_html=True)

        in_R = st.slider("📅 Son İşlemden Geçen Süre (Recency - Gün):", 1, 365, 140, help="Müşteri en son kaç gün önce alışveriş yaptı?")
        in_F = st.slider("🔄 Toplam İşlem Sıklığı (Frequency - Adet):", 1, 50, 8, help="Bugüne kadar toplam kaç kez sipariş verdi?")
        in_M = st.number_input("💳 Toplam Bıraktığı Ciro (Monetary - TL):", 800, 35000, 12500, step=500, help="Şirkete kazandırdığı toplam para.")
        in_Yas = st.slider("⏳ Şirketteki Üyelik Yaşı (Ay):", 1, 72, 14, help="Kaç aydır şirketimizin müşterisi?")
        in_Destek = st.slider("📞 Son 1 Ayda Açtığı Şikayet / Destek Kaydı:", 0, 10, 4, help="Çağrı merkezine veya destek hattına kaç kez şikayet bıraktı?")
        in_NPS = st.slider("⭐ Müşteri Memnuniyet Anketi Puanı (NPS 1-10):", 1, 10, 4, help="1: Çok Memnuniyetsiz, 10: Çok Memnun.")

        test_input = pd.DataFrame([[in_R, in_F, in_M, in_Yas, in_Destek, in_NPS]], columns=feature_cols)
        canli_risk = float(rf_model.predict_proba(test_input)[0][1] * 100)

    with s_right:
        st.markdown("#### 🤖 Yapay Zeka Risk Kadranı & Karar Reçetesi")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=canli_risk,
            number={"suffix": "%", "font": {"size": 42, "color": "#0F172A"}},
            title={"text": "Tahmini Müşteri Kayıp (Churn) Olasılığı", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1E293B"},
                "steps": [
                    {"range": [0, 35], "color": "#10B981"},
                    {"range": [35, 60], "color": "#F59E0B"},
                    {"range": [60, 100], "color": "#EF4444"}
                ],
                "threshold": {"line": {"color": "#DC2626", "width": 4}, "thickness": 0.8, "value": risk_esigi}
            }
        ))
        fig_gauge.update_layout(height=290, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if canli_risk >= 60:
            st.error(f"🚨 **KRİTİK ALARM (%{canli_risk:.1f} Kayıp Riski):** Bu müşteri şirketi terk etmek üzere! **Önerilen Otonom Aksiyon:** Müşteri temsilcisi bugün içinde doğrudan aramalı, açık şikayet kaydını çözmeli ve **%25 Sadakat İndirimi + 1 Ay Ücretsiz VIP Paket** tanımlamalıdır.")
        elif canli_risk >= 35:
            st.warning(f"⚠️ **ORTA RİSK (%{canli_risk:.1f} Kayıp Riski):** Müşteri soğuma evresinde. **Önerilen Otonom Aksiyon:** Otomatik SMS/E-posta ile **%15 Çapraz Alışveriş Kuponu** gönderilmeli.")
        else:
            st.success(f"✅ **GÜVENLİ MÜŞTERİ (%{canli_risk:.1f} Kayıp Riski):** Müşteri platforma bağlı ve memnun. **Önerilen Otonom Aksiyon:** Yıllık üst pakete geçiş (Upsell) teklifi sunulabilir.")

        imp_df = pd.DataFrame({
            "Kriter": ["Son İşlem Günü (R)", "İşlem Sıklığı (F)", "Toplam Ciro (M)", "Üyelik Yaşı (Ay)", "Şikayet Sayısı", "Memnuniyet (NPS)"],
            "Etki_Yuzdesi": rf_model.feature_importances_ * 100
        }).sort_values("Etki_Yuzdesi", ascending=True)
        fig_imp = px.bar(imp_df, x="Etki_Yuzdesi", y="Kriter", orientation="h", color="Etki_Yuzdesi", color_continuous_scale="Purpor", title="🧠 Yapay Zeka Bu Kararı Verirken Hangi Kritere Ne Kadar Ağırlık Verdi (%)")
        fig_imp.update_layout(height=260)
        st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------------------------------------------
# SEKME 3: CLV & KURTARMA BÜTÇESİ (ROI) SİMÜLATÖRÜ
# ------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("""
    <div class="explain-card">
        <b>💡 Bu Sekme Ne İşe Yarar? (Finansal Etki & Kurtarma ROI Simülatörü)</b><br>
        Yöneticilerin en çok sorduğu <i>"Riskli müşterilere indirim kampanyası yaparsak kasamızdan ne kadar çıkar, kaç müşteriyi kurtarırız ve net kârımız (ROI) ne olur?"</i> sorusunu canlı hesaplar.
    </div>
    """, unsafe_allow_html=True)

    roi_c1, roi_c2 = st.columns([1, 1.35])
    with roi_c1:
        st.markdown("#### 💸 Kampanya Bütçe & İndirim Ayarları")
        st.markdown('<div class="tool-tip">ℹ️ Riskli müşterileri elde tutmak için vereceğiniz indirim oranını ve beklenen başarıyı seçin:</div>', unsafe_allow_html=True)

        teklif_indirim = st.slider("🎁 Riskli Müşteriye Verilecek İndirim / Kupon Oranı (%):", 5, 40, 15, step=5)
        kurtarma_basarisi = st.slider("🎯 Kampanyanın Müşteriyi İkna Etme / Kurtarma Oranı (%):", 10, 80, 45, step=5)
        operasyon_maliyeti = st.number_input("📞 Müşteri Başı Arama / SMS Operasyon Maliyeti (TL):", 10, 500, 75, step=25)

        hedef_kitle_sayisi = len(riskli_df)
        toplam_riskli_ciro = float(riskli_df["Parasal_Deger_M_TL"].sum())
        kurtarilan_brut_ciro = toplam_riskli_ciro * (kurtarma_basarisi / 100.0)
        kampanya_indirim_maliyeti = kurtarilan_brut_ciro * (teklif_indirim / 100.0)
        toplam_operasyon_gideri = hedef_kitle_sayisi * operasyon_maliyeti
        toplam_kampanya_maliyeti = kampanya_indirim_maliyeti + toplam_operasyon_gideri
        net_kurtarilan_kar = kurtarilan_brut_ciro - toplam_kampanya_maliyeti
        roi_yuzde = (net_kurtarilan_kar / max(1, toplam_kampanya_maliyeti)) * 100

        st.metric("Kurtarılması Beklenen Müşteri Sayısı", f"{int(hedef_kitle_sayisi * kurtarma_basarisi / 100)} Kişi", delta=f"Toplam {hedef_kitle_sayisi} Riskli Müşteriden")
        st.metric("Toplam Kampanya & İndirim Maliyeti", f"₺{toplam_kampanya_maliyeti:,.0f}", delta="İndirim + Arama Gideri")
        st.metric("Şirkette Kalan Net Kurtarılmış Gelir", f"₺{net_kurtarilan_kar:,.0f}", delta=f" Yatırım Getirisi (ROI): %{roi_yuzde:.0f}")

    with roi_c2:
        st.markdown('<div class="tool-tip">ℹ️ <b>Finansal Karşılaştırma Grafiği Ne Gösterir?</b> Hiçbir şey yapılmazsa kaybedilecek para ile AI kampanyası sonrası kurtarılan net geliri kıyaslar.</div>', unsafe_allow_html=True)
        fin_df = pd.DataFrame({
            "Finansal_Kalem": [
                "1. Müdahale Edilmezse Kaybedilecek Ciro",
                "2. Kampanya ile Kurtarılan Brüt Ciro",
                "3. Kampanya & İndirim Maliyeti",
                "4. Şirkete Kalan NET KAZANÇ"
            ],
            "Tutar_TL": [toplam_riskli_ciro, kurtarilan_brut_ciro, toplam_kampanya_maliyeti, net_kurtarilan_kar],
            "Renk": ["Kaybedilecek Risk", "Brüt Kurtarılan", "Maliyet", "Net Kazanç"]
        })
        fig_roi = px.bar(
            fin_df, x="Finansal_Kalem", y="Tutar_TL", color="Renk", text_auto=".3s",
            color_discrete_map={"Kaybedilecek Risk": "#EF4444", "Brüt Kurtarılan": "#3B82F6", "Maliyet": "#F59E0B", "Net Kazanç": "#10B981"},
            title="💰 AI Müşteri Kurtarma Kampanyası Maliyet vs. Net Kazanç Analizi (TL)"
        )
        st.plotly_chart(fig_roi, use_container_width=True)

# ------------------------------------------------------------------------------
# SEKME 4: AI MODEL PERFORMANSI VE KOHORT ISI HARİTASI
# ------------------------------------------------------------------------------
with tabs[3]:
    st.markdown("""
    <div class="explain-card">
        <b>💡 Bu Sekme Ne İşe Yarar? (Akademik Model Doğrulama & Kohort Analizi)</b><br>
        Eğitilen <b>Random Forest</b> modelinin test verisi üzerindeki <b>Accuracy, Precision, Recall ve F1</b> başarı metriklerini, <b>Karmaşıklık Matrisini (Confusion Matrix)</b> ve hangi sözleşme/paket kombinasyonunda daha çok müşteri kaybı yaşandığını gösterir.
    </div>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy (Doğruluk)", f"%{model_metrics['accuracy']:.1f}", help="Doğru tahmin edilen toplam müşteri oranı.")
    mc2.metric("Precision (Kesinlik)", f"%{model_metrics['precision']:.1f}", help="Kaçacak dediğimiz müşterilerin gerçekten kaçma oranı.")
    mc3.metric("Recall (Duyarlılık)", f"%{model_metrics['recall']:.1f}", help="Gerçekte kaçan müşterilerin ne kadarını önceden yakaladığımız.")
    mc4.metric("F1-Score (Denge Skoru)", f"%{model_metrics['f1']:.1f}", help="Kesinlik ve Duyarlılığın harmonik ortalaması.")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown('<div class="tool-tip">ℹ️ <b>Karmaşıklık Matrisi (Confusion Matrix) Ne Gösterir?</b> Yapay zekanın test verisinde kaç müşteriyi doğru, kaçını yanlış sınıflandırdığını gösterir.</div>', unsafe_allow_html=True)
        cm = model_metrics["cm"]
        fig_cm = px.imshow(
            cm, text_auto=True, aspect="auto", color_continuous_scale="Blues",
            x=["Tahmin: Kalacak (0)", "Tahmin: Ayrılacak (1)"],
            y=["Gerçek: Kaldı (0)", "Gerçek: Ayrıldı (1)"],
            title="🎯 Random Forest Karmaşıklık Matrisi (Test Seti Sonuçları)"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with p_col2:
        st.markdown('<div class="tool-tip">ℹ️ <b>Paket x Sözleşme Isı Haritası Ne Gösterir?</b> Hangi abonelik ve sözleşme tipindeki müşterilerin daha yüksek oranda şirketi terk ettiğini (Kırmızı alanlar) ortaya çıkarır.</div>', unsafe_allow_html=True)
        pivot_churn = df_crm.pivot_table(index="Abonelik_Paketi", columns="Sozlesme_Tipi", values="Churn_Riski_Yuzde", aggfunc="mean").round(1)
        fig_heat = px.imshow(
            pivot_churn, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn_r",
            title="🔥 Abonelik Paketi x Sözleşme Tipi Ortalama Churn Riski (%) Isı Haritası"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ------------------------------------------------------------------------------
# SEKME 5: RİSKLİ MÜŞTERİLER VE OTONOM AKSİYON LİSTESİ (MATPLOTLIB HATASI GİDERİLDİ!)
# ------------------------------------------------------------------------------
with tabs[4]:
    st.markdown("""
    <div class="explain-card">
        <b>💡 Bu Sekme Ne İşe Yarar? (Otonom Pazarlama & İletişim Aksiyon Listesi)</b><br>
        Sol menüde belirlediğiniz risk eşiğinin üzerindeki tüm müşterileri en yüksek riskliden başlayarak sıralar ve her bir müşterinin neden kaçmak üzere olduğunu analiz ederek <b>otomatik kurtarma reçetesi</b> atar.
    </div>
    """, unsafe_allow_html=True)

    def otonom_aksiyon_uret(row):
        if row["Parasal_Deger_M_TL"] >= 20000 and row["Son_1_Ay_Destek_Talebi"] >= 3:
            return "🚨 VIP Kriz Müdahalesi: Özel Müşteri Direktörü Arasın + %25 İndirim"
        elif row["Son_1_Ay_Destek_Talebi"] >= 4:
            return "📞 Teknik Destek Önceliği: Açık Şikayet Kaydını Çöz + 1 Ay Ücretsiz Hediye"
        elif row["Sozlesme_Tipi"] == "Aylık Taahhütsüz":
            return "📝 Taahhüt Teklifi: Yıllık Sözleşmeye Geçişte %30 Sabit Fiyat Avantajı Sun"
        elif row["Yenilik_Gun_R"] >= 180:
            return "🎁 Özledik Kampanyası: Geri Dönüşe Özel 1.500 TL Hediye Çeki Gönder"
        else:
            return "✉️ Otomatik Sadakat E-Postası ve Memnuniyet Anketi Gönder"

    aksiyon_df = riskli_df.sort_values("Churn_Riski_Yuzde", ascending=False).copy()
    aksiyon_df["Otonom_AI_Aksiyon_Reçetesi"] = aksiyon_df.apply(otonom_aksiyon_uret, axis=1)

    st.markdown(f"#### 🚨 Acil Müdahale Gerektiren Müşteriler (`%{risk_esigi}` Risk Eşiğinin Üzerinde Toplam **{len(aksiyon_df)}** Müşteri)")
    st.markdown('<div class="tool-tip">ℹ️ <b>Tablo Nasıl Okunur?</b> "Churn Riski (%)" sütunundaki kırmızı barlar müşterinin kaçma ihtimalini, en sağdaki sütun ise yapay zekanın önerdiği aksiyonu gösterir. (Hiçbir ek kütüphane hatası vermez!)</div>', unsafe_allow_html=True)

    gosterim_kolonlari = [
        "Musteri_ID", "Segment", "Abonelik_Paketi", "Sozlesme_Tipi",
        "Churn_Riski_Yuzde", "Parasal_Deger_M_TL", "CLV_Yasam_Boyu_Deger_TL",
        "Son_1_Ay_Destek_Talebi", "Memnuniyet_Skoru_NPS", "Otonom_AI_Aksiyon_Reçetesi"
    ]

    st.dataframe(
        aksiyon_df[gosterim_kolonlari].head(50),
        column_config={
            "Musteri_ID": st.column_config.TextColumn("Müşteri Kodu"),
            "Segment": st.column_config.TextColumn("RFM Segmenti"),
            "Abonelik_Paketi": st.column_config.TextColumn("Paket"),
            "Sozlesme_Tipi": st.column_config.TextColumn("Sözleşme"),
            "Churn_Riski_Yuzde": st.column_config.ProgressColumn(
                "AI Churn Riski (%)",
                help="Yapay zekanın hesapladığı şirketi terk etme olasılığı",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),
            "Parasal_Deger_M_TL": st.column_config.NumberColumn("Toplam Ciro (TL)", format="₺%d"),
            "CLV_Yasam_Boyu_Deger_TL": st.column_config.NumberColumn("Yaşam Boyu Değer (CLV)", format="₺%d"),
            "Son_1_Ay_Destek_Talebi": st.column_config.NumberColumn("Şikayet Sayısı", format="%d Kayıt"),
            "Memnuniyet_Skoru_NPS": st.column_config.NumberColumn("NPS Puanı", format="%d / 10"),
            "Otonom_AI_Aksiyon_Reçetesi": st.column_config.TextColumn("🤖 AI Otonom Kurtarma Aksiyonu")
        },
        use_container_width=True,
        height=440,
        hide_index=True
    )
