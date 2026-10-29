import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time

# إعداد الصفحة
st.set_page_config(page_title="Zero-Day AI Detection System", layout="wide")

st.title("🛡️ Zero-Day Intrusion Detection System (Deep Learning)")
st.write("نظام مراقبة وتحليل سلوك الشبكة لاكتشاف الهجمات المجهولة باستخدام شبكات الـ Autoencoder")

# 1. تعريف النموذج وتحميل الأوزان
class NetworkAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(NetworkAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

input_dim = 6
model = NetworkAutoencoder(input_dim)
model.load_state_dict(torch.load("autoencoder_model.pth", map_location=torch.device('cpu')))
model.eval()

# العتبة المعتمدة
THRESHOLD = 0.055271

# 2. الواجهة التفاعلية
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ إعدادات المحاكاة")
    traffic_type = st.radio("نوع الترافيك المراد فحصه:", ["Normal Traffic (طبيعي)", "Zero-Day Attack (هجوم غير معروف)"])
    run_btn = st.button("فحص الاتصال عبر النموذج")

with col2:
    st.subheader("📊 حالة المراقبة والتحليل")
    
    if run_btn:
        # تحميل عينة عشوائية حسب الاختيار
        if "طبيعي" in traffic_type:
            data = np.load("X_test_normal.npy")
            sample = data[np.random.choice(len(data))]
            true_label = "طبيعي"
        else:
            data = np.load("X_test_attack.npy")
            sample = data[np.random.choice(len(data))]
            true_label = "هجوم Zero-Day"

        # تمرير العينة على النموذج
        tensor_sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            reconstruction = model(tensor_sample)
            error = torch.mean((tensor_sample - reconstruction) ** 2).item()

        # عرض النتائج والمقاييس
        st.write(f"**خطأ إعادة البناء (Reconstruction Error):** `{error:.6f}`")
        st.write(f"**عتبة الكشف الآمنة (Threshold):** `{THRESHOLD:.6f}`")

        # التقييم
        if error > THRESHOLD:
            st.error(f"🚨 تنبيه أمني عالي الخطورة: اكتشاف هجوم سلوكي غير معروف (Zero-Day)! (الخطأ تجاوز العتبة)")
        else:
            st.success("✅ الاتصال سليم ومطابق للسلوك الطبيعي للشبكة (Benign Traffic).")

        # عرض بيانات الحزمة
        features_df = pd.DataFrame([sample], columns=[
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length Fwd', 'Flow Bytes/s', 'Flow Packets/s'
        ])
        st.write("خصائص حزمة الاتصال المحللة:")
        st.dataframe(features_df)
