import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print(">>> المرحلة 1: تجهيز ومحاكاة بيانات الشبكة...")

# 1. توليد بيانات شبكة تحاكي خصائص حزم البيانات الطبيعية والهجمات
# الميزات: مدة الاتصال، عدد الحزم، حجم البيانات، معدل البايتات بالثانية
np.random.seed(42)

n_benign = 8000   # 8000 عينة ترافيك طبيعي
n_attack = 1500   # 1500 عينة هجوم (Zero-Day Target)

# الترافيك الطبيعي: قيم متوازنة ومستقرة
benign_data = {
    'Flow Duration': np.random.exponential(scale=1000, size=n_benign),
    'Total Fwd Packets': np.random.poisson(lam=10, size=n_benign),
    'Total Backward Packets': np.random.poisson(lam=12, size=n_benign),
    'Total Length of Fwd Packets': np.random.normal(loc=500, scale=100, size=n_benign),
    'Flow Bytes/s': np.random.normal(loc=1500, scale=300, size=n_benign),
    'Flow Packets/s': np.random.normal(loc=20, scale=5, size=n_benign),
    'Label': ['BENIGN'] * n_benign
}

# ترافيك الهجوم: شاذ جداً (مثلاً سيل حزم مفاجئ، أحجام غير اعتيادية)
attack_data = {
    'Flow Duration': np.random.exponential(scale=50, size=n_attack),
    'Total Fwd Packets': np.random.poisson(lam=150, size=n_attack),
    'Total Backward Packets': np.random.poisson(lam=2, size=n_attack),
    'Total Length of Fwd Packets': np.random.normal(loc=8000, scale=500, size=n_attack),
    'Flow Bytes/s': np.random.normal(loc=50000, scale=5000, size=n_attack),
    'Flow Packets/s': np.random.normal(loc=500, scale=50, size=n_attack),
    'Label': ['ATTACK_ZERO_DAY'] * n_attack
}

df_benign = pd.DataFrame(benign_data)
df_attack = pd.DataFrame(attack_data)

df = pd.concat([df_benign, df_attack], ignore_index=True)

# تنظيف القيم وضمان خلوها من السالب
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].clip(lower=0)

print(f"أبعاد البيانات الإجمالية: {df.shape}")
print("\nتوزيع الترافيك:")
print(df['Label'].value_counts())

# 2. عزل الطبيعي عن الهجوم لمحاكاة الـ Zero-Day
normal_traffic = df[df['Label'] == 'BENIGN'].drop(columns=['Label'])
attack_traffic = df[df['Label'] != 'BENIGN'].drop(columns=['Label'])

# 3. التقسيم: 80% من الطبيعي فقط للتدريب
X_train_normal, X_test_normal = train_test_split(normal_traffic, test_size=0.2, random_state=42)

# 4. التطبيع (MinMaxScaler)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_normal_scaled = scaler.transform(X_test_normal)
X_attack_scaled = scaler.transform(attack_traffic)

# 5. حفظ المصفوفات لاستخدامها في PyTorch
np.save("X_train.npy", X_train_scaled)
np.save("X_test_normal.npy", X_test_normal_scaled)
np.save("X_test_attack.npy", X_attack_scaled)

print("\n>>> تم إنشاء وحفظ مصفوفات التدريب والاختبار بنجاح:")
print(f"- حجم بيانات التدريب (طبيعي فقط): {X_train_scaled.shape}")
print(f"- حجم بيانات اختبار الترافيك الطبيعي: {X_test_normal_scaled.shape}")
print(f"- حجم بيانات اختبار الهجوم (الـ Zero-Day): {X_attack_scaled.shape}")
