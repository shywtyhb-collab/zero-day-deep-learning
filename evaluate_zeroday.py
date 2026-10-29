import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

print(">>> المرحلة 3: اختبار كشف هجمات الـ Zero-Day وتقييم الأداء...")

# 1. تحديد المعالج وتحميل البيانات
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

X_test_normal = np.load("X_test_normal.npy")
X_test_attack = np.load("X_test_attack.npy")

input_dim = X_test_normal.shape[1]

# 2. تعريف نفس معمارية النموذج لتحميل الأوزان المحفوظة
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

model = NetworkAutoencoder(input_dim).to(device)
model.load_state_dict(torch.load("autoencoder_model.pth"))
model.eval() # تفعيل وضع الاختبار

# 3. دالة لحساب خطأ إعادة البناء (Reconstruction Error) لكل عينة
def get_reconstruction_errors(data):
    tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(tensor_data)
        # حساب MSE لكل سطر (عينة ترافيك) على حدة
        errors = torch.mean((tensor_data - predictions) ** 2, dim=1)
    return errors.cpu().numpy()

normal_errors = get_reconstruction_errors(X_test_normal)
attack_errors = get_reconstruction_errors(X_test_attack)

print(f"\nمتوسط خطأ الترافيك الطبيعي: {normal_errors.mean():.6f}")
print(f"متوسط خطأ ترافيك الـ Zero-Day: {attack_errors.mean():.6f}")

# 4. تحديد عتبة الكشف (Anomaly Threshold)
# نعتمد عتبة: متوسط خطأ الترافيك الطبيعي + (3 * الانحراف المعياري)
threshold = normal_errors.mean() + (3 * normal_errors.std())
print(f"عتبة الكشف المعتمدة (Threshold): {threshold:.6f}")

# 5. التصنيف: إذا كان الخطأ أكبر من العتبة يعتبر هجوماً (1)، وإلا طبيعياً (0)
# دمج بيانات الاختبار الحقيقية: 0 للطبيعي، 1 للهجوم
y_true = np.array([0] * len(normal_errors) + [1] * len(attack_errors))
all_errors = np.concatenate([normal_errors, attack_errors])
y_pred = (all_errors > threshold).astype(int)

# 6. طباعة تقرير الأداء ومصفوفة الارتباك
print("\n" + "="*50)
print("مصفوفة الارتباك (Confusion Matrix):")
cm = confusion_matrix(y_true, y_pred)
print(f"True Negatives (طبيعي مصنف كطبيعي): {cm[0, 0]}")
print(f"False Positives (طبيعي صُنف كهجوم بالخطأ): {cm[0, 1]}")
print(f"False Negatives (هجوم لم يتم كشفه): {cm[1, 0]}")
print(f"True Positives (هجوم تم اكتشافه بنجاح): {cm[1, 1]}")

print("\nتقرير التصنيف الدقيق (Classification Report):")
print(classification_report(y_true, y_pred, target_names=["Normal Traffic", "Zero-Day Attack"]))
print("="*50)
