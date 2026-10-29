import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print(">>> المرحلة 2: بناء وتدريب نموذج الـ Autoencoder...")

# 1. تفعيل المعالج الرسومي لجهاز الـ Mac (Apple Silicon GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"جهاز التدريب المستخدم: {device}")

# 2. تحميل البيانات المعالجة
X_train = np.load("X_train.npy")
X_test_normal = np.load("X_test_normal.npy")
X_test_attack = np.load("X_test_attack.npy")

input_dim = X_train.shape[1]  # عدد الخصائص (6 خصائص)

# تحويل البيانات إلى Tensors خاصة بمكتبة PyTorch
train_tensor = torch.tensor(X_train, dtype=torch.float32)
train_loader = DataLoader(
    TensorDataset(train_tensor, train_tensor), batch_size=32, shuffle=True
)


# 3. بناء معمارية الـ Autoencoder
class NetworkAutoencoder(nn.Module):

  def __init__(self, input_dim):
    super(NetworkAutoencoder, self).__init__()
    # الـ Encoder: يضغط الخصائص من 6 إلى 4 ثم إلى 2 (Bottleneck / Latent Space)
    self.encoder = nn.Sequential(
        nn.Linear(input_dim, 4), nn.ReLU(), nn.Linear(4, 2), nn.ReLU()
    )
    # الـ Decoder: يعيد فك الضغط من 2 إلى 4 ثم يسترجع الـ 6 الأصلية
    self.decoder = nn.Sequential(
        nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, input_dim), nn.Sigmoid()
    )

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


# تهيئة النموذج ورفعه لمعالج الـ M4
model = NetworkAutoencoder(input_dim).to(device)

# دالة حساب الخطأ ودالة التحديث (Optimization)
criterion = nn.MSELoss()  # Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. دورة التدريب (Training Loop)
epochs = 30
print(f"بدء التدريب على {epochs} دورات (Epochs)...")

for epoch in range(1, epochs + 1):
  total_loss = 0.0
  for data, target in train_loader:
    data = data.to(device)
    target = target.to(device)

    # تصفير التدرجات، التمرير للأمام، حساب الخطأ، التمرير للخلف
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * data.size(0)

  epoch_loss = total_loss / len(train_loader.dataset)
  if epoch % 5 == 0 or epoch == 1:
    print(f"الدورة [{epoch}/{epochs}] - نسبة الخطأ (Loss): {epoch_loss:.6f}")

# 5. حفظ أوزان النموذج المدرب لاستخدامه لاحقاً
torch.save(model.state_dict(), "autoencoder_model.pth")
print("\n>>> تم تدريب النموذج وحفظه بنجاح في ملف 'autoencoder_model.pth'!")
