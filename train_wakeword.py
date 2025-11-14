import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader


class WakeWordDataset(Dataset):
    """Simple dataset expecting folder structure:
    root/train/wakeword/*.wav
    root/train/noise/*.wav
    root/test/...
    Returns mel-spectrogram tensors and labels (1 wakeword, 0 noise).
    """

    def __init__(self, root_dir):
        self.samples = []
        for cls, label in [("wakeword", 1), ("noise", 0)]:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(('.wav', '.flac', '.mp3')):
                    continue
                self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Load via soundfile to avoid optional torchaudio backends
        data, sr = sf.read(path, dtype='float32')
        # data shape: (nsamples,) or (nsamples, nchannels)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)
        # Resample to 16k if needed
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel, label


class WakeWordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Use adaptive pooling to produce a fixed spatial size regardless of input time dimension
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(data_root, epochs=10, batch_size=8, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = WakeWordDataset(os.path.join(data_root, "train"))
    test_ds = WakeWordDataset(os.path.join(data_root, "test"))
    if len(train_ds) == 0:
        raise RuntimeError("No training data found in " + os.path.join(data_root, "train"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = WakeWordModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, train loss: {running/len(train_loader):.4f}")

        # quick eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = torch.tensor(y, dtype=torch.long).to(device)
                out = model(x)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        if total > 0:
            print(f"  Eval acc: {correct/total:.3f}")

    torch.save(model.state_dict(), os.path.join(data_root, "wakeword_model.pth"))
    print("Saved wakeword model to", os.path.join(data_root, "wakeword_model.pth"))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/wakewords', help='Data root folder')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    train(args.data, epochs=args.epochs, batch_size=args.batch)
