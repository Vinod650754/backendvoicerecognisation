import os
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader


class CommandDataset(Dataset):
    """Expects a folder where each subfolder is a class label containing audio files."""

    def __init__(self, root_dir):
        self.samples = []
        self.labels_map = {}
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        classes.sort()
        for idx, label in enumerate(classes):
            self.labels_map[label] = idx
            label_dir = os.path.join(root_dir, label)
            for fname in os.listdir(label_dir):
                if not fname.lower().endswith(('.wav', '.flac', '.mp3')):
                    continue
                self.samples.append((os.path.join(label_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data, sr = sf.read(path, dtype='float32')
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel, label


class CommandModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(root_dir, epochs=12, batch_size=8, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CommandDataset(root_dir)
    if len(dataset) == 0:
        raise RuntimeError("No training data found in " + root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = CommandModel(num_classes=len(dataset.labels_map)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x = x.to(device)
            y = torch.tensor(y, dtype=torch.long).to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, loss: {total/len(loader):.4f}")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'labels_map': dataset.labels_map,
    }
    torch.save(checkpoint, os.path.join(root_dir, 'command_model.pth'))
    print('Saved command model to', os.path.join(root_dir, 'command_model.pth'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/commands/train', help='Training root folder')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    train(args.data, epochs=args.epochs, batch_size=args.batch)
