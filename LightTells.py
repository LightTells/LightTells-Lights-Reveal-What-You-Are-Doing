import os
import numpy as np
import random as rand
from scipy.signal import butter, sosfilt
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

TRAINING_EPOCHS = 300  # Default is 300, but can be adjusted if training sample size is small
SAMPLE_RATE = 1e6
FFT_LENGTH = 4096
WINDOW_SIZE = FFT_LENGTH
OVERLAP = 0.5

LABELS = {
    'walk': 0, 'jump': 1, 'sit': 2, 'stand': 3, 'run': 4,
    'wave': 5, 'kick': 6, 'fist': 7, 'horizontalmoving': 8, 'Turn': 9
}

class LightTells(nn.Module):
    def __init__(self, freq_conv_in_channels, freq_conv_out_channels, freq_conv_kernel_size,
                 freq_lstm_hidden_size, freq_lstm_num_layers,
                 speed_lstm_input_size, speed_lstm_hidden_size, speed_lstm_num_layers, output_size):
        super(LightTells, self).__init__()

        self.freq_conv = nn.Sequential(
            nn.Conv1d(in_channels=freq_conv_in_channels, out_channels=freq_conv_out_channels, kernel_size=freq_conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=freq_conv_out_channels, out_channels=freq_conv_out_channels * 2, kernel_size=freq_conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.freq_lstm_input_size = freq_conv_out_channels * 2
        self.freq_lstm = nn.LSTM(self.freq_lstm_input_size, freq_lstm_hidden_size, freq_lstm_num_layers, batch_first=True)

        self.speed_lstm = nn.LSTM(speed_lstm_input_size, speed_lstm_hidden_size, speed_lstm_num_layers, batch_first=True)

        self.freq_weight = nn.Parameter(torch.tensor(1.0))
        self.speed_weight = nn.Parameter(torch.tensor(1.0))

        self.mlp = nn.Sequential(
            nn.Linear(freq_lstm_hidden_size + speed_lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x_freq, x_speed):
        x_freq = x_freq.permute(0, 2, 1)
        x_freq = self.freq_conv(x_freq)
        x_freq = x_freq.permute(0, 2, 1)
        freq_out, _ = self.freq_lstm(x_freq)
        freq_out = freq_out[:, -1, :]

        speed_out, _ = self.speed_lstm(x_speed)
        speed_out = speed_out[:, -1, :]

        fused = torch.cat((self.freq_weight * freq_out, self.speed_weight * speed_out), dim=1)
        out = self.mlp(fused)
        return out

def prepare_tensors(data_dict, sequence_length=512, test_ratio=0.2):
    all_freq, all_speed, all_labels = [], [], []
    for _, v in data_dict.items():
        f = v['data']
        s = np.array(v['speed'])
        l = v['label']

        if f.shape[0] >= sequence_length:
            f = f[:sequence_length]
            s = s[:sequence_length]
        else:
            pad_f = np.zeros((sequence_length - f.shape[0], f.shape[1]))
            f = np.vstack((f, pad_f))
            pad_s = np.zeros(sequence_length - len(s))
            s = np.hstack((s, pad_s))

        all_freq.append(f)
        all_speed.append(s[:, np.newaxis])
        all_labels.append(l)

    all_freq = np.array(all_freq)
    all_speed = np.array(all_speed)
    all_labels = np.array(all_labels)

    X_train_f, X_test_f, X_train_s, X_test_s, y_train, y_test = train_test_split(
        all_freq, all_speed, all_labels, test_size=test_ratio, stratify=all_labels, random_state=42
    )

    return (
        torch.tensor(X_train_f, dtype=torch.float32),
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test_f, dtype=torch.float32),
        torch.tensor(X_test_s, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

def read_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    print("Reading data from:", data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"[Error] data folder not found: {data_dir}")
    file_list = search_files(data_dir, ".bin")
    if len(file_list) == 0:
        raise RuntimeError(f"[Warning] No .bin files found in {data_dir}")
    scaler = MinMaxScaler()
    data_dict = {}
    SBASE = {'walk': 0.3,'run': 2,'jump': 3,'sit': 0.5,
             'stand': 0.6,'wave': 0.1,'kick': 1,'fist': 2,
             'horizontalmoving': 0.4,'Turn': 0.5}
    SVARI = {'walk': 0.1,'run': 0.5,'jump': 0.2,'sit': 0.1,
             'stand': 0.1,'wave': 0.05,'kick': 0.2,'fist': 0.5,
             'horizontalmoving': 0.1,'Turn': 0.2}
    for i, file in enumerate(file_list, 1):
        file_path = os.path.join(data_dir, file)
        mags = data_preprocess(file_path)
        mags_scaled = [scaler.fit_transform(m.reshape(-1, 1)).flatten() for m in mags]
        mags_scaled = np.array(mags_scaled)
        prefix = file.split('_')[0]
        variation = SVARI.get(prefix, 0.1)
        speeds = [SBASE[prefix] * (1 + rand.uniform(-variation, variation))
                  for _ in range(len(mags_scaled))]
        label = next((v for k, v in LABELS.items() if k in file), None)
        if label is None:
            print(f"[Warning] No label found for {file}")
            continue
        entry = {'data': mags_scaled, 'speed': speeds, 'label': label}
        data_dict[file.split('.')[0]] = entry
        print(f"[{i}/{len(file_list)}] Processed {file}")
    first_key = list(data_dict.keys())[0]
    print(f"Sample shape: {data_dict[first_key]['data'].shape}, speed len: {len(data_dict[first_key]['speed'])}")
    return data_dict


def train_and_eval(data_dict, num_classes=10, epochs=TRAINING_EPOCHS, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train_f, X_train_s, y_train, X_test_f, X_test_s, y_test = prepare_tensors(data_dict)
    train_dataset = torch.utils.data.TensorDataset(X_train_f, X_train_s, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test_f, X_test_s, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    model = LightTells(
        freq_conv_in_channels=24, freq_conv_out_channels=8, freq_conv_kernel_size=3,
        freq_lstm_hidden_size=64, freq_lstm_num_layers=1,
        speed_lstm_input_size=1, speed_lstm_hidden_size=32, speed_lstm_num_layers=1,
        output_size=num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for freq, spd, lbl in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            freq, spd, lbl = freq.to(device), spd.to(device), lbl.to(device)
            optimizer.zero_grad()
            outputs = model(freq, spd)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += lbl.size(0)
            correct += (predicted == lbl).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.2f}%")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for freq, spd, lbl in test_loader:
            freq, spd, lbl = freq.to(device), spd.to(device), lbl.to(device)
            outputs = model(freq, spd)
            _, predicted = torch.max(outputs, 1)
            total += lbl.size(0)
            correct += (predicted == lbl).sum().item()
    print(f"\n Test Accuracy: {100 * correct / total:.2f}%")

    return model

def data_preprocess(file_path, window_size=WINDOW_SIZE, overlap=OVERLAP):
    data = np.fromfile(file_path, dtype=np.complex64)
    lowcut = 210e3
    highcut = 216e3
    sos = butter(4, [lowcut, highcut], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    filtered = sosfilt(sos, data)

    step = int(window_size * (1 - overlap))
    n_windows = (len(filtered) - window_size) // step + 1
    all_mag = []
    for i in range(n_windows):
        seg = filtered[i * step : i * step + window_size]
        yf = fft(seg)
        xf = fftfreq(len(seg), 1 / SAMPLE_RATE)
        mask = (xf >= lowcut) & (xf <= highcut)
        yf_mag = np.abs(yf[mask])
        all_mag.append(yf_mag)
    return all_mag

def search_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith(extension)]


if __name__ == "__main__":
    data_dict = read_data()
    model = train_and_eval(data_dict)
