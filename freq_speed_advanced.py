import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from nns import *
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

MARKER = 'o'
LINE_STYLE = '-'
LINE_WIDTH = 20
color1 = '#bababa'   #grey
color2 = '#cde1ec'   #light blue
color3 = '#8ec1da'   #blue
color4 = '#2066a8'   #deep blue
color5 = '#082a54'   #dark blue

color6 = '#f4a582'   #pink

class MixedLSTMModel(nn.Module):
    def __init__(self, freq_conv_in_channels, freq_conv_out_channels, freq_conv_kernel_size,
                 freq_lstm_hidden_size, freq_lstm_num_layers,
                 time_lstm_input_size, time_lstm_hidden_size, time_lstm_num_layers, output_size):
        super(MixedLSTMModel, self).__init__()

        self.freq_conv = nn.Sequential(
            nn.Conv1d(in_channels=freq_conv_in_channels, out_channels=freq_conv_out_channels, 
                      kernel_size=freq_conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=freq_conv_out_channels, out_channels=freq_conv_out_channels * 2, 
                      kernel_size=freq_conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=freq_conv_out_channels*2, out_channels=freq_conv_out_channels * 4, 
                      kernel_size=freq_conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.freq_lstm_input_size = freq_conv_out_channels * 4
        self.freq_hidden_size = freq_lstm_hidden_size
        self.freq_num_layers = freq_lstm_num_layers

        self.freq_lstm = nn.LSTM(self.freq_lstm_input_size, freq_lstm_hidden_size, freq_lstm_num_layers, 
                                 batch_first=True)
        

        self.time_lstm_input_size = time_lstm_input_size  
        self.time_hidden_size = time_lstm_hidden_size
        self.time_num_layers = time_lstm_num_layers

        self.time_lstm = nn.LSTM(self.time_lstm_input_size, time_lstm_hidden_size, time_lstm_num_layers, 
                                 batch_first=True)
        
        self.freq_weight = nn.Parameter(torch.tensor(1.0))
        self.time_weight = nn.Parameter(torch.tensor(1.0))
        
        self.mlp = nn.Sequential(
            nn.Linear(freq_lstm_hidden_size + time_lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )


    def forward(self, x_freq, x_time):
        x_freq = x_freq.permute(0, 2, 1)
        x_freq = self.freq_conv(x_freq)
        x_freq = x_freq.permute(0, 2, 1)

        h0_freq = torch.zeros(self.freq_num_layers, x_freq.size(0), self.freq_hidden_size).to(x_freq.device)
        c0_freq = torch.zeros(self.freq_num_layers, x_freq.size(0), self.freq_hidden_size).to(x_freq.device)
        freq_out_lstm, _ = self.freq_lstm(x_freq, (h0_freq, c0_freq))    
        freq_out_lstm  = freq_out_lstm[:, -1, :]

        h0_time = torch.zeros(self.time_num_layers, x_time.size(0), self.time_hidden_size).to(x_time.device)
        c0_time = torch.zeros(self.time_num_layers, x_time.size(0), self.time_hidden_size).to(x_time.device)
        time_out_lstm, _ = self.time_lstm(x_time, (h0_time, c0_time))
        time_out_lstm = time_out_lstm[:, -1, :]

        weighted_freq = self.freq_weight * freq_out_lstm
        weighted_time = self.time_weight * time_out_lstm

        combined_features = torch.cat((weighted_freq, weighted_time), dim=1)

        out = self.mlp(combined_features)

        return out

def time_statistics(all_time):
    # print(all_time.shape)
    # positive_value = np.where(all_time > 0, all_time, np.nan)
    mean_values = np.mean(all_time, axis=1)
    std_dev = np.std(all_time, axis=1)
    variances = np.var(all_time, axis=1)
    min_values = np.min(all_time, axis=1)
    max_values = np.max(all_time, axis=1)
    q1 = np.percentile(all_time, 25, axis=1)
    q2 = np.percentile(all_time, 50, axis=1)
    q3 = np.percentile(all_time, 75, axis=1)
    skewness = skew(all_time, axis=1)
    kurtosis_values = kurtosis(all_time, axis=1)
    
    return_value = np.stack((mean_values, std_dev, variances, max_values, 
                                q1, q2, q3, skewness, kurtosis_values), axis=1)

    # print(return_value.shape)
    # sys.exit()
    return return_value

def speed_statistics(all_speed):
    mean_values = np.mean(all_speed, axis=1)
    variances = np.var(all_speed, axis=1)
    
    return_value = np.stack((mean_values, variances), axis=1)

    return return_value

def load_dict_data(data_dict, sequence_length, test_size=None, mode=None, shuffle=True):
    # convert the dictionary to a list of NumPy arrays
    if mode is None and test_size is None:
        sys.exit("Please provide the mode and test_size")
    all_labels = []
    all_keys = []
    all_freq = []
    all_time = []
    all_speed = []

    for file_name, content in data_dict.items():
        label = content['label']
        freq_data = content['data']
        time_data = content['time_domain']
        speed_data = content['speed']

        if freq_data.shape[0] > sequence_length:
            freq_data = freq_data[:sequence_length]
            time_data = time_data[:sequence_length]
            speed_data = speed_data[:sequence_length]
        else:
            padding = np.zeros((sequence_length - freq_data.shape[0], freq_data.shape[1]))
            freq_data = np.vstack((freq_data, padding))
            time_data = np.vstack((time_data, padding))

            if speed_data.ndim == 2:
                padding = np.zeros((sequence_length - speed_data.shape[0], 1))
                speed_data = np.vstack((speed_data, padding))
            else:
                padding = np.zeros(sequence_length - len(speed_data))
                speed_data = np.vstack((speed_data[:, np.newaxis], padding[:, np.newaxis]))


        all_labels.append(label)
        all_keys.append(file_name)
        all_freq.append(freq_data)
        all_time.append(time_data)
        all_speed.append(speed_data)

    all_labels = np.array(all_labels)
    all_keys = np.array(all_keys)
    all_freq = np.array(all_freq)
    all_time = np.array(all_time)
    all_time = time_statistics(all_time)
    all_speed = np.array(all_speed)
    all_speed = speed_statistics(all_speed).squeeze(-1)

    # print(all_freq.shape)
    # print(all_time.shape)

    # Split the data into training and testing sets
    if mode == 'train':
        X_train_freq,X_train_time,X_train_speed,y_train,train_keys = all_freq, all_time, all_speed, all_labels, all_keys
        X_test_freq,X_test_time,X_test_speed, y_test, test_keys = None, None, None, None, None
        X_train_freq_tensor = torch.tensor(X_train_freq, dtype=torch.float32)
        X_train_time_tensor = torch.tensor(X_train_time, dtype=torch.float32)
        X_train_speed_tensor = torch.tensor(X_train_speed, dtype=torch.float32)
        X_test_freq_tensor = None
        X_test_time_tensor = None
        X_test_speed_tensor = None
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = None
    elif mode == 'test':
        X_test_freq, X_test_time, X_test_speed, y_test, test_keys = all_freq, all_time, all_speed, all_labels, all_keys
        X_train_freq, X_train_time, X_train_speed, y_train, train_keys = None, None, None, None, None
        X_train_freq_tensor = None
        X_train_time_tensor = None
        X_train_speed_tensor = None
        X_test_freq_tensor = torch.tensor(X_test_freq, dtype=torch.float32)
        X_test_time_tensor = torch.tensor(X_test_time, dtype=torch.float32)
        X_test_speed_tensor = torch.tensor(X_test_speed, dtype=torch.float32)
        y_train_tensor = None
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    elif mode == 'mixed':
        sys.exit("Not implemented yet")

    return_list = [X_train_freq_tensor, X_train_time_tensor, X_train_speed_tensor,
                    X_test_freq_tensor, X_test_time_tensor, X_test_speed_tensor,
                    y_train_tensor, y_test_tensor, train_keys, test_keys]

    return return_list

def load_matrix(matrix, location, test_size, msg, fig, ax):
    # actions = ['walk', 'jump', 'sit', 'stand', 'run', 'wave', 'kick']
    actions = ['fall', 'slownodding','fastnodding']
    dict_matrix = {}
    for i in range(len(actions)):
        for j in range(len(actions)):
            if i != j:
                dict_matrix['-'.join([actions[i], actions[j]])] = 0

    for data in matrix:
        action_pair = '-'.join([data[0], data[1]])
        if action_pair in dict_matrix:
            dict_matrix[action_pair] += 1

    matrix_size = len(actions)
    confusion_matrix = np.zeros((matrix_size, matrix_size))

    for key, value in dict_matrix.items():
        action1, action2 = key.split('-')
        i = actions.index(action1)
        j = actions.index(action2)
        confusion_matrix[i, j] = value

    row_sums = confusion_matrix.sum(axis=1)
    confusion_matrix_with_sums = np.hstack((confusion_matrix, row_sums.reshape(-1, 1)))

    actions_with_sum = actions + ["SUM"]

    # cax = ax.matshow(confusion_matrix, cmap='Blues')
    cax = ax.matshow(confusion_matrix_with_sums, cmap='Blues', vmin=0, vmax=20)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(actions_with_sum)))
    ax.set_yticks(np.arange(len(actions)))
    ax.set_xticklabels(actions_with_sum, rotation=45, ha="left")
    ax.set_yticklabels(actions)
    for (i, j), val in np.ndenumerate(confusion_matrix_with_sums):
        ax.text(j, i, f'{int(val)}', ha='center', va='center', color='black')
    ax.set_title(f"{location}, {test_size}, {msg}")
    # plt.savefig(os.path.join(PATH,'Accuracy_'+location+'.png'))


    
def define_model(freq_conv_in_channels=None, freq_conv_out_channels=32, freq_conv_kernel_size=3,
                 freq_lstm_hidden_size=128, freq_lstm_num_layers=3,
                 time_lstm_input_size=None,
                 time_lstm_hidden_size=32, time_lstm_num_layers=2, output_size=7):
    model = MixedLSTMModel(freq_conv_in_channels, freq_conv_out_channels, freq_conv_kernel_size,
                 freq_lstm_hidden_size, freq_lstm_num_layers,
                 time_lstm_input_size,
                 time_lstm_hidden_size, time_lstm_num_layers, output_size)
    return model

def train_model(model, X_train_freq, X_train_time, y_train, criterion, optimizer, device, num_epochs=900):
    model.to(device)
    train_losses = []
    weighted_freq = []
    weighted_time = []

    for epoch in range(num_epochs):
        model.train()
        X_train_freq, X_train_time, y_train = \
            X_train_freq.to(device), X_train_time.to(device), y_train.to(device)
        outputs = model(X_train_freq, X_train_time)
        loss = criterion(outputs.squeeze(), y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        weighted_freq.append(model.freq_weight.item())
        weighted_time.append(model.time_weight.item())
        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return train_losses, weighted_freq, weighted_time

def test_model(model, X_test_freq, X_test_time, y_test, label_map, test_keys, device):
    model.eval()
    with torch.no_grad():
        X_test_freq, X_test_time, y_test = X_test_freq.to(device), X_test_time.to(device), y_test.to(device)
        outputs = model(X_test_freq, X_test_time)
        predicted = torch.argmax(outputs, dim=1).cpu().numpy()
        accuracy = np.mean(predicted == y_test.cpu().numpy())
        accuracy_msg = f'Accuracy: {accuracy * 100:.2f}%'
        print(accuracy_msg)

    result_matrix = []
    for i, (pred, true_label) in enumerate(zip(predicted, y_test.cpu().numpy())):
        if pred != true_label:
            result_matrix.append([test_keys[i].split('_')[0], label_map[pred]])
    
    return result_matrix, accuracy_msg
    

###########################################
# ========== Model Save & Load ===========
###########################################

def save_trained_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved successfully to {path}")


def load_trained_model(path, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    print(f"Model loaded successfully from {path}")
    return model

########################################################################################################
def only_lab_training(epoch=300, mode='default'):
    if mode == 'default' or mode == 'obstacle':
        model_loaded = load_trained_model(
        r'D:\workarea\my_paper\LightTell\Work_Ambient_sensing\nn\default.pth',
        MixedLSTMModel,
        freq_conv_in_channels=786, 
        freq_conv_out_channels=32,
        freq_conv_kernel_size=3,
        freq_lstm_hidden_size=128,
        freq_lstm_num_layers=3,
        time_lstm_input_size=786,
        time_lstm_hidden_size=32,
        time_lstm_num_layers=2,
        output_size=7          
        )
    elif mode == 'inc':
        model_loaded = load_trained_model(
        r'D:\workarea\my_paper\LightTell\Work_Ambient_sensing\nn\inc.pth',
        MixedLSTMModel,
        freq_conv_in_channels=786, 
        freq_conv_out_channels=32,
        freq_conv_kernel_size=3,
        freq_lstm_hidden_size=128,
        freq_lstm_num_layers=3,
        time_lstm_input_size=786,
        time_lstm_hidden_size=32,
        time_lstm_num_layers=2,
        output_size=7          
        )
    
    SAVE_PATH = os.path.join('result','new_speed', 'freq_time')
    os.makedirs(SAVE_PATH, exist_ok=True)

    with open(os.path.join('data','dict_data','data_lab_speed_no_norm_time.pkl'), 'rb') as f:
        data_dict = pickle.load(f)
    with open(os.path.join('data','dict_data','test_lab_speed_no_norm_time.pkl'), 'rb') as f:
        data_dict_test_lab = pickle.load(f)
    # with open(os.path.join('data','dict_data','test_classroom_speed.pkl'), 'rb') as f:
    #     data_dict_test_classroom = pickle.load(f)
    model_loaded.eval()
    X_train_freq_tensor,X_train_time_tensor,X_train_speed_tensor, _,_,_,y_train_tensor,_, _,_ = \
        load_dict_data(data_dict,62, mode='train', shuffle=True)
    _,_,_, X_test_freq_tensor_lab,X_test_time_tensor_lab,X_test_speed_tensor_lab,_,y_test_tensor_lab,_,test_keys_lab = \
        load_dict_data(data_dict_test_lab,62, mode='test', shuffle=True)
    # _,_,_, X_test_freq_tensor_classroom,X_test_time_tensor_classroom,X_test_speed_tensor_classroom, _,\
    #       y_test_tensor_classroom, _,test_keys_classroom = load_dict_data(data_dict_test_classroom,62, mode='test', shuffle=True)

    model = define_model(freq_conv_in_channels=X_train_freq_tensor.shape[2], 
                         time_lstm_input_size=X_train_time_tensor.shape[2])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_losses, freq_weight, time_weight  = train_model(model, X_train_freq_tensor,X_train_time_tensor, 
                                                             y_train_tensor, criterion, optimizer, device, 
                                                             num_epochs=700)
    
    # save_trained_model(model, r'D:\workarea\my_paper\LightTell\Work_Ambient_sensing\nn\inc.pth')

    model.eval()
    X_train_freq_tensor = X_train_freq_tensor.to(device)
    X_train_time_tensor = X_train_time_tensor.to(device)
    X_train_speed_tensor = X_train_speed_tensor.to(device)
    output = model(X_train_freq_tensor, X_train_time_tensor)
    probabilities = output.softmax(dim=1)
    # print(probabilities.shape)
    # print(X_train_speed_tensor.shape)
    combined_features = torch.cat((probabilities, X_train_speed_tensor), dim=1)
    # print(combined_features.shape)

    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features.cpu().detach().numpy())

    svm = SVC(kernel='poly', degree=3, C=1, gamma='scale')
    svm.fit(combined_features, y_train_tensor.cpu())

    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(combined_features)
    # plt.figure(figsize=(10, 8))
    # for label in np.unique(y_train_tensor.cpu()):
    #     plt.scatter(
    #         reduced_features[y_train_tensor.cpu() == label, 0],
    #         reduced_features[y_train_tensor.cpu() == label, 1],
    #         label=f"Class {label}",
    #         alpha=0.7
    #     )
    # plt.title("Visualization of Combined Features")
    # plt.xlabel("Component 1")
    # plt.ylabel("Component 2")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # label_map = {0: 'walk', 1: 'jump', 2: 'sit', 3: 'stand', 4: 'run', 5: 'wave', 6: 'kick'}
    label_map = {0: 'fall', 1: 'slownodding', 2: 'fastnodding'}

    X_test_freq_tensor_lab = X_test_freq_tensor_lab.to(device)
    X_test_time_tensor_lab = X_test_time_tensor_lab.to(device)
    X_test_speed_tensor_lab = X_test_speed_tensor_lab.to(device)
    output = model(X_test_freq_tensor_lab, X_test_time_tensor_lab)
    probabilities = output.softmax(dim=1)
    combined_features = torch.cat((probabilities, X_test_speed_tensor_lab), dim=1)
    combined_features = scaler.transform(combined_features.cpu().detach().numpy())
    predicted = svm.predict(combined_features)
    accuracy = np.mean(predicted == y_test_tensor_lab.numpy())
    # print(f'LAB Accuracy: {accuracy * 100:.2f}%')

    draw_result(mode)
    
    # X_test_freq_tensor_classroom = X_test_freq_tensor_classroom.to(device)
    # X_test_time_tensor_classroom = X_test_time_tensor_classroom.to(device)
    # X_test_speed_tensor_classroom = X_test_speed_tensor_classroom.to(device)
    # output = model(X_test_freq_tensor_classroom, X_test_time_tensor_classroom)
    # probabilities = output.softmax(dim=1)
    # combined_features = torch.cat((probabilities, X_test_speed_tensor_classroom), dim=1)
    # combined_features = scaler.transform(combined_features.cpu().detach().numpy())
    # predicted = svm.predict(combined_features)
    # # accuracy = np.mean(predicted == y_test_tensor_classroom.numpy())
    # print(f'Classroom Accuracy: {accuracy * 100:.2f}%')
    
    
    # lab_matrix, lab_msg = test_model(model, X_test_freq_tensor_lab, X_test_time_tensor_lab,
    #                                   y_test_tensor_lab, label_map, test_keys_lab, device)
    # classroom_matrix, classroom_msg = test_model(model, X_test_freq_tensor_classroom, X_test_time_tensor_classroom,
    #                                               y_test_tensor_classroom, label_map, test_keys_classroom, device)
    
    # fig, axes  = plt.subplots(2, 2, figsize=(15, 10))

    # axes[1,0].plot(range(1, len(train_losses) + 1), train_losses)
    # axes[1,0].set_xlabel('Epoch')
    # axes[1,0].set_ylabel('Loss')
    # axes[1,0].set_title('Training Loss Over Epochs')

    # axes[1,1].plot(range(1, len(freq_weight) + 1), freq_weight, label='Freq')
    # axes[1,1].plot(range(1, len(time_weight) + 1), time_weight, label='Time')
    # axes[1,1].set_xlabel('Epoch')
    # axes[1,1].set_ylabel('Weight')
    # axes[1,1].set_title('Weighted Freq and Time')
    # axes[1,1].legend()
    
    # load_matrix(lab_matrix, 'lab', '140 test samples', lab_msg, fig, axes[0,0])
    # load_matrix(classroom_matrix, 'classroom', '140 test samples', classroom_msg, fig, axes[0,1])

    # plt.tight_layout()
    # plt.savefig(os.path.join(SAVE_PATH, 'only_lab_training.png'))

def draw_result(mode):
    if mode == 'inc':
        dicts_c = [inc0c, inc10c, inc20c, inc30c, inc40c, inc50c, inc60c, inc70c]
        dicts_l = [inc0l, inc10l, inc20l, inc30l, inc40l, inc50l, inc60l, inc70l]
        labels = [0, 10, 20, 30, 40, 50, 60, 70]

        def dict_to_matrix(d):
            m = np.zeros((12,12))
            for k,v in d.items():
                i,j = map(int, k.split(':'))
                m[i-1,j-1] = v
            return m

        def diag_mean(m):
            return np.mean(np.diag(m))

        mats_c = [dict_to_matrix(d) for d in dicts_c]
        mats_l = [dict_to_matrix(d) for d in dicts_l]

        means_c = [diag_mean(m) for m in mats_c]
        means_l = [diag_mean(m) for m in mats_l]

        fig, axes = plt.subplots(3, 8, figsize=(32, 12))

        for i, m in enumerate(mats_c):
            ax = axes[0, i]
            im = ax.imshow(m, cmap='Blues', vmin=0, vmax=100)
            ax.set_title(f'inc{labels[i]}c', fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])
            for (x, y), val in np.ndenumerate(m):
                ax.text(y, x, f'{int(val)}', ha='center', va='center', fontsize=7, color='black')

        for i, m in enumerate(mats_l):
            ax = axes[1, i]
            im = ax.imshow(m, cmap='Oranges', vmin=0, vmax=100)
            ax.set_title(f'inc{labels[i]}l', fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])
            for (x, y), val in np.ndenumerate(m):
                ax.text(y, x, f'{int(val)}', ha='center', va='center', fontsize=7, color='black')

        ax_line = axes[2, 0]
        for j in range(1, 8):
            axes[2, j].axis('off')

        ax_line.plot(labels, means_c, '-o', label='classroom', color='blue')
        ax_line.plot(labels, means_l, '-s', label='lobby', color='orange')
        ax_line.set_xlabel('Increment samples', fontsize=14)
        ax_line.set_ylabel('Accuracy', fontsize=14)
        ax_line.set_title('Accuracy vs Increment samples', fontsize=16)
        ax_line.set_ylim(0, 100)
        ax_line.legend(fontsize=12)
        ax_line.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(18, 14))

        ax.plot(labels, means_c, '-o', label='classroom', color=color1, linewidth=LINE_WIDTH, marker=MARKER, markersize=60)
        ax.plot(labels, means_l, '-s', label='lobby', color=color4, linewidth=LINE_WIDTH, marker=MARKER, markersize=60)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(50, 100)
        ax.grid(True, linestyle='--')
        for spine in ax.spines.values():
            spine.set_edgecolor('black') 
            spine.set_linewidth(5)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        plt.savefig('accuracy different increment samples.pdf', format='pdf', bbox_inches='tight')
        plt.show()

    elif mode == 'obstacle':
        dicts_c =[labbo10, labbo20, labbo30, labbo40, labbo50, labbo60, labbo70, labbo80, labbo90, labbo100]
        def dict_to_matrix(d):
            m = np.zeros((12,12))
            for k,v in d.items():
                i,j = map(int, k.split(':'))
                m[i-1,j-1] = v
            return m

        def diag_mean(m):
            return np.mean(np.diag(m))
        
        mats_c = [dict_to_matrix(d) for d in dicts_c]
        means_c = [diag_mean(m) for m in mats_c]

        fig, axes = plt.subplots(1, 10, figsize=(10, 8))
        for i, m in enumerate(mats_c):
            ax = axes[i]
            im = ax.imshow(m, cmap='Blues', vmin=0, vmax=100)
            title = 'labbo' if i == 0 else 'labio'
            ax.set_title(title, fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])
            for (x, y), val in np.ndenumerate(m):
                ax.text(y, x, f'{int(val)}', ha='center', va='center', fontsize=7, color='black')
        
        plt.tight_layout()
        plt.show()

        bar_width = 0.2
        fig, ax = plt.subplots(figsize=(18, 14))
        print(means_c)
        ax.plot(
            range(len(means_c)),
            means_c,
            '-o', label='classroom', color=color3, linewidth=LINE_WIDTH, marker=MARKER, markersize=60
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--')
        for spine in ax.spines.values():
            spine.set_edgecolor('black') 
            spine.set_linewidth(5)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        plt.savefig('accuracy different obstacle percentage.pdf', format='pdf', bbox_inches='tight')
        plt.show()

    elif mode == 'default':
        dicts_c =[labwave, labwnave]
        def dict_to_matrix(d):
            m = np.zeros((12,12))
            for k,v in d.items():
                i,j = map(int, k.split(':'))
                m[i-1,j-1] = v
            return m

        def diag_mean(m):
            return np.mean(np.diag(m))
        
        mats_c = [dict_to_matrix(d) for d in dicts_c]
        means_c = [diag_mean(m) for m in mats_c]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for i, m in enumerate(mats_c):
            ax = axes[i]
            im = ax.imshow(m, cmap='Blues', vmin=0, vmax=100)
            title = 'labwave' if i == 0 else 'labwnave'
            ax.set_title(title, fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])
            for (x, y), val in np.ndenumerate(m):
                ax.text(y, x, f'{int(val)}', ha='center', va='center', fontsize=7, color='black')
        
        plt.tight_layout()
        plt.show()

########################################################################################################
# def lab_to_classroom_training():
#     SAVE_PATH = os.path.join('result', 'new_speed','freq_speed')
#     os.makedirs(SAVE_PATH, exist_ok=True)

#     with open(os.path.join('data','dict_data','data_lab_speed.pkl'), 'rb') as f:
#         data_dict_lab = pickle.load(f)
#     with open(os.path.join('data','dict_data','data_classroom_speed.pkl'), 'rb') as f:
#         data_dict_classroom = pickle.load(f)
#     with open(os.path.join('data','dict_data','test_lab_speed.pkl'), 'rb') as f:
#         data_dict_test_lab = pickle.load(f)
#     with open(os.path.join('data','dict_data','test_classroom_speed.pkl'), 'rb') as f:
#         data_dict_test_classroom = pickle.load(f)
    
#     X_train_freq_tensor_lab,X_train_speed_tensor_lab, _, _,y_train_tensor_lab,_, X_train, _, _ = \
#         load_dict_data(data_dict_lab,62, mode='train', shuffle=True)
#     X_train_freq_tensor_classroom,X_train_speed_tensor_classroom, _, _,y_train_tensor_classroom,_, \
#         X_train, _, _ = load_dict_data(data_dict_classroom,62, mode='train', shuffle=True)
#     _,_, X_test_freq_tensor_lab,X_test_speed_tensor_lab, _, y_test_tensor_lab, _, _, test_keys_lab = load_dict_data(
#         data_dict_test_lab,62, mode='test', shuffle=True)
#     _,_, X_test_freq_tensor_classroom,X_test_speed_tensor_classroom, _, y_test_tensor_classroom, _, _, \
#         test_keys_classroom = load_dict_data(data_dict_test_classroom,62, mode='test', shuffle=True)

   
#     model = define_model(X_train.shape[2], X_train_speed_tensor_lab.shape[1])
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     train_losses_lab, lstm_weight_lab, mlp_weight_lab = \
#         train_model(model, X_train_freq_tensor_lab, X_train_speed_tensor_lab, y_train_tensor_lab, criterion, 
#                     optimizer, device, num_epochs=1000)
#     # Incremental training
#     # Freeze the LSTM layers
#     for param in model.lstm.parameters():
#         param.requires_grad = False
#     # Freeze the Convolutional layers
#     for param in model.conv.parameters():
#         param.requires_grad = False
#     # Freeze the MLP layers
#     for param in model.fc.parameters():
#         param.requires_grad = False
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

#     train_losses_classroom, lstm_weight_classroom, mlp_weight_classroom = \
#         train_model(model, X_train_freq_tensor_classroom, X_train_speed_tensor_classroom, y_train_tensor_classroom, 
#                     criterion, optimizer, device, num_epochs=1000)


#     label_map = {0: 'walk', 1: 'jump', 2: 'sit', 3: 'stand', 4: 'run', 5: 'wave', 6: 'kick'}
#     lab_matrix, lab_msg = test_model(model, X_test_freq_tensor_lab, X_test_speed_tensor_lab,
#                                       y_test_tensor_lab, label_map, test_keys_lab, device)
#     classroom_matrix, classroom_msg = test_model(model, X_test_freq_tensor_classroom, X_test_speed_tensor_classroom,
#                                                   y_test_tensor_classroom, label_map, test_keys_classroom, device)
    
#     fig, axes  = plt.subplots(2, 2, figsize=(15, 10))

#     axes[1,0].plot(range(1, len(train_losses_lab) + len(train_losses_classroom) + 1), 
#                    train_losses_lab+train_losses_classroom)
#     axes[1,0].set_xlabel('Epoch')
#     axes[1,0].set_ylabel('Loss')
#     axes[1,0].set_title('Training Loss Over Epochs')

#     axes[1,1].plot(range(1, len(lstm_weight_lab) + len(lstm_weight_classroom) + 1), 
#                    lstm_weight_lab+lstm_weight_classroom, label='LSTM')
#     axes[1,1].plot(range(1, len(mlp_weight_lab) + len(mlp_weight_classroom) + 1),
#                      mlp_weight_lab+mlp_weight_classroom, label='MLP')
#     axes[1,1].set_xlabel('Epoch')
#     axes[1,1].set_ylabel('Weight')
#     axes[1,1].set_title('Weighted LSTM and MLP')
#     axes[1,1].legend()
    
#     load_matrix(lab_matrix, 'lab', '140 test samples', lab_msg, fig, axes[0,0])
#     load_matrix(classroom_matrix, 'classroom', '140 test samples', classroom_msg, fig, axes[0,1])

#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_PATH, 'incremental_training.png'))

    



if __name__ == '__main__':
    # statistics_only_lab_training()
    only_lab_training(epoch=300, mode='obstacle')
    # lab_and_classroom_training(epoch=700)
    # lab_to_classroom_training(epoch=1000)
        # draw_result(mode='obstacle')

    