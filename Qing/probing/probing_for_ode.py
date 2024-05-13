import pickle
import sys
import os

import numpy as np
import argparse

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

torch.manual_seed(42)

class MMDetect(Dataset):
    def __init__(self, dataset, model, split, feature_key):
        if feature_key == "sentence":
            self.data = pickle.load(
                open(os.path.join("Qing/data/", '_'.join([dataset, model, split]) + "_sentence.bin"), "rb" ))
        elif feature_key == "question":
            self.data = pickle.load(
                open(os.path.join("Qing/data/", '_'.join([dataset, model, split]) + "_question.bin"), "rb"))
        elif feature_key == "response":
            self.data = pickle.load(
                open(os.path.join("Qing/data/", '_'.join([dataset, model, split]) + "_response.bin"), "rb"))
        print(len(self.data))
        qingli = 3
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        hidden_states = data["features"]
        label = data["label"]

        features = torch.tensor(hidden_states, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return features, label


class MMDetect_Train(Dataset):
    def __init__(self, dataset, model, split, feature_key):
        if feature_key == "sentence":
            self.data = pickle.load(
                open(os.path.join("Qing/data_clear_figure/", '_'.join([dataset, model, split]) + "_sentence.bin"), "rb" ))
        elif feature_key == "question":
            self.data = pickle.load(
                open(os.path.join("Qing/data_clear_figure/", '_'.join([dataset, model, split]) + "_question.bin"), "rb"))
        elif feature_key == "response":
            self.data = pickle.load(
                open(os.path.join("Qing/data_clear_figure/", '_'.join([dataset, model, split]) + "_response.bin"), "rb"))
        print(len(self.data))
        qingli = 3
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        hidden_states = data["features"]
        label = data["label"]

        features = torch.tensor(hidden_states, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return features, label

def get_train_data_loaders(dataset, model_name, feature_key, batch_size=64, shuffle=True):
    train_dataset = MMDetect(dataset, model=model_name, split="train", feature_key=feature_key)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader

def get_val_data_loaders(dataset, model_name, feature_key, batch_size=64, shuffle=True):
    val_dataset = MMDetect(dataset, model=model_name, split="val", feature_key=feature_key)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader

def get_test_data_loaders(dataset, model_name, feature_key, batch_size=64, shuffle=True):
    test_dataset = MMDetect(dataset, model=model_name, split="test", feature_key=feature_key)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


class Probe(nn.Module):
    def __init__(self, input_dim):
        super(Probe, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.sigmoid(x)


def train(model, criterion, optimizer, train_loader, test_loader, val_loader, num_epochs, device):
    model.train()
    best_accuracy, best_precision, best_recall, best_f1, best_pr_auc = 0, 0, 0, 0, 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # print(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
        print("Test:")
        accuracy, precision, recall, f1, pr_auc, all_outputs_prob, all_labels = evaluate(model, test_loader, device)
        print("Val:")
        accuracy_val, precision_val, recall_val, f1_val, pr_auc_val, all_outputs_prob_val, all_labels_val = evaluate(model, val_loader, device)


def evaluate(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_outputs_prob = []

    TP, FP, FN = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs)

            # Store labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_outputs_prob.extend(outputs.cpu().numpy())
            TP += ((predicted == 1) & (labels.unsqueeze(1) == 1)).sum().item()
            FP += ((predicted == 1) & (labels.unsqueeze(1) == 0)).sum().item()
            FN += ((predicted == 0) & (labels.unsqueeze(1) == 1)).sum().item()

    # Convert lists to numpy arrays for scikit-learn
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions).flatten()  # Flatten in case the predictions are in a column vector

    # Calculate metrics using scikit-learn
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, pos_label=1)
    recall = recall_score(all_labels, all_predictions, pos_label=1)
    f1 = f1_score(all_labels, all_predictions, pos_label=1)

    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_outputs_prob)
    pr_auc = auc(recall_curve, precision_curve)
    new_precision = TP / (TP + FP) if TP + FP > 0 else 0
    new_recall = TP / (TP + FN) if TP + FN > 0 else 0
    print(
        f"Accuracy: {accuracy:.2f} Precision: {precision:.2f} Recall: {recall:.2f} F1 Score: {f1:.2f} PR-AUC: {pr_auc:.2f}")
    # print(all_predictions)
    return accuracy, precision, recall, f1, pr_auc, all_outputs_prob, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='truthful_qa')   # pope_pop, mhal, self_data
    parser.add_argument("--model", type=str, default='llama15_7b')  # llava16_moe, llava15_7b, llava16_7b
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--feature_key", type=str, default="response")  # response, sentence

    args = parser.parse_args()

    train_loader = get_train_data_loaders(args.dataset, model_name=args.model, feature_key=args.feature_key,
                                                  batch_size=args.bs)
    # train_dataset = 'pope_adv'
    # train_model = 'llava16_moe'
    # train_loader = get_train_data_loaders(train_dataset, model_name=train_model, feature_key="response", batch_size=args.bs)

    val_loader = get_val_data_loaders(args.dataset, model_name=args.model, feature_key=args.feature_key,
                                          batch_size=args.bs)

    test_loader = get_test_data_loaders(args.dataset, model_name=args.model, feature_key=args.feature_key,
                                      batch_size=args.bs)

    # test_dataset = 'self_data'
    # test_model = 'llava16_7b'
    # test_loader = get_val_data_loaders(test_dataset, model_name=test_model, feature_key="response", batch_size=args.bs)


    input_dim = 4096  # Example dimension
    model = Probe(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # # Train the model
    num_epochs = 20
    train(model, criterion, optimizer, train_loader, test_loader, val_loader, num_epochs, device)


