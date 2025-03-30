# give me code to evaluate the yolo11 model on data
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def evaluate_model(model, data_loader, device):
    
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total_samples
            print(f"Accuracy: {accuracy:.4f}")


