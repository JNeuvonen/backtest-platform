TRAIN_TEMPLATE = """
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dataset import load_train_data
import pickle
from config import append_app_data_path
from orm import ModelWeightsQuery
from log import get_logger
from constants import DomEventChannels

{MODEL_CLASS}
{CRITERION_AND_OPTIMIZER}

def train():
    logger = get_logger()
    x_train, y_train = load_train_data({DATASET_NAME}, {TARGET_COL}, {NULL_FILL_STRATEGY})
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset, batch_size={BATCH_SIZE}, shuffle={SHUFFLE})
    model = Model(x_train.shape[1])
    criterion, optimizer = get_criterion_and_optimizer(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range({NUM_EPOCHS}):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            l1_lambda = 0.01
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = criterion(outputs, labels) + l1_lambda * l1_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        model_weights_dump = pickle.dumps(model.state_dict())
        ModelWeightsQuery.create_model_weights_entry({TRAIN_JOB_ID}, epoch, model_weights_dump)
        logger.log(
            f"Epoch [{epoch}/{NUM_EPOCHS}] complete, Loss: {loss.item():.4f}",
            logging.INFO,
            True,
            True,
            DomEventChannels.REFETCH_ALL_DATASETS.value,
        )

train()
"""
