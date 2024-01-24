TRAIN_TEMPLATE = """
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import TensorDataset, DataLoader
from dataset import load_data
import pickle
from config import append_app_data_path
from orm import ModelWeightsQuery, TrainJobQuery
from log import get_logger
from constants import DomEventChannels

{MODEL_CLASS}
{CRITERION_AND_OPTIMIZER}

def train():

    x_train, y_train, x_val, y_val = load_data({DATASET_NAME}, {TARGET_COL}, {NULL_FILL_STRATEGY}, {TRAIN_VAL_SPLIT})

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size={BATCH_SIZE}, shuffle={SHUFFLE})
    val_loader = DataLoader(val_dataset, batch_size={BATCH_SIZE}, shuffle=False)

    model = Model(x_train.shape[1])
    criterion, optimizer = get_criterion_and_optimizer(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    canceled_by_user_request = False
    save_every_epoch = {SAVE_MODEL_EVERY_EPOCH}
    logger = get_logger()
    train_job_state = TrainJobQuery.get_train_job({TRAIN_JOB_ID})
    last_log_time = time.time() - 5

    
    for epoch in range({NUM_EPOCHS}):
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            l1_lambda = 0.01
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = criterion(outputs, labels) + l1_lambda * l1_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()


        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                total_val_loss += criterion(val_outputs, val_labels).item()

        train_loss_mean = total_train_loss / len(train_loader)
        val_loss_mean = total_val_loss / len(val_loader)
        current_time = time.time()

        if current_time - last_log_time >= 5:
            logger.log(
                f"Epoch [{epoch}/{NUM_EPOCHS}] complete, Train Loss: {train_loss_mean:.4f}, Val Loss: {val_loss_mean:.4f}",
                logging.INFO,
                True,
                True,
                DomEventChannels.REFETCH_ALL_DATASETS.value,
            )
            last_log_time = time.time()
        else:
            #not displayed to the UI
            logger.log(
                f"Epoch [{epoch}/{NUM_EPOCHS}] complete, Train Loss: {train_loss_mean:.4f}, Val Loss: {val_loss_mean:.4f}",
                logging.INFO,
            )


        if save_every_epoch is True:
            model_weights_dump = pickle.dumps(model.state_dict())
            ModelWeightsQuery.create_model_weights_entry({TRAIN_JOB_ID}, epoch, model_weights_dump, train_loss_mean, val_loss_mean)

        #check for cancel
        is_train_job_active = TrainJobQuery.is_job_training({TRAIN_JOB_ID})

        if is_train_job_active is False:
            logger.log(
                f"Finished training by user request",
                logging.INFO,
                True,
                True,
                DomEventChannels.REFETCH_ALL_DATASETS.value,
            )
            canceled_by_user_request = True
            break

        TrainJobQuery.set_curr_epoch({TRAIN_JOB_ID}, epoch)


    if canceled_by_user_request is False:
        TrainJobQuery.set_training_status({TRAIN_JOB_ID}, False)
train()
"""
