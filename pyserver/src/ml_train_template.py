TRAIN_TEMPLATE = """
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dataset import load_train_data
import pickle
from config import append_app_data_path
from orm import ModelWeightsQuery, TrainJobQuery
from log import get_logger
from constants import DomEventChannels

{MODEL_CLASS}
{CRITERION_AND_OPTIMIZER}

def train():
    x_train, y_train, x_val, y_val = load_train_data({DATASET_NAME}, {TARGET_COL}, {NULL_FILL_STRATEGY}, {TRAIN_VAL_SPLIT})
    dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset, batch_size={BATCH_SIZE}, shuffle={SHUFFLE})
    model = Model(x_train.shape[1])
    criterion, optimizer = get_criterion_and_optimizer(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    canceled_by_user_request = False
    save_every_epoch = {SAVE_MODEL_EVERY_EPOCH}
    logger = get_logger()
    train_job_state = TrainJobQuery.get_train_job({TRAIN_JOB_ID})

    
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



        logger.log(
            f"Epoch [{epoch}/{NUM_EPOCHS}] complete, Loss: {loss.item():.4f}",
            logging.INFO,
            True,
            True,
            DomEventChannels.REFETCH_ALL_DATASETS.value,
        )


        if train_job_state.backtest_on_validation_set is True:
            pass

        if save_every_epoch is True:
            model_weights_dump = pickle.dumps(model.state_dict())
            ModelWeightsQuery.create_model_weights_entry({TRAIN_JOB_ID}, epoch, model_weights_dump)



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


    if canceled_by_user_request is False:
        TrainJobQuery.set_training_status({TRAIN_JOB_ID}, False)
train()
"""
