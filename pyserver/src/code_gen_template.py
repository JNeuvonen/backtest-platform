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
from query_weights import ModelWeightsQuery 
from query_trainjob import TrainJobQuery
from log import get_logger
from constants import DomEventChannels, Signals
from datetime import timedelta

{MODEL_CLASS}
{CRITERION_AND_OPTIMIZER}

def train():
    x_train, y_train, x_val, y_val, val_kline_open_times, val_prices = load_data({DATASET_NAME}, {MODEL_ID}, {TARGET_COL}, {NULL_FILL_STRATEGY}, {TRAIN_VAL_SPLIT}, {SCALING_STRATEGY}, {SCALE_TARGET})
    TrainJobQuery.set_backtest_data({TRAIN_JOB_ID}, val_prices, val_kline_open_times.values.tolist())
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size={BATCH_SIZE}, shuffle={SHUFFLE})
    val_loader = DataLoader(val_dataset, batch_size={BATCH_SIZE})

    model = Model(x_train.shape[1])
    criterion, optimizer = get_criterion_and_optimizer(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    TrainJobQuery.add_device({TRAIN_JOB_ID}, 'cuda' if torch.cuda.is_available() else 'cpu')

    canceled_by_user_request = False
    save_every_epoch = {SAVE_MODEL_EVERY_EPOCH}
    logger = get_logger()
    train_job_state = TrainJobQuery.get_train_job({TRAIN_JOB_ID})

    logger.log(
        Signals.OPEN_TRAINING_TOOLBAR,
        logging.DEBUG,
    )

    timer = time.time() - 0
    
    for epoch in range(1, {NUM_EPOCHS} + 1):
        model.train()
        total_train_loss = 0
        train_predictions = []
        epoch_start_time = time.time()
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
            train_predictions.extend(outputs.cpu().detach().numpy().tolist())


        model.eval()
        total_val_loss = 0
        validation_predictions = []
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                validation_predictions.extend(val_outputs.cpu().numpy().tolist())
                l1_lambda = 0.01
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                total_val_loss += criterion(val_outputs, val_labels).item() + l1_lambda * l1_norm

        train_loss_mean = total_train_loss / len(train_loader)
        val_loss_mean = total_val_loss / len(val_loader)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_duration_str = str(timedelta(seconds=int(epoch_duration)))

        if timer < epoch_end_time - 1:
            timer = time.time()
            logger.log(
                f"Finished training by user request",
                logging.INFO,
                False,
                False,
                DomEventChannels.REFETCH_COMPONENT.value,
            )
            logger.log(
                Signals.OPEN_TRAINING_TOOLBAR,
                logging.DEBUG,
            )


        logger.log(
            Signals.EPOCH_COMPLETE.format(
                EPOCHS_RAN=epoch, 
                MAX_EPOCHS={NUM_EPOCHS}, 
                TRAIN_LOSS=train_loss_mean, 
                VAL_LOSS=val_loss_mean, 
                EPOCH_TIME=epoch_duration_str,
                TRAIN_JOB_ID={TRAIN_JOB_ID},
            ),
            logging.DEBUG,
        )

        if save_every_epoch is True:
            model_weights_dump = pickle.dumps(model.state_dict())
            ModelWeightsQuery.create_model_weights_entry({TRAIN_JOB_ID}, epoch, model_weights_dump, train_loss_mean, val_loss_mean, validation_predictions, train_predictions)

        #check for cancel
        is_train_job_active = TrainJobQuery.is_job_training({TRAIN_JOB_ID})

        if is_train_job_active is False:
            logger.log(
                f"Finished training by user request",
                logging.INFO,
                True,
                True,
                DomEventChannels.REFETCH_COMPONENT.value,
            )
            canceled_by_user_request = True
            break

        TrainJobQuery.set_curr_epoch({TRAIN_JOB_ID}, epoch)


    if canceled_by_user_request is False:
        logger.log(
            f"Finished training. Epochs completed: {NUM_EPOCHS}/{NUM_EPOCHS}",
            logging.INFO,
            True,
            True,
            DomEventChannels.REFETCH_COMPONENT.value,
        )
        TrainJobQuery.set_training_status({TRAIN_JOB_ID}, False)


    logger.log(
        Signals.CLOSE_TOOLBAR,
        logging.DEBUG,
    )
train()
"""


BACKTEST_MODEL_TEMPLATE = """
{ENTER_AND_EXIT_CRITERIA_FUNCS}

enter_trade = get_enter_trade_criteria({PREDICTION}) 
exit_trade = get_exit_trade_criteria({PREDICTION}) 
"""

BACKTEST_MANUAL_TEMPLATE = """
{OPEN_LONG_TRADE_FUNC}
{OPEN_SHORT_TRADE_FUNC}
{CLOSE_LONG_TRADE_FUNC}
{CLOSE_SHORT_TRADE_FUNC}

should_open_long = open_long_trade(df_row) 
should_open_short = open_short_trade(df_row) 
should_close_long = close_long_trade(df_row)
should_close_short = close_short_trade(df_row)
"""
