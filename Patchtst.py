from transformers import PatchTSTConfig

import os

# Third Party
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
)
import numpy as np
import pandas as pd

# First Party
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index


from transformers import set_seed

set_seed(69)

if __name__ == '__main__':
    dataset_path = "data/processed/SPY.csv"
    timestamp_column = "datetime" 
    id_columns = []

    context_length = 512
    forecast_horizon = 96
    patch_length = 16
    num_workers = 14  # Reduce this if you have low number of CPU cores
    batch_size = 64  # Adjust according to GPU memory

    data = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_column],
    )
    forecast_columns = list(data.columns[1:])

    # get split
    num_train = int(len(data) * 0.7)
    num_test = int(len(data) * 0.2)
    num_valid = len(data) - num_train - num_test
    border1s = [
        0,
        num_train - context_length,
        len(data) - num_test - context_length,
    ]
    border2s = [num_train, num_train + num_valid, len(data)]

    train_start_index = border1s[0]  # None indicates beginning of dataset
    train_end_index = border2s[0]

    # we shift the start of the evaluation period back by context length so that
    # the first evaluation timestamp is immediately following the training data
    valid_start_index = border1s[1]
    valid_end_index = border2s[1]

    test_start_index = border1s[2]
    test_end_index = border2s[2]

    train_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=train_start_index,
        end_index=train_end_index,
    )
    valid_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=valid_start_index,
        end_index=valid_end_index,
    )
    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )

    time_series_preprocessor = TimeSeriesPreprocessor(
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        input_columns=forecast_columns,
        output_columns=forecast_columns,
        scaling=True,
    )
    time_series_preprocessor = time_series_preprocessor.train(train_data)

    # Preprocess data for train, valid, and test and add 'future_values' if missing
    processed_train = time_series_preprocessor.preprocess(train_data)
    if 'future_values' not in processed_train.columns:
        processed_train['future_values'] = processed_train[forecast_columns].values.tolist()

    processed_valid = time_series_preprocessor.preprocess(valid_data)
    if 'future_values' not in processed_valid.columns:
        processed_valid['future_values'] = processed_valid[forecast_columns].values.tolist()

    processed_test = time_series_preprocessor.preprocess(test_data)
    if 'future_values' not in processed_test.columns:
        processed_test['future_values'] = processed_test[forecast_columns].values.tolist()

    train_dataset = ForecastDFDataset(
        processed_train,
        id_columns=id_columns,
        timestamp_column="datetime",
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    valid_dataset = ForecastDFDataset(
        processed_valid,
        id_columns=id_columns,
        timestamp_column="datetime",
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    test_dataset = ForecastDFDataset(
        processed_test,
        id_columns=id_columns,
        timestamp_column="datetime",
        context_length=context_length,
        prediction_length=forecast_horizon,
    )

    config = PatchTSTConfig(
        num_input_channels=len(forecast_columns),
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_length,
        prediction_length=forecast_horizon,
        random_mask_ratio=0.4,
        d_model=128,
        num_attention_heads=16,
        num_hidden_layers=3,
        ffn_dim=256,
        dropout=0.2,
        head_dropout=0.2,
        pooling_type=None,
        channel_attention=False,
        scaling="std",
        loss="mse",
        pre_norm=True,
        norm_type="batchnorm",
    )
    model = PatchTSTForPrediction(config)

    # Move model to MPS device if available
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model.to(device)
    
    training_args = TrainingArguments(
        output_dir="./checkpoint/patchtst/spy/pretrain/output/",
        overwrite_output_dir=True,
        # learning_rate=0.001,
        num_train_epochs=100,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        logging_dir="./checkpoint/patchtst/spy/pretrain/logs/",  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        label_names=["future_values"],
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0001,  # Minimum improvement required to consider as improvement
    )

    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback],
        #compute_metrics=compute_metrics,
    )

    # pretrain
    trainer.train()

    results = trainer.evaluate(test_dataset)
    print("Test result:")
    print(results)

    save_dir = "patchtst/spy/model/pretrain/"
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)



