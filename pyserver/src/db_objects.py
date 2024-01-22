class ModelObject:
    def __init__(
        self,
        model_id,
        dataset_id,
        target_col,
        drop_cols,
        null_fill_strat,
        model,
        name,
        hyper_params_and_optimizer_code,
        validation_split,
    ):
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.target_col = target_col
        self.drop_cols = drop_cols.split(",") if drop_cols else []
        self.null_fill_strat = null_fill_strat
        self.model = model
        self.name = name
        self.hyper_params_and_optimizer_code = hyper_params_and_optimizer_code
        self.validation_split = (
            [float(val) for val in validation_split.split(",")]
            if validation_split
            else []
        )

    @classmethod
    def from_db_row(cls, row):
        return cls(*row)

    def __repr__(self):
        return f"<ModelObject {self.model_id}: {self.name}>"


class TrainJobObject:
    def __init__(
        self,
        job_id,
        name,
        model_name,
        num_epochs,
        save_model_every_epoch,
        backtest_on_validation_set,
        enter_trade_criteria,
        exit_trade_criteria,
    ):
        self.name = name
        self.job_id = job_id
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.save_model_every_epoch = save_model_every_epoch
        self.backtest_on_validation_set = backtest_on_validation_set
        self.enter_trade_criteria = enter_trade_criteria
        self.exit_trade_criteria = exit_trade_criteria

    @classmethod
    def from_db_row(cls, row):
        return cls(*row)

    def __repr__(self):
        return f"<TrainJobObject {self.job_id}: {self.model_name}>"


class DatasetObject:
    def __init__(self, dataset_id, dataset_name, timeseries_column):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.timeseries_column = timeseries_column

    @classmethod
    def from_db_row(cls, row):
        return cls(*row)

    def __repr__(self):
        return f"<DatasetObject {self.dataset_id}: {self.dataset_name}>"
