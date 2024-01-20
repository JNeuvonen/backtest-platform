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
