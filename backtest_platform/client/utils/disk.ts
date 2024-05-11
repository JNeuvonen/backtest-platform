export const DISK_KEYS = {
  backtest_form: "BACKTEST_FORM",
  long_short_form: "LONG_SHORT_FORM",
  deploy_strategy_form: "DEPLOY_STRATEGY_FORM",
  app_settings: "APP_SETTINGS",
  mass_long_short_form: "MASS_LONG_SHORT_FORM",
  ml_backtest_form: "ML_BASED_BACKTEST_FORM",
  run_python_on_dataset: "RUN_PYTHON_ON_DATASET",
  manage_code_presets_filters: "MANAGE_CODE_PRESETS_FILTERS",
};

export class DiskManager {
  key: string;
  constructor(key: string) {
    this.key = key;
  }

  save(values: object) {
    localStorage.setItem(this.key, JSON.stringify(values));
  }

  reset() {
    localStorage.removeItem(this.key);
  }

  read() {
    const savedData = localStorage.getItem(this.key);
    return savedData ? JSON.parse(savedData) : null;
  }
}
