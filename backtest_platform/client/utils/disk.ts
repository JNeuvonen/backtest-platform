export const DISK_KEYS = {
  backtest_form: "BACKTEST_FORM",
  deploy_strategy_form: "DEPLOY_STRATEGY_FORM",
  app_settings: "APP_SETTINGS",
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
