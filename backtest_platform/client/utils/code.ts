import { CodeHelper } from "./constants";

export const ENTER_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def open_long_trade(tick):");
  code.addIndent();
  code.appendLine("return True");
  return code.get();
};

export const EXIT_LONG_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def close_long_trade(tick):");
  code.addIndent();
  code.appendLine("return True");
  return code.get();
};

export const EXIT_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def open_short_trade(tick):");
  code.addIndent();
  code.appendLine("return True");
  return code.get();
};

export const EXIT_SHORT_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def close_short_trade(tick):");
  code.addIndent();
  code.appendLine("return True");
  return code.get();
};

export const CREATE_COLUMNS_DEFAULT = () => {
  const code = new CodeHelper();
  code.appendLine("dataset = get_dataset()");

  return code.get();
};

export const ML_ENTER_TRADE_COND = () => {
  const code = new CodeHelper();

  code.appendLine("def get_enter_trade_criteria(prediction):");
  code.addIndent();
  code.appendLine("return prediction > 1.01");
  return code.get();
};

export const ML_EXIT_TRADE_COND = () => {
  const code = new CodeHelper();

  code.appendLine("def get_exit_trade_criteria(prediction):");
  code.addIndent();
  code.appendLine("return prediction < 0.99");
  return code.get();
};
