import { CodeHelper } from "./constants";

export const ENTER_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def enter_trade(tick):");
  code.addIndent();
  code.appendLine("return true");
  return code.get();
};

export const EXIT_TRADE_DEFAULT = () => {
  const code = new CodeHelper();

  code.appendLine("def exit_trade(tick):");
  code.addIndent();
  code.appendLine("return true");
  return code.get();
};

export const CREATE_COLUMNS_DEFAULT = () => {
  const code = new CodeHelper();
  code.appendLine("dataset = get_dataset()");

  return code.get();
};
