import { Button, Text } from "@chakra-ui/react";
import { ICellRendererParams } from "ag-grid-community";
import { Link } from "react-router-dom";
import { toast } from "react-toastify";
import {
  getNumberDisplayColor,
  inferAssets,
  roundNumberFloor,
} from "src/common_js";
import { repayMarginLoanRequest } from "src/http";
import { BUTTON_VARIANTS, COLOR_CONTENT_PRIMARY } from "src/theme";
import {
  getStrategyPath,
  getTradeViewPath,
  getViewBinanceChartUri,
} from "src/utils";

export const StrategyNameCellRenderer = (params: ICellRendererParams) => {
  if (!params.value) {
    return;
  }
  return (
    <Link
      to={getStrategyPath(params.value.toLowerCase())}
      className="link-default"
    >
      {params.value}
    </Link>
  );
};

export const SymbolCellRenderer = (params: ICellRendererParams) => {
  if (!params.value) {
    return;
  }
  const { baseAsset, quoteAsset } = inferAssets(params.value);
  return (
    <Link
      to={getViewBinanceChartUri(baseAsset, quoteAsset)}
      className="link-default"
      target={"_blank"}
    >
      {params.value}
    </Link>
  );
};

export const PercColumnCellRenderer = (params: ICellRendererParams) => {
  if (!params.value) {
    return null;
  }

  return (
    <Text color={getNumberDisplayColor(params.value, COLOR_CONTENT_PRIMARY)}>
      {roundNumberFloor(params.value, 2)}%
    </Text>
  );
};

export const ProfitColumnCellRenderer = (params: ICellRendererParams) => {
  if (!params.value) {
    return null;
  }

  return (
    <Text color={getNumberDisplayColor(params.value, COLOR_CONTENT_PRIMARY)}>
      {roundNumberFloor(params.value, 2)}$
    </Text>
  );
};

export const PayOffLoanCellRenderer = (params: ICellRendererParams) => {
  if (params.value === 0 || params.data.openTrades > 0) {
    return null;
  }

  const postRequest = async () => {
    const res = await repayMarginLoanRequest(params.data.asset);
    if (res.success) {
      toast.success(`Repaid margin loan on: ${params.data.asset}`, {
        theme: "dark",
      });
    } else {
      toast.error("Failed to repay margin loan", { theme: "dark" });
    }
  };

  return (
    <Button variant={BUTTON_VARIANTS.nofill} onClick={postRequest}>
      Repay loan
    </Button>
  );
};

export const TradeIdCellRenderer = (params: ICellRendererParams) => {
  if (!params.value) {
    return null;
  }

  return (
    <Link to={getTradeViewPath(params.value)} className={"link-default"}>
      {params.value}
    </Link>
  );
};
