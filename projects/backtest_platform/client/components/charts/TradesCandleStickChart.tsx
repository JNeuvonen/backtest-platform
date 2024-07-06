import React, { CSSProperties, useEffect, useRef, useState } from "react";
import { OhlcvTick, Trade } from "../../clients/queries/response-types";
import {
  CandlestickData,
  ChartOptions,
  LineData,
  Range,
  Time,
  createChart,
} from "lightweight-charts";
import { generateChartMarkers } from "../../utils/lightweight-charts";
import { DeepPartial } from "@chakra-ui/react";
import { COLOR_CONTENT_PRIMARY } from "../../utils/colors";

interface Props {
  trades: Trade[];
  ohlcvData: OhlcvTick[];
  styles?: CSSProperties;
  chartContainerStyles?: CSSProperties;
  isShortSellingTrades: boolean;
  hideTexts?: boolean;
  hideEnters?: boolean;
  hideExits?: boolean;
  getAltDataTicks?: () => LineData<Time>[];
  useAltData?: boolean;
  showBacktestResult?: boolean;
  reDrawChart?: number;
  getBacktestResultTicks?: () => LineData<Time>[];
  visibleRange: Range<Time> | null;
  setVisibleRange: React.SetStateAction<React.Dispatch<Range<Time> | null>>;
  redrawChart?: number;
}

export const TradesCandleStickChart = ({
  trades,
  ohlcvData,
  styles,
  chartContainerStyles = {
    width: "calc(100% + 32px)",
    height: "800px",
    marginLeft: "-16px",
    marginTop: "-16px",
  },
  isShortSellingTrades,
  hideTexts = true,
  hideEnters = false,
  hideExits = false,
  useAltData = false,
  getAltDataTicks = null,
  showBacktestResult = false,
  getBacktestResultTicks = null,
  visibleRange,
  setVisibleRange,
  redrawChart = 0,
}: Props) => {
  const chartContainerRef = useRef();

  useEffect(() => {
    const chartOptions: DeepPartial<ChartOptions> = {
      layout: {
        textColor: COLOR_CONTENT_PRIMARY,
        background: { type: "solid", color: "#161A25" },
      },
      grid: {
        vertLines: {
          color: "#232632",
        },
        horzLines: {
          color: "#232632",
        },
      },
    };

    if (!chartContainerRef.current || ohlcvData.length === 0) {
      return;
    }

    const chart = createChart(chartContainerRef.current, chartOptions);

    const series = chart.addCandlestickSeries({
      upColor: "rgb(8,153,129)",
      downColor: "rgb(242,54,69)",
      borderVisible: false,
      wickUpColor: "rgb(8,153,129)",
      wickDownColor: "rgb(242,54,69)",
      priceScaleId: "right",
    });

    const data: CandlestickData<Time>[] = [];

    const lineSeries = chart.addLineSeries({
      color: "#98a7d9",
      lineWidth: 1,
      priceScaleId: "left",
    });

    if (useAltData || showBacktestResult) {
      if (showBacktestResult) {
        lineSeries.setData(
          getBacktestResultTicks ? getBacktestResultTicks() : []
        );
      } else if (getAltDataTicks) {
        lineSeries.setData(getAltDataTicks ? getAltDataTicks() : []);
      }

      chart.applyOptions({
        rightPriceScale: {
          visible: true,
          borderColor: "rgba(197, 203, 206, 1)",
          textColor: COLOR_CONTENT_PRIMARY,
        },
        leftPriceScale: {
          visible: true,
          borderColor: "rgba(197, 203, 206, 1)",
          textColor: COLOR_CONTENT_PRIMARY,
        },
      });
    } else {
    }

    ohlcvData.forEach((item) => {
      data.push({
        open: item.open_price,
        high: item.high_price,
        low: item.low_price,
        close: item.close_price,
        time: item.kline_open_time / 1000,
      });
    });

    const handleVisibleRangeChange = () => {
      setVisibleRange(chart.timeScale().getVisibleRange());
    };

    chart.timeScale().subscribeVisibleTimeRangeChange(handleVisibleRangeChange);

    series.setData(data);
    series.setMarkers(
      generateChartMarkers(
        trades,
        isShortSellingTrades,
        hideTexts,
        hideEnters,
        hideExits
      )
    );

    if (visibleRange) {
      chart.timeScale().setVisibleRange(visibleRange);
    } else {
      chart.timeScale().fitContent();
    }
    return () => {
      chart
        .timeScale()
        .unsubscribeVisibleTimeRangeChange(handleVisibleRangeChange);
      chart.remove();
    };
  }, [
    trades,
    ohlcvData,
    isShortSellingTrades,
    hideTexts,
    hideEnters,
    hideExits,
    useAltData,
    showBacktestResult,
    redrawChart,
  ]);

  return (
    <div style={styles}>
      <div ref={chartContainerRef} style={chartContainerStyles} />
    </div>
  );
};
