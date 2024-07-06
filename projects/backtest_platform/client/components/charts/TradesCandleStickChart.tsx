import React, { CSSProperties, useEffect, useRef, useState } from "react";
import { OhlcvTick, Trade } from "../../clients/queries/response-types";
import { convertMillisToDateDict } from "../../utils/date";
import {
  CandlestickData,
  ChartOptions,
  Time,
  createChart,
} from "lightweight-charts";
import { generateChartMarkers } from "../../utils/lightweight-charts";
import { DeepPartial } from "@chakra-ui/react";

interface Props {
  trades: Trade[];
  ohlcvData: OhlcvTick[];
  styles?: CSSProperties;
  chartContainerStyles?: CSSProperties;
  isShortSellingTrades: boolean;
  hideTexts?: boolean;
  hideEnters?: boolean;
  hideExits?: boolean;
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
}: Props) => {
  const chartContainerRef = useRef();
  const [visibleRange, setVisibleRange] = useState(null);

  useEffect(() => {
    const chartOptions: DeepPartial<ChartOptions> = {
      layout: {
        textColor: "black",
        background: { type: "solid", color: "white" },
      },
    };

    if (!chartContainerRef.current || ohlcvData.length === 0) {
      return;
    }

    const chart = createChart(chartContainerRef.current, chartOptions);

    const series = chart.addCandlestickSeries({
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });

    const data: CandlestickData<Time>[] = [];

    ohlcvData.forEach((item) => {
      data.push({
        open: item.open_price,
        high: item.high_price,
        low: item.low_price,
        close: item.close_price,
        time: convertMillisToDateDict(item.kline_open_time),
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
    return () => chart.remove();
  }, [
    trades,
    ohlcvData,
    isShortSellingTrades,
    hideTexts,
    hideEnters,
    hideExits,
  ]);

  return (
    <div style={styles}>
      <div ref={chartContainerRef} style={chartContainerStyles} />
    </div>
  );
};
