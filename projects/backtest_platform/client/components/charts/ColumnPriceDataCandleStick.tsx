import React, { CSSProperties, useEffect, useRef } from "react";
import { OhlcvTick } from "../../clients/queries/response-types";
import {
  CandlestickData,
  ChartOptions,
  DeepPartial,
  IChartApi,
  Range,
  Time,
  createChart,
} from "lightweight-charts";
import {
  CANDLE_STICK_CHART_DEFAULTS,
  CANDLE_STICK_CHART_OPTIONS,
  CANDLE_STICK_PRICE_SCALE_DEFAULTS,
} from "../../utils/constants";

interface Props {
  ohlcvData: OhlcvTick[];
  styles?: CSSProperties;
  chartContainerStyles?: CSSProperties;
  setColumnDataTicks: (chartAPI: IChartApi) => void;
  visibleRange: Range<Time> | null;
  setVisibleRange: React.SetStateAction<React.Dispatch<Range<Time> | null>>;
}

export const ColumnPriceDataCandleStickChart = ({
  ohlcvData,
  styles,

  chartContainerStyles = {
    width: "calc(100% + 64px)",
    height: "800px",
    marginLeft: "-32px",
    marginTop: "-16px",
  },
  setColumnDataTicks,
  visibleRange,
  setVisibleRange,
}: Props) => {
  const chartContainerRef = useRef();

  useEffect(() => {
    const chartOptions: DeepPartial<ChartOptions> = CANDLE_STICK_CHART_OPTIONS;

    if (!chartContainerRef.current || ohlcvData.length === 0) {
      return;
    }

    const chart = createChart(chartContainerRef.current, chartOptions);

    const series = chart.addCandlestickSeries(CANDLE_STICK_CHART_DEFAULTS);

    chart.applyOptions({
      rightPriceScale: CANDLE_STICK_PRICE_SCALE_DEFAULTS,
      leftPriceScale: CANDLE_STICK_PRICE_SCALE_DEFAULTS,
    });

    const data: CandlestickData<Time>[] = [];

    ohlcvData.forEach((item) => {
      data.push({
        open: item.open_price,
        high: item.high_price,
        low: item.low_price,
        close: item.close_price,
        time: (item.kline_open_time / 1000) as Time,
      });
    });

    const handleVisibleRangeChange = () => {
      setVisibleRange(chart.timeScale().getVisibleRange());
    };

    chart.timeScale().subscribeVisibleTimeRangeChange(handleVisibleRangeChange);

    setColumnDataTicks(chart);
    series.setData(data);

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
  }, [ohlcvData]);
  return (
    <div style={styles}>
      <div ref={chartContainerRef} style={chartContainerStyles} />
    </div>
  );
};
