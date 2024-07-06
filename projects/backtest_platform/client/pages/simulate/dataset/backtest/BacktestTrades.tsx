import React, { useEffect, useRef } from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import {
  useBacktestById,
  useDatasetOhlcvCols,
} from "../../../../clients/queries/queries";
import { CandlestickData, Time, createChart } from "lightweight-charts";
import { LightWeightChartOhlcvTick } from "../../../../clients/queries/response-types";
import { convertMillisToDateDict } from "../../../../utils/date";
import { generateChartMarkers } from "../../../../utils/lightweight-charts";

interface PathParams {
  datasetName: string;
  backtestId: number;
}

export const BacktestTradesPage = () => {
  const { backtestId, datasetName } = usePathParams<PathParams>();
  const backtestQuery = useBacktestById(Number(backtestId));
  const chartContainerRef = useRef();
  const ohlcvColsData = useDatasetOhlcvCols(datasetName);

  useEffect(() => {
    const chartOptions = {
      layout: {
        textColor: "black",
        background: { type: "solid", color: "white" },
      },
    };

    if (
      !chartContainerRef.current ||
      !ohlcvColsData.data ||
      !backtestQuery.data
    ) {
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

    ohlcvColsData.data.forEach((item) => {
      data.push({
        open: item.open_price,
        high: item.high_price,
        low: item.low_price,
        close: item.close_price,
        time: convertMillisToDateDict(item.kline_open_time),
      });
    });

    series.setData(data);

    series.setMarkers(
      generateChartMarkers(
        backtestQuery.data?.trades,
        backtestQuery.data.data.is_short_selling_strategy
      )
    );

    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [ohlcvColsData.data]);

  return (
    <div>
      <div
        ref={chartContainerRef}
        style={{
          width: "calc(100% + 32px)",
          height: "800px",
          marginLeft: "-16px",
          marginTop: "-16px",
        }}
      />
    </div>
  );
};
