import React, { CSSProperties, useEffect, useState } from "react";
import { EpochInfo } from "../../clients/queries/response-types";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface Props {
  kline_open_times: number[];
  _prices: number[];
  epoch: EpochInfo;
  containerStyles?: CSSProperties;
}

interface PredAndPriceChartTick {
  price: number;
  prediction: number;
  kline_open_time: string;
}

export const PredAndPriceChart = ({
  kline_open_times,
  _prices,
  epoch,
  containerStyles,
}: Props) => {
  const [klineOpenTimes] = useState<number[]>(kline_open_times);
  const [prices] = useState<number[]>(_prices);
  const [preds, setPreds] = useState<number[]>(
    epoch.val_predictions.map((item) => item.prediction)
  );

  useEffect(() => {
    setPreds(epoch.val_predictions.map((item) => item.prediction));
  }, [epoch.val_predictions]);

  const generateChartData = () => {
    const ret = [] as PredAndPriceChartTick[];
    const increment = Math.max(Math.ceil(klineOpenTimes.length / 200), 1);

    if (!prices) return [];
    for (let i = 0; i < klineOpenTimes.length; i += increment) {
      const item = {
        price: prices[i],
        prediction: preds[i],
        kline_open_time: new Date(klineOpenTimes[i]).toLocaleDateString(
          "default",
          {
            year: "numeric",
            month: "short",
          }
        ),
        price_in_24_ticks:
          kline_open_times.length > i + 24 ? prices[i + 24] : null,
        price_in_72_ticks:
          kline_open_times.length > i + 72 ? prices[i + 72] : null,
      };
      ret.push(item);
    }

    return ret;
  };

  const data = generateChartData();

  if (data.length === 0) return null;

  return (
    <div style={containerStyles}>
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={data}>
          <CartesianGrid />
          <XAxis dataKey="kline_open_time" />
          <YAxis yAxisId="price" />
          <YAxis
            yAxisId="prediction"
            orientation="right"
            label={{ value: "Prediction", angle: 90, position: "insideRight" }}
            domain={["auto", "auto"]}
          />
          <Tooltip
            contentStyle={{ color: "black" }}
            formatter={(value, name) => {
              if (
                name === "price_in_24_ticks" ||
                name === "price_in_72_ticks"
              ) {
                return [value !== null ? value : "N/A", name];
              }
              return [value, name];
            }}
            labelFormatter={(label) => `Time: ${label}`}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="price"
            stroke="#8884d8"
            yAxisId="price"
            dot={false}
          />

          <Line
            type="monotone"
            dataKey="prediction"
            stroke="#82ca9d"
            yAxisId="prediction"
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};
