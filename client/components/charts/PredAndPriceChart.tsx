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
  kline_open_times: string;
  _prices: string;
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
  const [klineOpenTimes] = useState<number[]>(JSON.parse(kline_open_times));
  const [prices] = useState<number[]>(JSON.parse(_prices));
  const [preds, setPreds] = useState<number[]>(
    JSON.parse(epoch.val_predictions)
  );

  useEffect(() => {
    setPreds(JSON.parse(epoch.val_predictions));
  }, [epoch.val_predictions]);

  const generateChartData = () => {
    const ret = [] as PredAndPriceChartTick[];
    const increment = Math.max(Math.ceil(klineOpenTimes.length / 200), 1);
    for (let i = 0; i < klineOpenTimes.length; i += increment) {
      const item = {
        price: prices[i],
        prediction: preds[i][0],
        kline_open_time: new Date(klineOpenTimes[i]).toLocaleString("en-US", {
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        }),
        price_in_24_ticks:
          kline_open_times.length > i + 24 ? prices[i + 24] : null,
        price_in_72_ticks:
          kline_open_times.length > i + 72 ? prices[i + 72] : null,
      };
      ret.push(item);
    }

    return ret;
  };

  return (
    <div style={containerStyles}>
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={generateChartData()}>
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
