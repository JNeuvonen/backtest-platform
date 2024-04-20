import React from "react";
import { CSSProperties } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface Props {
  data: object[];
  height: number;
  containerStyles?: CSSProperties;
  xAxisKey: string;
  children: React.ReactNode | React.ReactNode[];
  xAxisTickFormatter?: any;
}

export const ShareYAxisMultilineChart = ({
  height,
  containerStyles,
  data,
  xAxisKey,
  children,
  xAxisTickFormatter,
}: Props) => {
  return (
    <ResponsiveContainer width={"100%"} height={height} style={containerStyles}>
      <ComposedChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={xAxisKey} tickFormatter={xAxisTickFormatter} />
        <YAxis />
        <Tooltip
          formatter={(value: any, name: string) => {
            return name === "kline_open_time"
              ? new Date(value).toLocaleString()
              : value;
          }}
          labelFormatter={(label: any) => new Date(label).toLocaleString()}
        />

        <Legend />
        {children}
      </ComposedChart>
    </ResponsiveContainer>
  );
};
