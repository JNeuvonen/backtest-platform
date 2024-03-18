import React, { CSSProperties } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface TwoLineChartProps {
  data: { [key: string]: object }[];
  xKey: string;
  line1Key: string;
  line2Key: string;
  height: number;
  containerStyles?: CSSProperties;
  showDots?: boolean;
}

export const ShareYAxisTwoLineChart = ({
  data,
  xKey,
  line1Key,
  line2Key,
  height,
  containerStyles,
  showDots = true,
}: TwoLineChartProps) => (
  <ResponsiveContainer width={"100%"} height={height} style={containerStyles}>
    <LineChart data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey={xKey} />
      <YAxis />
      <Tooltip />
      <Legend />
      <Line type="monotone" dataKey={line1Key} stroke="green" dot={showDots} />
      <Line type="monotone" dataKey={line2Key} stroke="red" dot={showDots} />
    </LineChart>
  </ResponsiveContainer>
);
