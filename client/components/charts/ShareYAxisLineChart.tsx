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
}

export const ShareYAxisTwoLineChart = ({
  data,
  xKey,
  line1Key,
  line2Key,
  height,
}: TwoLineChartProps) => (
  <ResponsiveContainer width={"100%"} height={height}>
    <LineChart data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey={xKey} />
      <YAxis />
      <Tooltip />
      <Legend />
      <Line type="monotone" dataKey={line1Key} stroke="green" />
      <Line type="monotone" dataKey={line2Key} stroke="red" />
    </LineChart>
  </ResponsiveContainer>
);
