import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

interface TwoLineChartProps {
  data: { [key: string]: any }[];
  xKey: string;
  line1Key: string;
  line2Key: string;
  width: number;
  height: number;
}

export const ShareYAxisTwoLineChart = ({
  data,
  xKey,
  line1Key,
  line2Key,
  width,
  height,
}: TwoLineChartProps) => (
  <LineChart width={width} height={height} data={data}>
    <CartesianGrid strokeDasharray="3 3" />
    <XAxis dataKey={xKey} />
    <YAxis />
    <Tooltip />
    <Legend />
    <Line type="monotone" dataKey={line1Key} stroke="#8884d8" />
    <Line type="monotone" dataKey={line2Key} stroke="#82ca9d" />
  </LineChart>
);
