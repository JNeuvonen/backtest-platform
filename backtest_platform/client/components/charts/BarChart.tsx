import React, { CSSProperties } from "react";
import {
  Bar,
  BarChart,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface Props {
  data: object[];
  xAxisKey: string;
  yAxisKey: string;
  containerStyles?: CSSProperties;
}

export const GenericBarChart = ({
  data,
  xAxisKey,
  yAxisKey,
  containerStyles,
}: Props) => {
  return (
    <div style={containerStyles}>
      <ResponsiveContainer width={"100%"} height={400}>
        <BarChart data={data}>
          <XAxis dataKey={xAxisKey} />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey={yAxisKey} fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
