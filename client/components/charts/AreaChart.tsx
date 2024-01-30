import React, { CSSProperties } from "react";
import {
  Area,
  AreaChart,
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

export const GenericAreaChart = ({
  data,
  xAxisKey,
  yAxisKey,
  containerStyles,
}: Props) => {
  return (
    <div style={containerStyles}>
      <ResponsiveContainer width={"100%"} height={400}>
        <AreaChart data={data}>
          <XAxis dataKey={xAxisKey} />
          <YAxis />
          <Tooltip />
          <Legend />
          <Area
            type="monotone"
            dataKey={yAxisKey}
            stroke="#8884d8"
            fill="#8884d8"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
