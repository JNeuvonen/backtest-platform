import React, { CSSProperties } from "react";
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Scatter,
  ComposedChart,
} from "recharts";
import { COLOR_BRAND_PRIMARY, COLOR_BRAND_SECONDARY } from "../../utils/colors";

interface TwoLineChartProps {
  data: { [key: string]: object }[];
  xKey: string;
  line1Key: string;
  line2Key: string;
  height: number;
  containerStyles?: CSSProperties;
  showDots?: boolean;
  displayTradeEntersAndExits?: boolean;
  xAxisTickFormatter?: any;
  yAxisTickFormatter?: any;
  tooltipFormatter?: any;
  tooltipLabelFormatter?: any;
}

export const ShareYAxisTwoLineChart = ({
  data,
  xKey,
  line1Key,
  line2Key,
  height,
  containerStyles,
  showDots = true,
  displayTradeEntersAndExits = false,
  yAxisTickFormatter,
  xAxisTickFormatter,
  tooltipFormatter,
  tooltipLabelFormatter,
}: TwoLineChartProps) => (
  <ResponsiveContainer width={"100%"} height={height} style={containerStyles}>
    <ComposedChart data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey={xKey} tickFormatter={xAxisTickFormatter} />
      <YAxis tickFormatter={yAxisTickFormatter} />
      <Tooltip
        formatter={tooltipFormatter}
        labelFormatter={tooltipLabelFormatter}
      />
      <Legend />
      <Line
        type="monotone"
        dataKey={line1Key}
        stroke={COLOR_BRAND_PRIMARY}
        dot={showDots}
      />
      <Line
        type="monotone"
        dataKey={line2Key}
        stroke={COLOR_BRAND_SECONDARY}
        dot={showDots}
      />

      {displayTradeEntersAndExits && (
        <>
          <Scatter dataKey="enter_trade" fill="green" />
          <Scatter dataKey="exit_trade" fill="red" />
        </>
      )}
    </ComposedChart>
  </ResponsiveContainer>
);
