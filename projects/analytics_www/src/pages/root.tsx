import { useBalanceSnapshotsQuery } from "src/http/queries";
import { Heading, Spinner, Text } from "@chakra-ui/react";
import { Line } from "recharts";
import { ShareYAxisMultilineChart } from "src/components";
import {
  COLOR_BRAND_PRIMARY,
  COLOR_BRAND_PRIMARY_SHADE_FOUR,
  COLOR_BRAND_SECONDARY,
} from "src/theme";
import { getDiffToPresentFormatted } from "common_js";

export const RootPage = () => {
  const balanceSnapShots = useBalanceSnapshotsQuery();

  if (!balanceSnapShots.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const lastTick = balanceSnapShots.data[balanceSnapShots.data.length - 1];

  return (
    <div>
      <div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Heading size={"lg"}>Equity</Heading>
          <Text fontSize={"13px"}>
            Last updated:{" "}
            {getDiffToPresentFormatted(new Date(lastTick.created_at))} ago
          </Text>
        </div>
        <ShareYAxisMultilineChart
          containerStyles={{ marginTop: "16px" }}
          height={500}
          data={balanceSnapShots.data}
          xAxisKey={"created_at"}
          yAxisTickFormatter={(value: number) => `${value}$`}
          xAxisTickFormatter={(tick: number) =>
            new Date(tick).toLocaleDateString("default", {
              year: "numeric",
              month: "short",
              day: "numeric",
              hour: "numeric",
            })
          }
        >
          <Line
            type="monotone"
            dataKey={"debt"}
            stroke={COLOR_BRAND_PRIMARY_SHADE_FOUR}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey={"value"}
            stroke={COLOR_BRAND_PRIMARY}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey={"long_assets_value"}
            stroke={COLOR_BRAND_SECONDARY}
            dot={false}
          />
        </ShareYAxisMultilineChart>
      </div>
    </div>
  );
};
