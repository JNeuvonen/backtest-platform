import { useBalanceSnapshotsQuery } from "src/http/queries";
import {
  Heading,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
  Text,
} from "@chakra-ui/react";
import { Line } from "recharts";
import { ShareYAxisMultilineChart } from "src/components";
import {
  COLOR_BRAND_PRIMARY,
  COLOR_BRAND_PRIMARY_SHADE_FOUR,
  COLOR_BRAND_SECONDARY,
} from "src/theme";
import { BalanceSnapshot, getDiffToPresentFormatted } from "common_js";
import BalanceInfoCard from "src/components/BalanceInfoCard";

export const RootPage = () => {
  const balanceSnapShots = useBalanceSnapshotsQuery();

  const getTickClosestToOneDayAgo = () => {
    if (!balanceSnapShots.data || balanceSnapShots.data.length === 0)
      return null;

    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);

    let closestTick = balanceSnapShots.data[0];
    let closestDiff = Math.abs(
      new Date(balanceSnapShots.data[0].created_at).getTime() -
        oneDayAgo.getTime(),
    );

    for (let i = 1; i < balanceSnapShots.data.length; i++) {
      const currentDiff = Math.abs(
        new Date(balanceSnapShots.data[i].created_at).getTime() -
          oneDayAgo.getTime(),
      );
      if (currentDiff < closestDiff) {
        closestTick = balanceSnapShots.data[i];
        closestDiff = currentDiff;
      }
    }

    return closestTick;
  };

  const getTickClosestToWeekAgo = () => {
    if (!balanceSnapShots.data || balanceSnapShots.data.length === 0)
      return null;

    const oneDayAgo = new Date(Date.now() - 24 * 7 * 60 * 60 * 1000);

    let closestTick = balanceSnapShots.data[0];
    let closestDiff = Math.abs(
      new Date(balanceSnapShots.data[0].created_at).getTime() -
        oneDayAgo.getTime(),
    );

    for (let i = 1; i < balanceSnapShots.data.length; i++) {
      const currentDiff = Math.abs(
        new Date(balanceSnapShots.data[i].created_at).getTime() -
          oneDayAgo.getTime(),
      );
      if (currentDiff < closestDiff) {
        closestTick = balanceSnapShots.data[i];
        closestDiff = currentDiff;
      }
    }

    return closestTick;
  };

  const getFirstTickOfCurrentMonth = () => {
    if (!balanceSnapShots.data || balanceSnapShots.data.length === 0)
      return null;

    const currentDate = new Date();
    const firstDayOfMonth = new Date(
      currentDate.getFullYear(),
      currentDate.getMonth(),
      1,
    );

    let firstTickOfMonth = null;

    for (let i = balanceSnapShots.data.length - 1; i >= 0; i--) {
      const tickDate = new Date(balanceSnapShots.data[i].created_at);
      if (tickDate >= firstDayOfMonth) {
        firstTickOfMonth = balanceSnapShots.data[i];
      } else {
        break;
      }
    }

    return firstTickOfMonth;
  };

  const getFirstTickOfCurrentYear = () => {
    if (!balanceSnapShots.data || balanceSnapShots.data.length === 0)
      return null;

    const currentDate = new Date();
    const firstDayOfYear = new Date(currentDate.getFullYear(), 0, 1);

    let firstTickOfYear = null;

    for (let i = balanceSnapShots.data.length - 1; i >= 0; i--) {
      const tickDate = new Date(balanceSnapShots.data[i].created_at);
      if (tickDate >= firstDayOfYear) {
        firstTickOfYear = balanceSnapShots.data[i];
      } else {
        break;
      }
    }

    return firstTickOfYear;
  };

  if (!balanceSnapShots.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }

  const lastTick = balanceSnapShots.data[balanceSnapShots.data.length - 1];
  const oneDayAgoTick = getTickClosestToOneDayAgo();
  const oneWeekAgoTick = getTickClosestToWeekAgo();
  const monthFirstTick = getFirstTickOfCurrentMonth();
  const yearsFirstTick = getFirstTickOfCurrentYear();

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
            Last account snapshot:{" "}
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
      <div style={{ marginTop: "16px" }}>
        <BalanceInfoCard
          heading={"24h changes"}
          lastTick={lastTick}
          comparisonTick={oneDayAgoTick as BalanceSnapshot}
        />
      </div>
      <div style={{ marginTop: "16px" }}>
        <BalanceInfoCard
          heading={"One week changes"}
          lastTick={lastTick}
          comparisonTick={oneWeekAgoTick as BalanceSnapshot}
        />
      </div>
      <div style={{ marginTop: "16px" }}></div>
      <div
        style={{
          marginTop: "16px",
          display: "flex",
          alignItems: "center",
          gap: "16px",
        }}
      >
        <BalanceInfoCard
          heading={"MTD"}
          lastTick={lastTick}
          comparisonTick={monthFirstTick as BalanceSnapshot}
          showOnlyNav={true}
        />
        <BalanceInfoCard
          heading={"YTD"}
          lastTick={lastTick}
          comparisonTick={monthFirstTick as BalanceSnapshot}
          showOnlyNav={true}
        />
      </div>
    </div>
  );
};
