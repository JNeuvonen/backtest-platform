import { SeriesMarker, Time } from "lightweight-charts";
import { Trade } from "../clients/queries/response-types";
import { convertMillisToDateDict } from "./date";
import {
  COLOR_BRAND_PRIMARY,
  COLOR_BRAND_PRIMARY_HIGHLIGHT,
  COLOR_BRAND_SECONDARY,
  COLOR_BRAND_SECONDARY_SHADE_ONE,
} from "./colors";

export interface ChartMarkerTick {
  time: object;
  position: string;
  color: string;
  shape: string;
  text: string;
}

export const generateChartMarkers = (
  trades: Trade[],
  isShortStrategy: boolean,
  hideTexts: boolean,
  hideEnters: boolean,
  hideExits: boolean
) => {
  const markers: SeriesMarker<Time>[] = [];

  const sortedTrades = trades.sort((a, b) => a.open_time - b.open_time);

  sortedTrades.forEach((item) => {
    const openDateObj = item.open_time / 1000;
    const closeDateObj = item.close_time / 1000;

    if (isShortStrategy) {
      const markerOpen = {
        time: openDateObj,
        position: "aboveBar",
        color: "#9de67c",
        shape: "arrowDown",
        text: hideTexts ? "" : `Sell @ ${item.open_price}`,
      };
      const markerClose = {
        time: closeDateObj,
        position: "belowBar",
        color: "#c93c43",
        shape: "arrowUp",
        text: hideTexts ? "" : `Buy @ ${item.close_price}`,
      };

      if (!hideEnters) {
        markers.push(markerOpen);
      }

      if (!hideExits) {
        markers.push(markerClose);
      }
    } else {
      const markerOpen = {
        time: openDateObj,
        position: "belowBar",
        color: "#9de67c",
        shape: "arrowUp",
        text: hideTexts ? "" : `Buy @ ${item.open_price}`,
      };
      const markerClose = {
        time: closeDateObj,
        position: "aboveBar",
        color: "#c93c43",
        shape: "arrowDown",
        text: hideTexts ? "" : `Sell @ ${item.close_price}`,
      };

      if (!hideEnters) {
        markers.push(markerOpen);
      }

      if (!hideExits) {
        markers.push(markerClose);
      }
    }
  });

  return markers;
};
