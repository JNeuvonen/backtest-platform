import { SeriesMarker, Time } from "lightweight-charts";
import { Trade } from "../clients/queries/response-types";
import { convertMillisToDateDict } from "./date";

export interface ChartMarkerTick {
  time: object;
  position: string;
  color: string;
  shape: string;
  text: string;
}

export const generateChartMarkers = (
  trades: Trade[],
  isShortStrategy: boolean
) => {
  const markers: SeriesMarker<Time>[] = [];

  trades.forEach((item) => {
    const openDateObj = convertMillisToDateDict(item.open_time);
    const closeDateObj = convertMillisToDateDict(item.close_time);

    if (isShortStrategy) {
      const markerOpen = {
        time: openDateObj,
        position: "aboveBar",
        color: "#e91e63",
        shape: "arrowDown",
        text: `Sell @ ${item.open_price}`,
      };
      const markerClose = {
        time: closeDateObj,
        position: "belowBar",
        color: "#2196F3",
        shape: "arrowUp",
        text: `Buy @ ${item.close_price}`,
      };
      markers.push(markerOpen);
      markers.push(markerClose);
    } else {
      const markerOpen = {
        time: openDateObj,
        position: "belowBar",
        color: "#2196F3",
        shape: "arrowUp",
        text: `Buy @ ${item.open_price}`,
      };
      const markerClose = {
        time: closeDateObj,
        position: "aboveBar",
        color: "#e91e63",
        shape: "arrowDown",
        text: `Sell @ ${item.close_price}`,
      };
      markers.push(markerOpen);
      markers.push(markerClose);
    }
  });

  return markers;
};
