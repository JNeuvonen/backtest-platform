import React from "react";
import { useTrainJobDetailed } from "../clients/queries/queries";
import { usePathParams } from "../hooks/usePathParams";
import { Spinner } from "@chakra-ui/react";
import { ShareYAxisTwoLineChart } from "../components/charts/ShareYAxisLineChart";

type TrainingProgessChartTicks = {
  valLoss: number;
  trainLoss: number;
  epoch: number;
}[];

export const TrainJobPage = () => {
  const { trainJobId } = usePathParams<{ trainJobId: string }>();
  const { data } = useTrainJobDetailed(trainJobId);

  const generateTrainingProgressChart = () => {
    if (!data?.epochs) return [];

    const ret = [] as TrainingProgessChartTicks;

    const increment = Math.max(Math.floor(data.epochs.length / 15), 1);

    for (let i = 0; i < data.epochs.length; i += increment) {
      const dataItem = data.epochs[i];
      ret.push({
        valLoss: dataItem.val_loss,
        trainLoss: dataItem.train_loss,
        epoch: i,
      });
    }

    return ret;
  };

  if (!data)
    return (
      <div>
        <Spinner />
      </div>
    );

  const trainingProgessTicks = generateTrainingProgressChart();

  return (
    <div>
      <ShareYAxisTwoLineChart
        data={trainingProgessTicks}
        xKey="epoch"
        line1Key="trainLoss"
        line2Key="valLoss"
        width={500}
        height={500}
      />
    </div>
  );
};
