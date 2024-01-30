import React, { useEffect, useState } from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import { useTrainJobDetailed } from "../../../../clients/queries/queries";
import { useAppContext } from "../../../../context/app";
import { LAYOUT } from "../../../../utils/constants";
import { Spinner } from "@chakra-ui/react";
import Title from "../../../../components/typography/Title";
import { ShareYAxisTwoLineChart } from "../../../../components/charts/ShareYAxisLineChart";
import { ChakraSlider } from "../../../../components/chakra/Slider";
import { GenericBarChart } from "../../../../components/charts/BarChart";
import { getNormalDistributionItems } from "../../../../utils/number";
import { WithLabel } from "../../../../components/form/WithLabel";

type TrainingProgessChartTicks = {
  valLoss: number;
  trainLoss: number;
  epoch: number;
}[];

export const TrainjobInfoPage = () => {
  const { trainJobId, datasetName } = usePathParams<{
    trainJobId: string;
    datasetName?: string;
  }>();

  const { data, refetch } = useTrainJobDetailed(trainJobId);
  const { setInnerSideNavWidth } = useAppContext();
  const [epochSlider, setEpochSlider] = useState(1);

  useEffect(() => {
    if (datasetName) {
      setInnerSideNavWidth(LAYOUT.inner_side_nav_width_px);
    }
    return () => {
      if (datasetName) {
        setInnerSideNavWidth(0);
      }
    };
  }, [datasetName]);

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
  const epochPredictions = JSON.parse(
    data.epochs[epochSlider - 1].val_predictions
  ).map((item: number[]) => item[0]);

  return (
    <div>
      <WithLabel
        label={`Epochs ran: ${data.train_job.epochs_ran}/${data.train_job.num_epochs}`}
      >
        <ShareYAxisTwoLineChart
          data={trainingProgessTicks}
          xKey="epoch"
          line1Key="trainLoss"
          line2Key="valLoss"
          height={500}
        />
      </WithLabel>
      <ChakraSlider
        label={`Epoch number: ${epochSlider}`}
        containerStyles={{ maxWidth: "300px" }}
        min={1}
        max={data.epochs.length}
        onChange={setEpochSlider}
        defaultValue={1}
      />

      <WithLabel
        label="Validation predictions normal distribution"
        containerStyles={{ marginTop: "16px" }}
      >
        <GenericBarChart
          data={getNormalDistributionItems(epochPredictions)}
          yAxisKey="count"
          xAxisKey="label"
          containerStyles={{ marginTop: "16px" }}
        />
      </WithLabel>
    </div>
  );
};
