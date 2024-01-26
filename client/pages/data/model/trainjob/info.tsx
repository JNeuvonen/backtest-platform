import React, { useEffect } from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import { useTrainJobDetailed } from "../../../../clients/queries/queries";
import { useAppContext } from "../../../../context/app";
import { useMessageListener } from "../../../../hooks/useMessageListener";
import { DOM_EVENT_CHANNELS, LAYOUT, PATHS } from "../../../../utils/constants";
import { Spinner } from "@chakra-ui/react";
import { Breadcrumbs } from "../../../../components/chakra/Breadcrumbs";
import {
  getDatasetInfoPagePath,
  getModelInfoPath,
} from "../../../../utils/navigate";
import Title from "../../../../components/typography/Title";
import { ShareYAxisTwoLineChart } from "../../../../components/charts/ShareYAxisLineChart";

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

  return (
    <div>
      <Title>
        Epochs ran: {data.train_job.epochs_ran}/{data.train_job.num_epochs}
      </Title>
      <ShareYAxisTwoLineChart
        data={trainingProgessTicks}
        xKey="epoch"
        line1Key="trainLoss"
        line2Key="valLoss"
        height={500}
      />
    </div>
  );
};
