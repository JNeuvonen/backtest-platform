import React, { useEffect } from "react";
import { useTrainJobDetailed } from "../clients/queries/queries";
import { usePathParams } from "../hooks/usePathParams";
import { Spinner } from "@chakra-ui/react";
import { ShareYAxisTwoLineChart } from "../components/charts/ShareYAxisLineChart";
import { useAppContext } from "../context/App";
import {
  DOM_EVENT_CHANNELS,
  LAYOUT,
  PATHS,
  PATH_KEYS,
} from "../utils/constants";
import { useMessageListener } from "../hooks/useMessageListener";
import Title from "../components/typography/Title";
import { Breadcrumbs } from "../components/chakra/Breadcrumbs";
import { getDatasetInfoPagePath, getModelInfoPath } from "../utils/navigate";

type TrainingProgessChartTicks = {
  valLoss: number;
  trainLoss: number;
  epoch: number;
}[];

export const TrainJobPage = () => {
  const { trainJobId, datasetName } = usePathParams<{
    trainJobId: string;
    datasetName?: string;
  }>();

  const { data, refetch } = useTrainJobDetailed(trainJobId);
  const { setInnerSideNavWidth } = useAppContext();

  useMessageListener({
    messageName: DOM_EVENT_CHANNELS.refetch_component,
    messageCallback: refetch,
  });

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
    <div style={{ paddingTop: "16px" }}>
      <Breadcrumbs
        items={[
          { label: "Data", href: PATHS.data.index },
          {
            label: "Dataset",
            href: getDatasetInfoPagePath(data.dataset.dataset_name),
          },
          {
            label: "Models",
            href: getModelInfoPath(
              data.dataset.dataset_name,
              data.model.model_name
            ),
          },
          { label: "Training info", href: "test" },
        ]}
      />
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
