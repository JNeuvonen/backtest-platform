import React, { useEffect, useState } from "react";
import { usePathParams } from "../../../../hooks/usePathParams";
import { useTrainJobDetailed } from "../../../../clients/queries/queries";
import { useAppContext } from "../../../../context/app";
import { LAYOUT } from "../../../../utils/constants";
import {
  Box,
  Heading,
  Spinner,
  Stack,
  StackDivider,
  Text,
} from "@chakra-ui/react";
import { ShareYAxisTwoLineChart } from "../../../../components/charts/ShareYAxisLineChart";
import { ChakraSlider } from "../../../../components/chakra/Slider";
import { GenericBarChart } from "../../../../components/charts/BarChart";
import {
  calculateStdDevAndMean,
  getArrayMax,
  getArrayMedian,
  getArrayMin,
  getNormalDistributionItems,
  roundNumberDropRemaining,
} from "../../../../utils/number";
import { WithLabel } from "../../../../components/form/WithLabel";
import { GenericAreaChart } from "../../../../components/charts/AreaChart";
import { ChakraCard } from "../../../../components/chakra/Card";

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

  const { data } = useTrainJobDetailed(trainJobId);
  const { setInnerSideNavWidth } = useAppContext();
  const [epochSlider, setEpochSlider] = useState(1);

  useEffect(() => {}, [epochSlider, data?.epochs]);

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

  const getDataForSortedPredictions = (epochPreds: number[]) => {
    const ret = [] as { prediction: number; num: number }[];
    const copy = [...epochPreds];
    copy.sort((a, b) => a - b);
    const increment = Math.max(Math.floor(copy.length / 500), 1);
    for (let i = 0; i < copy.length; i += increment) {
      const item = copy[i];
      ret.push({
        prediction: item,
        num: i,
      });
    }
    return ret;
  };

  const trainingProgessTicks = generateTrainingProgressChart();
  let epochPredictions: number[] =
    data.epochs.length > 0
      ? data.epochs[epochSlider - 1].val_predictions.map((item) => {
          return item["prediction"];
        })
      : [];

  const { mean, stdDev } = calculateStdDevAndMean(epochPredictions);

  return (
    <div>
      <WithLabel
        label={
          <Heading size="md">
            Epochs ran: {data.train_job.epochs_ran}/{data.train_job.num_epochs}
          </Heading>
        }
      >
        <ShareYAxisTwoLineChart
          data={trainingProgessTicks}
          xKey="epoch"
          line1Key="trainLoss"
          line2Key="valLoss"
          height={500}
        />
      </WithLabel>

      <Heading size="md">Stats</Heading>

      <Stack
        direction={"row"}
        alignItems={"center"}
        gap={"48px"}
        justifyContent={"space-between"}
        marginTop={"16px"}
      >
        <ChakraSlider
          label={`Epoch number: ${epochSlider}`}
          containerStyles={{ maxWidth: "300px" }}
          min={1}
          max={data.epochs.length}
          onChange={setEpochSlider}
          defaultValue={1}
          value={epochSlider}
        />
        <ChakraCard
          heading={<Heading size="md">Stats on validation set</Heading>}
        >
          <Stack divider={<StackDivider />} spacing="4" direction={"row"}>
            <Box>
              <Heading size="xs" textTransform="uppercase">
                Mean
              </Heading>
              <Text pt="2" fontSize="sm">
                {roundNumberDropRemaining(mean, 5)}
              </Text>
            </Box>
            <Box>
              <Heading size="xs" textTransform="uppercase">
                Standard deviation
              </Heading>
              <Text pt="2" fontSize="sm">
                {roundNumberDropRemaining(stdDev, 5)}
              </Text>
            </Box>
            <Box>
              <Heading size="xs" textTransform="uppercase">
                Median
              </Heading>
              <Text pt="2" fontSize="sm">
                {roundNumberDropRemaining(getArrayMedian(epochPredictions), 5)}
              </Text>
            </Box>
            <Box>
              <Heading size="xs" textTransform="uppercase">
                Max
              </Heading>
              <Text pt="2" fontSize="sm">
                {roundNumberDropRemaining(getArrayMax(epochPredictions), 5)}
              </Text>
            </Box>
            <Box>
              <Heading size="xs" textTransform="uppercase">
                Min
              </Heading>
              <Text pt="2" fontSize="sm">
                {roundNumberDropRemaining(getArrayMin(epochPredictions), 5)}
              </Text>
            </Box>
          </Stack>
        </ChakraCard>
      </Stack>

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

      <ChakraSlider
        label={`Epoch number: ${epochSlider}`}
        containerStyles={{ maxWidth: "300px" }}
        min={1}
        max={data.epochs.length}
        onChange={setEpochSlider}
        defaultValue={1}
        value={epochSlider}
      />
      <WithLabel
        label="Validation predictions sorted by smallest first"
        containerStyles={{ marginTop: "32px" }}
      >
        <GenericAreaChart
          data={getDataForSortedPredictions(epochPredictions)}
          yAxisKey="prediction"
          xAxisKey="num"
          containerStyles={{ marginTop: "16px" }}
        />
      </WithLabel>
    </div>
  );
};
