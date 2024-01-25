import React from "react";
import { useAppContext } from "../context/App";
import { LAYOUT } from "../utils/constants";
import { COLOR_BG_PRIMARY, COLOR_BG_PRIMARY_SHADE_TWO } from "../utils/colors";
import { Button } from "@chakra-ui/react";
import { BUTTON_VARIANTS } from "../theme";
import { MdOutlinePause } from "react-icons/md";
import { roundNumberDropRemaining } from "../utils/number";
import { stopTrain } from "../clients/requests";
import { Link } from "react-router-dom";
import { getTrainJobFromToolbar } from "../utils/navigate";

const TrainingToolbar = () => {
  const {
    innerSideNavWidth,
    contentIndentPx,
    epochsRan,
    maximumEpochs,
    trainLosses,
    valLosses,
    trainJobId,
  } = useAppContext();

  const cancelTrain = async () => {
    await stopTrain(trainJobId);
  };
  return (
    <div
      style={{
        width: "100%",
        position: "fixed",
        height: LAYOUT.training_toolbar_height,
        background: COLOR_BG_PRIMARY,
        zIndex: 2,
        top: 0,
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "8px",
        paddingLeft: innerSideNavWidth + contentIndentPx,
        paddingRight: "16px",
        borderBottom: `1px solid ${COLOR_BG_PRIMARY_SHADE_TWO}`,
      }}
    >
      <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
        Epoch: {epochsRan}/{maximumEpochs}, Train loss:{" "}
        {roundNumberDropRemaining(trainLosses[trainLosses.length - 1], 3)}, Val
        loss: {roundNumberDropRemaining(valLosses[valLosses.length - 1], 3)}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <Link className="link-default" to={getTrainJobFromToolbar(trainJobId)}>
          Details
        </Link>
        <Button
          leftIcon={<MdOutlinePause />}
          variant={BUTTON_VARIANTS.grey2}
          style={{ height: "28px" }}
          key={4}
          onClick={cancelTrain}
        >
          Pause
        </Button>
      </div>
    </div>
  );
};

export const LayoutToolbar = () => {
  const { toolbarMode } = useAppContext();

  if (toolbarMode === "TRAINING") {
    return <TrainingToolbar />;
  }
  return null;
};
