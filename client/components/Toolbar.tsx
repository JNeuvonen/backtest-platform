import React from "react";
import { useAppContext } from "../context/App";
import { LAYOUT } from "../utils/constants";
import { COLOR_BG_PRIMARY, COLOR_BG_PRIMARY_SHADE_TWO } from "../utils/colors";
import { Button } from "@chakra-ui/react";
import { BUTTON_VARIANTS } from "../theme";
import { MdOutlinePause } from "react-icons/md";

const TrainingToolbar = () => {
  const {
    innerSideNavWidth,
    contentIndentPx,
    epochsRan,
    maximumEpochs,
    trainLosses,
    valLosses,
  } = useAppContext();
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
      <div>
        Epoch: {epochsRan}/{maximumEpochs}, Train loss:{" "}
        {trainLosses[trainLosses.length - 1]}, Val loss:{" "}
        {valLosses[valLosses.length - 1]}
      </div>
      <div>
        <Button
          leftIcon={<MdOutlinePause />}
          variant={BUTTON_VARIANTS.grey}
          style={{ height: "28px" }}
          key={4}
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
