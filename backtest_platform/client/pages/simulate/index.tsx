import { Heading } from "@chakra-ui/react";
import React from "react";
import ExternalLink from "../../components/ExternalLink";
import { PATHS } from "../../utils/constants";

const addModeToPath = (mode: string) => {
  return PATHS.simulate.select_dataset + `?mode=${mode}`;
};

export const UI_BACKTEST_MODES = {
  simple: "invidual",
  long_short_simple: "long_short",
};

export const SimulateSelectMode = () => {
  return (
    <div>
      <Heading size={"md"}>Select backtest mode</Heading>
      <div
        style={{
          marginTop: "16px",
          gap: "16px",
          display: "flex",
          alignItems: "center",
        }}
      >
        <ExternalLink
          to={addModeToPath(UI_BACKTEST_MODES.simple)}
          linkText={"Invidual"}
        />
        <ExternalLink
          to={PATHS.simulate.bulk_long_short}
          linkText={"Bulk Long/short"}
        />
      </div>
    </div>
  );
};
