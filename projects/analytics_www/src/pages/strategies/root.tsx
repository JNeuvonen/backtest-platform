import { Spinner } from "@chakra-ui/react";
import { useBinanceSpotPriceInfo, useStrategiesQuery } from "src/http/queries";
import { ChakraTabs } from "src/components/chakra/Tabs";
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";
import "ag-grid-community/styles/ag-theme-alpine.css";
import "ag-grid-community/styles/ag-theme-balham.css";
import { DirectionalStrategiesTab } from ".";
import { PairTradeTab } from "./pairtradetab";

export const StrategiesPage = () => {
  const strategiesQuery = useStrategiesQuery();

  if (strategiesQuery.isLoading || !strategiesQuery.data) {
    return (
      <div>
        <Spinner />
      </div>
    );
  }
  return (
    <div>
      <div>
        <ChakraTabs
          labels={["Directional", "Pair-trade"]}
          tabs={[
            <DirectionalStrategiesTab strategiesRes={strategiesQuery.data} />,
            <PairTradeTab strategiesRes={strategiesQuery.data} />,
          ]}
          top={16}
          style={{ overflowY: "scroll" }}
        />
      </div>
    </div>
  );
};
