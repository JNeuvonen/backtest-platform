import React from "react";
import { WithLabel } from "../../components/form/WithLabel";
import { ChakraInput } from "../../components/chakra/input";
import { IconButton } from "@chakra-ui/react";
import { BUTTON_VARIANTS } from "../../theme";
import { IoIosRefresh } from "react-icons/io";
import { useAppContext } from "../../context/app";
import { createPredServApiKey } from "../../clients/requests";

export const SettingsPage = () => {
  const appContext = useAppContext();

  const refetchApiKey = async () => {
    const apiKey = await createPredServApiKey();
    appContext.updatePredServAPIKey(apiKey);
  };

  return (
    <div>
      <WithLabel>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <ChakraInput
            label="Prediction service API key"
            containerStyle={{ width: "500px" }}
            disabled={true}
            value={appContext.appSettings?.predServAPIKey || ""}
          />
          <IconButton
            aria-label="refresh-btn"
            marginTop={"32px"}
            variant={BUTTON_VARIANTS.grey2}
            icon={<IoIosRefresh size={25} />}
            onClick={refetchApiKey}
          />
        </div>
      </WithLabel>
    </div>
  );
};
