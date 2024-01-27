import { Spinner } from "@chakra-ui/react";
import React from "react";

export const CenteredSpinner = () => {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: "100%",
        height: "100%",
      }}
    >
      <Spinner />
    </div>
  );
};
