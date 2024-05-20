import { Heading, Spinner } from "@chakra-ui/react";
import React from "react";

interface Props {
  text: string;
}

export const CenteredSpinner = ({ text }: Props) => {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: "100%",
        height: "50vh",
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          width: "100%",
          height: "50vh",
          gap: "32px",
        }}
      >
        <Heading>{text}</Heading>
        <Spinner />
      </div>
    </div>
  );
};
