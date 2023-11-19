import React from "react";
import { useDatasetsQuery } from "../clients/queries";

export const AvailablePage = () => {
  const datasets = useDatasetsQuery();
  return <div>Available</div>;
};
