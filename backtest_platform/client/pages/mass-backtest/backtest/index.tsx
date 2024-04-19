import React from "react";
import { usePathParams } from "../../../hooks/usePathParams";
import { useMassbacktest } from "../../../clients/queries/queries";
import { Spinner } from "@chakra-ui/react";

interface PathParams {
  massBacktestId: number;
}

export const InvidualMassbacktestDetailsPage = () => {
  const { massBacktestId } = usePathParams<PathParams>();
  const massBacktestQuery = useMassbacktest(Number(massBacktestId));

  if (massBacktestQuery.isLoading || !massBacktestQuery.data) {
    return <Spinner />;
  }

  console.log(massBacktestQuery.data);

  return <div>Hello world</div>;
};
