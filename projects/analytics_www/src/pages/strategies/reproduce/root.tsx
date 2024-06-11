import { usePathParams } from "src/hooks";
import { useStrategyGroupQuery } from "src/http/queries";

export const ReproduceDirectionalLiveTrades = () => {
  const { strategyName } = usePathParams<{ strategyName: string }>();
  const strategyGroupQuery = useStrategyGroupQuery(strategyName);
  console.log(strategyGroupQuery);
  return <div>Test</div>;
};
