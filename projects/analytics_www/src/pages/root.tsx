import { useBalanceSnapshotsQuery } from "src/http/queries";

export const RootPage = () => {
  const balanceSnapShots = useBalanceSnapshotsQuery();
  return <div>Root</div>;
};
