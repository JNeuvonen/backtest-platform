import { Route, Routes } from "react-router-dom";
import {
  AssetsPage,
  LongShortTickersPage,
  LsStrategyPage,
  ProfilePage,
  RootPage,
  StrategiesPage,
  StrategyPage,
  StrategySymbolsPage,
} from "./pages";
import { ViewTradePage } from "./pages/trade";
import { PATHS } from "./utils";

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path={PATHS.root} element={<RootPage />} />
      <Route path={PATHS.strategies} element={<StrategiesPage />} />
      <Route path={PATHS.profile} element={<ProfilePage />} />
      <Route path={PATHS.strategy} element={<StrategyPage />} />
      <Route path={PATHS.lsStrategy} element={<LsStrategyPage />} />
      <Route path={PATHS.strategySymbols} element={<StrategySymbolsPage />} />
      <Route path={PATHS.viewTradePath} element={<ViewTradePage />} />
      <Route path={PATHS.lsTickers} element={<LongShortTickersPage />} />
      <Route path={PATHS.assets} element={<AssetsPage />} />
    </Routes>
  );
};
