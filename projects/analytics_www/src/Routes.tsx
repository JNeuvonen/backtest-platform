import { Route, Routes } from "react-router-dom";
import {
  ProfilePage,
  RootPage,
  StrategyPage,
  StrategiesPage,
  LsStrategyPage,
} from "./pages";
import { PATHS } from "./utils";

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path={PATHS.root} element={<RootPage />} />
      <Route path={PATHS.strategies} element={<StrategiesPage />} />
      <Route path={PATHS.profile} element={<ProfilePage />} />
      <Route path={PATHS.strategy} element={<StrategyPage />} />
      <Route path={PATHS.lsStrategy} element={<LsStrategyPage />} />
    </Routes>
  );
};
