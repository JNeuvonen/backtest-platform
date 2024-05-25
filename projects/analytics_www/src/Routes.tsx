import { Route, Routes } from "react-router-dom";
import { ProfilePage, RootPage } from "./pages";
import { StrategiesPage } from "./pages/strategies/root";
import { PATHS } from "./utils";

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path={PATHS.root} element={<RootPage />} />
      <Route path={PATHS.strategies} element={<StrategiesPage />} />
      <Route path={PATHS.profile} element={<ProfilePage />} />
    </Routes>
  );
};
