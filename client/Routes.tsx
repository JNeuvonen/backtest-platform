import React from "react";
import { Route, Routes } from "react-router-dom";
import { Home } from "./Pages";
import { PATHS } from "./utils/constants";
import { DatasetsPage } from "./pages/Datasets";
import { AvailablePage } from "./pages/Available";
import { BinancePage } from "./pages/BinancePage";

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path={PATHS.datasets.index} element={<DatasetsPage />}>
        <Route
          path={PATHS.datasets.subpaths.available.index}
          element={<AvailablePage />}
        />
        <Route
          path={PATHS.datasets.subpaths.binance.index}
          element={<BinancePage />}
        />
      </Route>
      <Route path={PATHS.simulate.index} element={<Home />} />
    </Routes>
  );
};
