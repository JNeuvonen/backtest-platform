import React from "react";
import { Route, Routes } from "react-router-dom";
import { Home } from "./Pages";
import { PATHS } from "./utils/constants";
import { DatasetsPage } from "./pages/Datasets";
import { BinancePage } from "./pages/BinancePage";
import { AvailablePage } from "./pages/Available";

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path={PATHS.datasets.path} element={<DatasetsPage />}>
        <Route
          path={PATHS.datasets.subpaths.available.path}
          element={<AvailablePage />}
        />
        <Route
          path={PATHS.datasets.subpaths.stock_market.path}
          element={<DatasetsPage />}
        />
        <Route
          path={PATHS.datasets.subpaths.binance.path}
          element={<BinancePage />}
        />
      </Route>
      <Route path={PATHS.simulate.path} element={<Home />} />
    </Routes>
  );
};
