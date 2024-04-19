import React from "react";
import { Route, Routes } from "react-router-dom";
import { PATHS } from "./utils/constants";
import { DatasetColumnInfoPage } from "./pages/data/dataset/column";
import { DataRouteIndex } from "./pages";
import { DatasetIndex } from "./pages/data/dataset/index";
import { DatasetModelIndex } from "./pages/data/model";
import { ModelInfoPage } from "./pages/data/model/info";
import { TrainJobIndex } from "./pages/data/model/trainjob";
import { SimulateIndex } from "./pages/simulate";
import { SimulateDatasetIndex } from "./pages/simulate/dataset";
import { BacktestProvider } from "./context/backtest";
import { DatasetBacktestPage } from "./pages/simulate/dataset/backtest";
import { SettingsPage } from "./pages/settings";
import { AllMassBacktests } from "./pages/mass-backtest/all-backtests";
import { InvidualMassbacktestDetailsPage } from "./pages/mass-backtest/backtest";

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<div>root</div>} />
      <Route path={PATHS.data.index} element={<DataRouteIndex />}>
        <Route path={PATHS.data.dataset.index} element={<DatasetIndex />} />
        <Route path={PATHS.data.model.index} element={<DatasetModelIndex />}>
          <Route path={PATHS.data.model.info} element={<ModelInfoPage />} />
        </Route>
        <Route path={PATHS.data.dataset.editor} element={<DatasetIndex />} />
        <Route path={PATHS.data.model.train} element={<TrainJobIndex />} />
      </Route>
      <Route
        path={PATHS.data.dataset.column}
        element={<DatasetColumnInfoPage />}
      />
      <Route path={PATHS.train} element={<TrainJobIndex />} />
      <Route path={PATHS.simulate.path} element={<SimulateIndex />} />
      <Route
        path={PATHS.simulate.dataset}
        element={
          <BacktestProvider>
            <SimulateDatasetIndex />
          </BacktestProvider>
        }
      />
      <Route
        path={PATHS.simulate.backtest}
        element={
          <BacktestProvider>
            <DatasetBacktestPage />
          </BacktestProvider>
        }
      />
      <Route path={PATHS.mass_backtest.root} element={<AllMassBacktests />} />
      <Route
        path={PATHS.mass_backtest.backtest}
        element={<InvidualMassbacktestDetailsPage />}
      />
      <Route path={PATHS.settings} element={<SettingsPage />} />
    </Routes>
  );
};
