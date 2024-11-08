import React from "react";
import { Route, Routes } from "react-router-dom";
import { PATHS } from "./utils/constants";
import { DatasetColumnInfoPage } from "./pages/data/dataset/column";
import { DataRouteIndex } from "./pages";
import { DatasetIndex } from "./pages/data/dataset/index";
import { DatasetModelIndex } from "./pages/data/model";
import { ModelInfoPage } from "./pages/data/model/info";
import { TrainJobIndex } from "./pages/data/model/trainjob";
import { SimulateDatasetIndex } from "./pages/simulate/dataset";
import { BacktestProvider } from "./context/backtest";
import { DatasetBacktestPage } from "./pages/simulate/dataset/backtest";
import { SettingsPage } from "./pages/settings";
import { AllMassBacktests } from "./pages/mass-backtest/all-backtests";
import { InvidualMassbacktestDetailsPage } from "./pages/mass-backtest/backtest";
import { SimulateSelectDataset } from "./pages/simulate/select-dataset-page";
import { SimulateSelectMode } from "./pages/simulate";
import { BulkLongShortSimPage } from "./pages/simulate/bulk/long-short";
import { MassPairTradeProvider } from "./context/masspairtrade";
import { LongShortBacktestsDetailsView } from "./pages/simulate/bulk/backtest-details-view";
import { MachineLearningBacktestPage } from "./pages/simulate/machine-learning";
import { MLBasedBacktestProvider } from "./context/mlbasedbacktest";
import { MLBasedBacktestPage } from "./pages/ml-based-backtest";
import { RuleBasedSimOnUniverseBacktest } from "./pages/rule-based-on-universe-backtest";
import { ViewRuleBasedMassBacktestPage } from "./pages/rule-based-on-universe-backtest/view-by-id";
import { MultiStrategyBacktestPage } from "./pages/multistrategy";
import { MultiStrategyPageById } from "./pages/multistrategy/view-by-id";
import { BacktestTradesPage } from "./pages/simulate/dataset/backtest/BacktestTrades";

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
      <Route path={PATHS.simulate.path} element={<SimulateSelectMode />} />
      <Route
        path={PATHS.simulate.select_dataset}
        element={<SimulateSelectDataset />}
      />
      <Route
        path={PATHS.simulate.backtest_trades}
        element={
          <BacktestProvider>
            <BacktestTradesPage />
          </BacktestProvider>
        }
      />
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
      <Route
        path={PATHS.rule_based_on_universe.root}
        element={<RuleBasedSimOnUniverseBacktest />}
      />
      <Route
        path={PATHS.simulate.multistrategy_by_id}
        element={<MultiStrategyPageById />}
      />
      <Route
        path={PATHS.rule_based_on_universe.by_id}
        element={<ViewRuleBasedMassBacktestPage />}
      />
      <Route
        path={PATHS.simulate.bulk_long_short}
        element={
          <MassPairTradeProvider>
            <BulkLongShortSimPage />
          </MassPairTradeProvider>
        }
      />
      <Route
        path={PATHS.simulate.machine_learning}
        element={
          <MLBasedBacktestProvider>
            <MachineLearningBacktestPage />
          </MLBasedBacktestProvider>
        }
      />
      <Route
        path={PATHS.simulate.multistrategy}
        element={<MultiStrategyBacktestPage />}
      />
      <Route
        path={PATHS.mass_backtest.pairtrade}
        element={<LongShortBacktestsDetailsView />}
      />
      <Route path={PATHS.mass_backtest.root} element={<AllMassBacktests />} />
      <Route
        path={PATHS.mass_backtest.backtest}
        element={<InvidualMassbacktestDetailsPage />}
      />
      <Route path={PATHS.ml_based.backtest} element={<MLBasedBacktestPage />} />
      <Route path={PATHS.settings} element={<SettingsPage />} />
    </Routes>
  );
};
