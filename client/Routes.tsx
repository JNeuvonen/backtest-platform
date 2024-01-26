import React from "react";
import { Route, Routes } from "react-router-dom";
import { PATHS } from "./utils/constants";
import { DatasetColumnInfoPage } from "./pages/data/dataset/column";
import { DataRouteIndex } from "./pages";
import { DatasetIndex } from "./pages/data/dataset/index";
import { DatasetModelIndex } from "./pages/data/model";
import { ModelInfoPage } from "./pages/data/model/info";
import { TrainJobPage } from "./pages/trainjob";

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
        <Route path={PATHS.data.model.train} element={<TrainJobPage />} />
      </Route>
      <Route
        path={PATHS.data.dataset.column}
        element={<DatasetColumnInfoPage />}
      />

      <Route path={PATHS.train} element={<TrainJobPage />} />
      <Route path={PATHS.simulate.path} element={<div>sim</div>} />
    </Routes>
  );
};
