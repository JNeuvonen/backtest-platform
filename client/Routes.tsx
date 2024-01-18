import React from "react";
import { Route, Routes } from "react-router-dom";
import { Home } from "./pages";
import { PATHS } from "./utils/constants";
import { BrowseDatasetsPage } from "./pages/Datasets";
import { DatasetIndex } from "./pages/dataset";
import { DatasetInfoPage } from "./pages/dataset/info";
import { DatasetEditorPage } from "./pages/dataset/editor";
import { DatasetColumnInfoPage } from "./pages/dataset/column";

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path={PATHS.datasets.path} element={<BrowseDatasetsPage />} />
      <Route path={PATHS.datasets.dataset} element={<DatasetIndex />}>
        <Route path={PATHS.datasets.info} element={<DatasetInfoPage />} />
        <Route path={PATHS.datasets.editor} element={<DatasetEditorPage />} />
        <Route path={PATHS.datasets.editor} element={<DatasetEditorPage />} />
        <Route
          path={PATHS.datasets.column}
          element={<DatasetColumnInfoPage />}
        />
      </Route>
      <Route path={PATHS.simulate.path} element={<Home />} />
    </Routes>
  );
};
