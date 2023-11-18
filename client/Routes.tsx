import React from "react";
import { Route, Routes } from "react-router-dom";
import { Home } from "./Pages";
import { PATHS } from "./utils/constants";
import { DatasetsPage } from "./pages/Datasets";

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path={PATHS.datasets} element={<DatasetsPage />} />
      <Route path={PATHS.simulate} element={<Home />} />
    </Routes>
  );
};
