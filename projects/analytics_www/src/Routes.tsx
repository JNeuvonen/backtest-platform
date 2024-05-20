import { Route, Routes } from "react-router-dom";
import { RootPage } from "./pages";
import { PATHS } from "./utils";

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path={PATHS.root} element={<RootPage />} />
    </Routes>
  );
};
