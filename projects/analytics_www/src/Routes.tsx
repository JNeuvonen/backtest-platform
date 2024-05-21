import { Route, Routes } from "react-router-dom";
import { ProfilePage, RootPage } from "./pages";
import { PATHS } from "./utils";

export const AppRoutes = () => {
  return (
    <Routes>
      <Route path={PATHS.root} element={<RootPage />} />
      <Route path={PATHS.profile} element={<ProfilePage />} />
    </Routes>
  );
};
