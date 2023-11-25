import React from "react";
import { AppRoutes } from "./Routes";
import { BrowserRouter as Router } from "react-router-dom";
import { Layout } from "./components/layout";

function App() {
  return (
    <Router>
      <Layout>
        <AppRoutes />
      </Layout>
    </Router>
  );
}

export default App;
