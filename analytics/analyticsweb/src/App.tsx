import React from "react";
import { BrowserRouter as Router } from "react-router-dom";
import { LayoutContainer } from "./layout";

function App() {
  return (
    <Router>
      <LayoutContainer>
        <div>Test</div>
      </LayoutContainer>
    </Router>
  );
}

export default App;
