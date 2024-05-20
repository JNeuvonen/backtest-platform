import { BrowserRouter as Router } from "react-router-dom";
import { LayoutContainer } from "./layout";
import { AppRoutes } from "./Routes";

function App() {
  return (
    <Router>
      <LayoutContainer>
        <AppRoutes />
      </LayoutContainer>
    </Router>
  );
}

export default App;
