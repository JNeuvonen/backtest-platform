import { BrowserRouter as Router } from "react-router-dom";
import Auth0ProviderWithHistory from "./context/auth";
import { LayoutContainer } from "./layout";
import { AppRoutes } from "./Routes";

function App() {
  return (
    <Router>
      <Auth0ProviderWithHistory>
        <LayoutContainer>
          <AppRoutes />
        </LayoutContainer>
      </Auth0ProviderWithHistory>
    </Router>
  );
}

export default App;
