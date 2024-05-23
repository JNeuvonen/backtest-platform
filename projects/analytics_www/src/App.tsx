import { BrowserRouter as Router } from "react-router-dom";
import Auth0ProviderWithHistory from "./context/auth";
import { UserProvider } from "./context/user";
import { LayoutContainer } from "./layout";
import { AppRoutes } from "./Routes";

function App() {
  return (
    <Router>
      <Auth0ProviderWithHistory>
        <UserProvider>
          <LayoutContainer>
            <AppRoutes />
          </LayoutContainer>
        </UserProvider>
      </Auth0ProviderWithHistory>
    </Router>
  );
}

export default App;
