import React, {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from "react";
import { useAuth0 } from "@auth0/auth0-react";
import { LoginPage } from "src/pages/login";
import { DiskManager } from "common_js";
import { DISK_KEYS } from "src/utils/keys";

interface AccessTokenContextType {
  accessToken: string | null;
}

interface AccessTokenProviderProps {
  children: ReactNode;
}

export const AccessTokenContext = createContext<AccessTokenContextType>(
  {} as AccessTokenContextType,
);

const diskManager = new DiskManager(DISK_KEYS.access_token);

export const AccessTokenProvider: React.FC<AccessTokenProviderProps> = ({
  children,
}) => {
  const { isAuthenticated, getAccessTokenSilently } = useAuth0();
  const [accessToken, setAccessToken] = useState<string | null>(null);

  useEffect(() => {
    const fetchAccessToken = async () => {
      try {
        const token = await getAccessTokenSilently();
        setAccessToken(token);
        diskManager.save({ token });
      } catch (error) {
        console.error("Error fetching access token", error);
      }
    };

    if (isAuthenticated) {
      fetchAccessToken();
    }
  }, [isAuthenticated, getAccessTokenSilently]);

  if (!accessToken) {
    return <LoginPage />;
  }

  return (
    <AccessTokenContext.Provider value={{ accessToken }}>
      {children}
    </AccessTokenContext.Provider>
  );
};

export const useAccessTokenContext = () => useContext(AccessTokenContext);
