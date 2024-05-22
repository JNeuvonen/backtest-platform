import { useAuth0 } from "@auth0/auth0-react";
import {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from "react";

interface UserContextType {
  user: any;
}

interface UserProviderProps {
  children: ReactNode;
}

export const UserContext = createContext<UserContextType>(
  {} as UserContextType,
);

export const UserProvider: React.FC<UserProviderProps> = ({ children }) => {
  const { isAuthenticated, user, getAccessTokenSilently } = useAuth0();
  const [userFromDb, setUserFromDb] = useState(null);

  useEffect(() => {
    const asyncHelper = async () => {
      const token = await getAccessTokenSilently({
        authorizationParams: {
          audience: `https://${process.env.REACT_APP_AUTH0_DOMAIN}/api/v2/`,
          scope: "read:current_user",
        },
      });
    };

    asyncHelper();
  }, [isAuthenticated, user]);

  return (
    <UserContext.Provider value={{ user: null }}>
      {children}
    </UserContext.Provider>
  );
};

export const useUserContext = () => useContext(UserContext);
