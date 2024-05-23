import { useAuth0 } from "@auth0/auth0-react";
import { useToast } from "@chakra-ui/react";

export const RootPage = () => {
  const {
    loginWithRedirect,
    logout,
    isAuthenticated,
    user,
    getAccessTokenSilently,
  } = useAuth0();
  const toast = useToast();
  const getToken = async () => {
    const token = await getAccessTokenSilently({
      authorizationParams: {
        audience: `https://${process.env.REACT_APP_AUTH0_DOMAIN}/api/v2/`,
        scope: "read:current_user",
      },
    });
    const userInfoResponse = await fetch(
      `https://${process.env.REACT_APP_AUTH0_DOMAIN}/userinfo`,
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      },
    );
    const userInfo = await userInfoResponse.json();
    console.log(token);
    console.log(userInfo);
  };

  const testToast = () => {
    toast({
      title: "Title",
      status: "info",
      duration: 5000,
      isClosable: true,
    });
    console.log("exec here");
  };
  return (
    <div>
      <button onClick={() => loginWithRedirect()}>Log In</button>

      <button onClick={getToken}>Token</button>

      <button onClick={testToast}>Test toast</button>
    </div>
  );
};
