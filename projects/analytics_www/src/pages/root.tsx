import { useAuth0 } from "@auth0/auth0-react";

export const RootPage = () => {
  const {
    loginWithRedirect,
    logout,
    isAuthenticated,
    user,
    getAccessTokenSilently,
  } = useAuth0();
  console.log(isAuthenticated, user);
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
  return (
    <div>
      <button onClick={() => loginWithRedirect()}>Log In</button>

      <button onClick={getToken}>Token</button>
    </div>
  );
};
