import { useState, useEffect } from "react";

const usePath = () => {
  const [path, setPath] = useState(window.location.pathname);

  useEffect(() => {
    const handleLocationChange = () => setPath(window.location.pathname);

    window.history.pushState = ((f) =>
      function pushState() {
        const ret = f.apply(this, arguments);
        window.dispatchEvent(new Event("pushstate"));
        window.dispatchEvent(new Event("locationchange"));
        return ret;
      })(window.history.pushState);

    window.history.replaceState = ((f) =>
      function replaceState() {
        const ret = f.apply(this, arguments);
        window.dispatchEvent(new Event("replacestate"));
        window.dispatchEvent(new Event("locationchange"));
        return ret;
      })(window.history.replaceState);

    window.addEventListener("popstate", () =>
      window.dispatchEvent(new Event("locationchange"))
    );
    window.addEventListener("locationchange", handleLocationChange);

    return () => {
      window.removeEventListener("locationchange", handleLocationChange);
    };
  }, []);

  return { path };
};

export default usePath;
