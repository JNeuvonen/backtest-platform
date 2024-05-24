import { ChakraProvider } from "@chakra-ui/provider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { ToastContainer } from "react-toastify";
import App from "./App";
import { AppProvider } from "./context";
import "./styles/css/styles.css";
import { customChakraTheme } from "./theme";

const queryClient = new QueryClient();

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement,
);
root.render(
  <React.StrictMode>
    <ChakraProvider theme={customChakraTheme}>
      <QueryClientProvider client={queryClient}>
        <AppProvider>
          <ToastContainer />
          <App />
        </AppProvider>
      </QueryClientProvider>
    </ChakraProvider>
  </React.StrictMode>,
);
