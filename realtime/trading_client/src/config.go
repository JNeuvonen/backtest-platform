package main

import (
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

type TradingConfig struct {
	TestnetApiKey      string
	TestnetApiSecret   string
	ProdApiKey         string
	ProdApiSecret      string
	LiveTradingEnabled int32
}

type PredServerConfig struct {
	URI     string
	API_KEY string
}

func getPredServerConfig() PredServerConfig {
	err := godotenv.Load()
	if err != nil {
		panic("No ENV variables provided")
	}

	predServerURI := os.Getenv("PREDICTION_SERVICE_URI")
	predServerAPIKey := os.Getenv("PREDICTION_SERVICE_API_KEY")

	if predServerURI == "" || predServerAPIKey == "" {
		panic("PREDICTION_SERVICE_URI or PREDICTION_SERVICE_API_KEY is not set or empty")
	}

	config := PredServerConfig{
		URI:     predServerURI,
		API_KEY: predServerAPIKey,
	}

	return config
}

func getTradingConfig() TradingConfig {
	err := godotenv.Load()
	if err != nil {
		panic("No ENV variables provided")
	}

	config := TradingConfig{
		TestnetApiKey:    os.Getenv("TESTNET_API_KEY"),
		TestnetApiSecret: os.Getenv("TESTNET_API_SECRET"),
		ProdApiKey:       os.Getenv("PROD_API_KEY"),
		ProdApiSecret:    os.Getenv("PROD_API_SECRET"),
	}

	liveTradingEnabledStr := os.Getenv("LIVE_TRADING_ENABLED")
	liveTradingEnabledInt, err := strconv.ParseInt(liveTradingEnabledStr, 10, 32)
	if err != nil {
		panic("LIVE_TRADING_ENABLED must be an integer")
	}
	config.LiveTradingEnabled = int32(liveTradingEnabledInt)

	return config
}

func isProd() (isProd bool) {
	err := godotenv.Load()
	if err != nil {
		panic("No ENV variables provided")
	}

	isProd = os.Getenv("IS_PROD") == "1"
	return isProd
}
