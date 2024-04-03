package main

import (
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

type TradingConfig struct {
	ApiKey             string
	ApiSecret          string
	BaseUrl            string
	LiveTradingEnabled int32
}

type PredServerConfig struct {
	URI     string
	API_KEY string
}

func GetPredServerConfig() PredServerConfig {
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

func GetBinanceAPIBaseURL() string {
	err := godotenv.Load()

	liveTradingEnabledStr := os.Getenv("LIVE_TRADING_ENABLED")
	liveTradingEnabledInt, err := strconv.ParseInt(liveTradingEnabledStr, 10, 32)
	if err != nil {
		panic("LIVE_TRADING_ENABLED must be an integer 0 or 1")
	}

	if liveTradingEnabledInt == 1 {
		return MAINNET
	}

	return TESTNET
}

func GetTradingConfig() TradingConfig {
	err := godotenv.Load()
	if err != nil {
		panic("No ENV variables provided")
	}

	config := TradingConfig{
		ApiKey:    os.Getenv("API_KEY"),
		ApiSecret: os.Getenv("API_SECRET"),
	}

	liveTradingEnabledStr := os.Getenv("LIVE_TRADING_ENABLED")
	liveTradingEnabledInt, err := strconv.ParseInt(liveTradingEnabledStr, 10, 32)
	if err != nil {
		panic("LIVE_TRADING_ENABLED must be an integer 0 or 1")
	}
	config.LiveTradingEnabled = int32(liveTradingEnabledInt)

	if config.LiveTradingEnabled == 1 {
		config.BaseUrl = MAINNET
	} else {
		config.BaseUrl = TESTNET
	}

	return config
}

func IsProd() (isProd bool) {
	err := godotenv.Load()
	if err != nil {
		panic("No ENV variables provided")
	}

	isProd = os.Getenv("IS_PROD") == "1"
	return isProd
}
