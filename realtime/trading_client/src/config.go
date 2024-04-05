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

func GetAccountName() string {
	err := godotenv.Load()
	if err != nil {
		panic("No ENV variables provided")
	}

	account := os.Getenv("ACCOUNT_NAME")

	if account == "" {
		panic("ACCOUNT_NAME is not set")
	}
	return account
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

func GetAllowedToSendOrders() bool {
	err := godotenv.Load()

	allowedToSendOrdersStr := os.Getenv("USE_REAL_ORDERS")
	allowedToSendOrdersInt, err := strconv.ParseInt(allowedToSendOrdersStr, 10, 32)
	if err != nil {
		panic("USE_REAL_ORDERS must be an integer 0 or 1")
	}
	return allowedToSendOrdersInt == 1
}

func GetBinanceAPIBaseURL() string {
	err := godotenv.Load()

	mainNetDataStr := os.Getenv("USE_MAINNET_DATA")
	mainNetDataInt, err := strconv.ParseInt(mainNetDataStr, 10, 32)
	if err != nil {
		panic("LIVE_TRADING_ENABLED must be an integer 0 or 1")
	}

	if mainNetDataInt == 1 {
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
