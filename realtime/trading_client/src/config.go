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
