package main

import (
	"os"
	"strings"

	"github.com/adshao/go-binance/v2"
	"github.com/joho/godotenv"
)

func getBinanceKeys() (apiKey string, apiSecret string, baseUrl string) {
	err := godotenv.Load()
	if err != nil {
		panic("No ENV variables provided")
	}

	apiKey = os.Getenv("API_KEY")
	apiSecret = os.Getenv("API_SECRET")
	baseUrl = os.Getenv("TRADING_BASE_URL")

	if apiKey == "" || apiSecret == "" || baseUrl == "" {
		panic("No API_KEY, API_SECRET or TRADING_BASE_URL provided")
	}

	return apiKey, apiSecret, baseUrl
}

func isProd() (isProd bool) {
	err := godotenv.Load()
	if err != nil {
		panic("No ENV variables provided")
	}

	isProd = os.Getenv("IS_PROD") == "1"
	return isProd
}

func assertIsTestnet(baseUrl string) {
	if !strings.Contains(strings.ToLower(baseUrl), "testnet") {
		panic("ENV TRADING_BASE_URL was not for testnet even though it is required.")
	}
}

func getBinanceClient(apiKey string, apiSecret string) (client *binance.Client) {
	client = binance.NewClient(apiKey, apiSecret)

	if !isProd() {
		binance.UseTestnet = true
	}

	return client
}
