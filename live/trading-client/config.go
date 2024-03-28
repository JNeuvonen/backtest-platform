package main

import (
	"os"

	"github.com/adshao/go-binance/v2"
	"github.com/joho/godotenv"
)

func getBinanceKeys() (apiKey string, apiSecret string) {
	err := godotenv.Load()
	if err != nil {
		panic("No ENV variables provided")
	}

	apiKey = os.Getenv("API_KEY")
	apiSecret = os.Getenv("API_SECRET")

	if apiKey == "" || apiSecret == "" {
		panic("No API_KEY or API_SECRET provided")
	}

	return apiKey, apiSecret
}

func isProd() (isProd bool) {
	err := godotenv.Load()
	if err != nil {
		panic("No ENV variables provided")
	}

	isProd = os.Getenv("IS_PROD") == "1"
	return isProd
}

func getBinanceClient(apiKey string, apiSecret string) (client *binance.Client) {
	client = binance.NewClient(apiKey, apiSecret)

	if !isProd() {
		binance.UseTestnet = true
	}

	return client
}
