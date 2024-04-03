package main

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMarketOrder(t *testing.T) {
	tradingConfig := GetTradingConfig()

	binanceClient := NewBinanceClient(tradingConfig)
	binanceClient.SendOrder("BTCUSDT", "BUY", "MARKET", 0.001, true)
}

func TestCallingPredServer(t *testing.T) {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	client := NewHttpClient(predServConfig.URI, headers)
	_, err := client.Get("")
	assert.Nil(t, err, "Error calling prediction server: %v", err)
}

func TestFetchingStrategies(t *testing.T) {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	client := NewHttpClient(predServConfig.URI, headers)
	strategies := client.FetchStrategies()

	for _, strategy := range strategies {
		fmt.Printf(
			"Strategy ID: %d, Symbol: %s, Priority: %d\n",
			strategy.ID,
			strategy.Symbol,
			strategy.Priority,
		)
	}
}

func TestCloudLog(t *testing.T) {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	client := NewHttpClient(predServConfig.URI, headers)
	err := client.CreateCloudLog("hello_world", "info")
	assert.Nil(t, err, "Error creating log: %v", err)
}

func TestFetchingBalances(t *testing.T) {
	tradingConfig := GetTradingConfig()
	binanceClient := NewBinanceClient(tradingConfig)
	binanceClient.FetchBalances()
}

func TestFetchingPrice(t *testing.T) {
	tradingConfig := GetTradingConfig()
	binanceClient := NewBinanceClient(tradingConfig)

	price, err := binanceClient.FetchLatestPrice("BTCUSDT")
	if err != nil {
		fmt.Println(err.Error())
	}
	fmt.Println(price)
}

func TestFetchingAllPrices(t *testing.T) {
	prices, err := getAllUSDTPrices()
	if err != nil {
		fmt.Println(err.Error())
	}

	for _, price := range prices {
		fmt.Println(price.Price)
	}
}

func TestCalculatingPortfolioUSDTValue(t *testing.T) {
	tradingConfig := GetTradingConfig()
	binanceClient := NewBinanceClient(tradingConfig)

	value, err := binanceClient.GetPortfolioValueInUSDT()
	if err != nil {
		fmt.Println(err.Error())
	}
	fmt.Println(value)
}

// func TestTradingLoop(t *testing.T) {
// 	timeout := time.After(10 * time.Second)
// 	done := make(chan bool)
//
// 	go func() {
// 		TradingLoop()
// 		done <- true
// 	}()
//
// 	select {
// 	case <-timeout:
// 		t.Log("Test reached 10 second timeout")
// 	case <-done:
// 	}
// }
