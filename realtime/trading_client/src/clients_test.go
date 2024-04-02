package main

import (
	"encoding/json"
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

	response, err := client.Get(PRED_SERV_V1_STRAT)
	assert.Nil(t, err, "Error calling prediction server: %v", err)

	if response != nil && len(response) > 0 {
		var strategyResponse StrategyResponse
		err := json.Unmarshal(response, &strategyResponse)
		assert.Nil(t, err, "Error unmarshaling JSON: %v", err)

		for _, strategy := range strategyResponse.Data {
			fmt.Printf(
				"Strategy ID: %d, Symbol: %s, Priority: %d\n",
				strategy.ID,
				strategy.Symbol,
				strategy.Priority,
			)
		}
	} else {
		fmt.Println("Empty response or nil")
	}
}
