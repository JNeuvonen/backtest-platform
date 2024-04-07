package main

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

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
	binanceClient.FetchSpotBalances()
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

func TestFetchingAccountCrossMarginDebt(t *testing.T) {
	tradingConfig := GetTradingConfig()
	binanceClient := NewBinanceClient(tradingConfig)
	binanceClient.GetAccountDebtInUSDT()
}

func TestFetchingAccountDebtRatio(t *testing.T) {
	tradingConfig := GetTradingConfig()
	binanceClient := NewBinanceClient(tradingConfig)

	debtRatio, err := binanceClient.GetAccountDebtRatio()
	if err != nil {
		fmt.Println(err.Error())
	}
	fmt.Println(debtRatio)
}

func TestFetchingAccountByName(t *testing.T) {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	client := NewHttpClient(predServConfig.URI, headers)

	accountName := GetAccountName()
	account, err := client.FetchAccount(accountName)
	if err != nil {
		fmt.Println(err.Error())
	}
	fmt.Println(account)
}

func TestEnterStrategyTrade(t *testing.T) {
	tradingConfig := GetTradingConfig()
	binanceClient := NewBinanceClient(tradingConfig)

	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)

	accountName := GetAccountName()
	account, err := predServClient.FetchAccount(accountName)
	if err != nil {
		fmt.Println(err.Error())
	}

	strategies := predServClient.FetchStrategies()

	for _, strat := range strategies {
		if !strat.IsInPosition && ShouldEnterTrade(strat) {
			EnterStrategyTrade(binanceClient, strat, account)
		}
	}
}

func TestGetAccountNetValue(t *testing.T) {
	tradingConfig := GetTradingConfig()
	binanceClient := NewBinanceClient(tradingConfig)
	binanceClient.GetAccountNetValueUSDT()
}

func TestCreatingTradeEntry(t *testing.T) {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)
	predServClient.CreateTradeEntry(map[string]interface{}{
		"direction":    "SHORT",
		"open_price":   65000,
		"quantity":     0.00026,
		"open_time_ms": 1712340861903,
		"strategy_id":  1,
	})

	predServClient.CreateTradeEntry(map[string]interface{}{
		"direction":    DIRECTION_LONG,
		"open_price":   69312.01,
		"open_time_ms": 1712485308309,
		"quantity":     0.00025,
		"strategy_id":  1,
	})
}

func TestUpdateStrategy(t *testing.T) {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)

	predServClient.UpdateStrategy(map[string]interface{}{
		"active_trade":               1,
		"id":                         1,
		"klines_left_till_autoclose": 24,
		"price_on_trade_open":        65000,
		"time_on_trade_open_ms":      1712340861903,
		"is_in_position":             true,
	})
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
