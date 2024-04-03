package main

import (
	"fmt"
)

func TradingLoop() {
	predServConfig := GetPredServerConfig()
	tradingConfig := GetTradingConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)
	binanceClient := NewBinanceClient(tradingConfig)

	accountValueInUSDT, _ := binanceClient.GetCurrentAccountWorthInUSDT()
	fmt.Println(accountValueInUSDT)

	for {
		balances := binanceClient.FetchBalances()

		for _, balance := range balances.Balances {
			fmt.Println(balance.Free)
		}

		strategies := predServClient.FetchStrategies()

		optimal_sizes_map := make(map[string]float64)

		for _, strat := range strategies {
			// fmt.Println(strat.Symbol)

			if strat.ShouldEnterTrade {
				optimal_sizes_map[strat.Symbol] = 0.0
			}
		}

		fmt.Println(optimal_sizes_map)
		predServClient.CreateCloudLog("Trading loop completed", "info")
	}
}
