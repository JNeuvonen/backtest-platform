package main

import (
	"fmt"
)

func shouldStopLossClose(strat Strategy, price float64) bool {
	if strat.IsShortSellingStrategy {
		threshold := 1 + (strat.StopLossThresholdPerc / 100)
		return price > strat.PriceOnTradeOpen*threshold
	} else {
		threshold := 1 - (strat.StopLossThresholdPerc / 100)
		return price < strat.PriceOnTradeOpen*threshold
	}
}

func shouldTimebasedClose(strat Strategy) bool {
	if strat.UseTimeBasedClose {
		return strat.KlinesLeftTillAutoclose == 0
	}
	return false
}

func shouldProfitBasedClose(strat Strategy, price float64) bool {
	profitThreshold := 1 - (strat.TakeProfitThresholdPerc / 100)

	if strat.IsShortSellingStrategy {
		return price < strat.PriceOnTradeOpen*profitThreshold
	} else {
		return price > strat.PriceOnTradeOpen*(1+strat.TakeProfitThresholdPerc/100)
	}
}

func ShouldCloseTrade(bc *BinanceClient, strat Strategy) bool {
	price, err := bc.FetchLatestPrice(strat.Symbol)
	if err != nil {
		return false
	}

	return shouldStopLossClose(strat, price) || shouldTimebasedClose(strat) ||
		shouldProfitBasedClose(strat, price) || strat.ShouldCloseTrade
}

func ShouldEnterTrade(bc *BinanceClient, strat Strategy) bool {
	currTimeMs := GetTimeInMs()

	if strat.TimeOnTradeOpenMs-int(currTimeMs) >= strat.MinimumTimeBetweenTradesMs {
		return true
	}
	return false
}

func CloseStrategyTrade(bc *BinanceClient, strat Strategy) {
}

func EnterStrategyTrade(bc *BinanceClient, strat Strategy) {
}

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

		optimalSizesMap := make(map[string]float64)

		for _, strat := range strategies {
			if strat.IsInPosition {
				if ShouldCloseTrade(binanceClient, strat) {
					CloseStrategyTrade(binanceClient, strat)
				}
			} else if !strat.IsInPosition {
				if ShouldEnterTrade(binanceClient, strat) {
					EnterStrategyTrade(binanceClient, strat)
				}
			}
		}

		fmt.Println(optimalSizesMap)
		predServClient.CreateCloudLog("Trading loop completed", "info")
	}
}
