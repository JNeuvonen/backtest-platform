package main

import (
	"errors"
	"fmt"
	"math"
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

func ShouldEnterTrade(strat Strategy) bool {
	currTimeMs := GetTimeInMs()

	if strat.TimeOnTradeOpenMs-int(currTimeMs) >= strat.MinimumTimeBetweenTradesMs {
		return strat.ShouldEnterTrade
	}
	return false
}

func CloseStrategyTrade(bc *BinanceClient, strat Strategy) {
}

func GetStrategyAvailableBetsizeUSDT(bc *BinanceClient, strat Strategy, account Account) float64 {
	accUSDTValue, err := bc.GetPortfolioValueInUSDT()
	if err != nil {
		return LogAndRetFallback(err, 0.0)
	}

	if accUSDTValue == 0.0 {
		return LogAndRetFallback(err, 0.0)
	}

	balances := bc.FetchBalances()

	if balances != nil {
		if strat.IsShortSellingStrategy {
			accDebtRatio, err := bc.GetAccountDebtRatio()
			if err != nil {
				return LogAndRetFallback(err, 0.0)
			}

			if accDebtRatio > account.MaxDebtRatio {
				return LogAndRetFallback(errors.New(
					"GetStrategyAvailableBetsizeUSDT(): maximum leverage already used"), 0.0,
				)
			}

			maxAllocatedUSDTValue := strat.AllocatedSizePerc * accUSDTValue
			debtInUSDT, _ := bc.GetAccountDebtInUSDT()

			if maxAllocatedUSDTValue+debtInUSDT/accUSDTValue < account.MaxDebtRatio {
				return maxAllocatedUSDTValue
			} else {
				return (account.MaxDebtRatio - accDebtRatio) * accUSDTValue
			}

		} else {
			freeUSDT, err := GetFreeBalanceForAsset(balances.Balances, "USDT")
			if err != nil {
				return LogAndRetFallback(err, 0.0)
			}

			maxAllocatedUSDTValue := (strat.AllocatedSizePerc / 100) * accUSDTValue
			return math.Min(ParseToFloat64(freeUSDT, 0.0), maxAllocatedUSDTValue)
		}
	}
	return 0.0
}

func EnterStrategyTrade(bc *BinanceClient, strat Strategy, account Account) {
	size := GetStrategyAvailableBetsizeUSDT(bc, strat, account)
	fmt.Println("size", size)
}

func TradingLoop() {
	predServConfig := GetPredServerConfig()
	tradingConfig := GetTradingConfig()
	accountName := GetAccountName()

	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)
	binanceClient := NewBinanceClient(tradingConfig)

	for {
		balances := binanceClient.FetchBalances()

		for _, balance := range balances.Balances {
			fmt.Println(balance.Free)
		}

		strategies := predServClient.FetchStrategies()
		account, _ := predServClient.FetchAccount(accountName)

		for _, strat := range strategies {
			if strat.IsInPosition {
				if ShouldCloseTrade(binanceClient, strat) {
					CloseStrategyTrade(binanceClient, strat)
				}
			} else if !strat.IsInPosition {
				if ShouldEnterTrade(strat) {
					EnterStrategyTrade(binanceClient, strat, account)
				}
			}
		}

		predServClient.CreateCloudLog("Trading loop completed", "info")
	}
}
