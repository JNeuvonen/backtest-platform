package main

import (
	"fmt"
	"os"
	"runtime/debug"
)

func main() {
	defer func() {
		if r := recover(); r != nil {
			stackTrace := string(debug.Stack())
			CreateCloudLog(
				fmt.Sprintf("Application panicked: %v\nStack trace: %s", r, stackTrace),
				LOG_EXCEPTION,
			)
			os.Exit(1)
		}
	}()

	predServConfig := GetPredServerConfig()
	tradingConfig := GetTradingConfig()
	accountName := GetAccountName()

	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)
	binanceClient := NewBinanceClient(tradingConfig)

	lastLogMsg := int64(0)

	for {
		strategies := predServClient.FetchStrategies()
		account, _ := predServClient.FetchAccount(accountName)

		if IsEmptyStruct(account) || account.PreventAllTrading {
			continue
		}

		ratioOfLongsToNav := GetRatioOfLongsToNav(binanceClient)

		if ratioOfLongsToNav > account.MaxRatioOfLongsToNav {
			RemoveRiskFromLongStrats(
				binanceClient,
				strategies,
				ratioOfLongsToNav,
				account,
				0,
			)
		}

		for _, strat := range strategies {
			if strat.IsOnPredServErr || strat.IsDisabled {
				continue
			}

			if strat.IsInPosition && ShouldCloseTrade(binanceClient, strat) {
				CloseStrategyTrade(binanceClient, strat)
			} else if !strat.IsInPosition && ShouldEnterTrade(strat) {
				EnterStrategyTrade(binanceClient, strat, account)
			}
		}

		longShortGroups := predServClient.FetchLongShortStrategies()

		for _, longshortGroup := range longShortGroups {
			ProcessLongShortGroup(binanceClient, predServClient, &longshortGroup)
		}

		if GetTimeInMs() >= lastLogMsg+MINUTE_IN_MS {
			predServClient.CreateCloudLog("Trading loop completed", LOG_INFO)
			lastLogMsg = GetTimeInMs()
		}
	}
}
