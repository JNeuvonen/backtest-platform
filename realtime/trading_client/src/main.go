package main

func main() {
	predServConfig := GetPredServerConfig()
	tradingConfig := GetTradingConfig()
	accountName := GetAccountName()

	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)
	binanceClient := NewBinanceClient(tradingConfig)

	for {
		strategies := predServClient.FetchStrategies()
		account, _ := predServClient.FetchAccount(accountName)

		if IsEmptyStruct(account) || account.PreventAllTrading ||
			GetNumFailedCallsToPredServer() >= FAILED_CALLS_TO_UPDATE_STRAT_STATE_LIMIT {
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
			if strat.IsOnPredServErr {
				continue
			}

			if strat.IsInPosition && ShouldCloseTrade(binanceClient, strat) {
				CloseStrategyTrade(binanceClient, strat)
			} else if !strat.IsInPosition && ShouldEnterTrade(strat) {
				EnterStrategyTrade(binanceClient, strat, account)
			}
		}
		predServClient.CreateCloudLog("Trading loop completed", LOG_INFO)
	}
}
