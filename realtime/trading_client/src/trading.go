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

	currTimeMs := int64(GetTimeInMs())
	riskManagementParams := GetRiskManagementParams()

	if currTimeMs-currTimeMs-riskManagementParams.TradingCooldownStartedTs < UNABLE_TO_REACH_BE_TRADE_COOLDOWN_MS {
		return false
	}

	return shouldStopLossClose(strat, price) || shouldTimebasedClose(strat) ||
		shouldProfitBasedClose(strat, price) || strat.ShouldCloseTrade
}

func ShouldEnterTrade(strat Strategy) bool {
	currTimeMs := int64(GetTimeInMs())
	riskManagementParams := GetRiskManagementParams()

	if currTimeMs-strat.TimeOnTradeOpenMs >= int64(strat.MinimumTimeBetweenTradesMs) &&
		currTimeMs-riskManagementParams.TradingCooldownStartedTs > UNABLE_TO_REACH_BE_TRADE_COOLDOWN_MS {
		return strat.ShouldEnterTrade
	}
	return false
}

func closeShortTrade(bc *BinanceClient, strat Strategy) {
}

func closeLongTrade(bc *BinanceClient, strat Strategy) {
}

func CloseStrategyTrade(bc *BinanceClient, strat Strategy) {
	if strat.IsShortSellingStrategy {
		closeShortTrade(bc, strat)
	} else {
		closeLongTrade(bc, strat)
	}
}

func calculateShortStratBetSizeUSDT(
	bc *BinanceClient,
	account Account,
	strat Strategy,
	accUSDTValue float64,
) float64 {
	accDebtRatio := bc.GetAssetDebtRatioUSDT()

	if accDebtRatio > account.MaxDebtRatio {
		CreateCloudLog(
			NewFmtError(errors.New("accDebtRatio > account.MaxDebtRatio"), CaptureStack()).Error(),
			"info",
		)
		return 0.0
	}

	maxAllocatedUSDTValue := (strat.AllocatedSizePerc / 100) * accUSDTValue
	debtInUSDT, _ := bc.GetAccountDebtInUSDT()

	if maxAllocatedUSDTValue+debtInUSDT/accUSDTValue < account.MaxDebtRatio {
		return maxAllocatedUSDTValue
	} else {
		return math.Min((account.MaxDebtRatio-accDebtRatio)*accUSDTValue, maxAllocatedUSDTValue)
	}
}

func calculateLongStratBetSizeUSDT(
	bc *BinanceClient,
	strat Strategy,
	accUSDTValue float64,
) float64 {
	marginBalances := bc.FetchMarginBalances()

	if marginBalances == nil {
		return 0. - 1
	}

	freeUSDT := GetFreeBalanceForMarginAsset(marginBalances, "USDT")

	if freeUSDT == 0.0 {
		CreateCloudLog(
			NewFmtError(
				errors.New("Strategy wanted to long but free USDT was 0"),
				CaptureStack(),
			).Error(),
			"info",
		)
	}

	maxAllocatedUSDTValue := (strat.AllocatedSizePerc / 100) * accUSDTValue
	return math.Min(freeUSDT, maxAllocatedUSDTValue)
}

func getShortSellingStrategyCloseSize(bc *BinanceClient, strat Strategy) float64 {
	marginBalancesRes := bc.FetchMarginBalances()

	freeQuoteAsset := GetFreeBalanceForMarginAsset(marginBalancesRes, strat.QuoteAsset)
	interestInAsset := GetInterestInAsset(marginBalancesRes, strat.BaseAsset)

	if freeQuoteAsset == 0.0 {
		return 0.0
	}

	price, err := bc.FetchLatestPrice(strat.Symbol)
	if err != nil {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			"exception",
		)
		return 0.0
	}

	availableMaxBuy := RoundToPrecision(
		SafeDivide(freeQuoteAsset, price),
		int32(strat.TradeQuantityPrecision),
	)

	return math.Min(availableMaxBuy, strat.QuantityOnTradeOpen+interestInAsset)
}

func getLongStrategyCloseSize(bc *BinanceClient, strat Strategy) float64 {
	res := bc.FetchMarginBalances()
	if res == nil {
		CreateCloudLog(
			NewFmtError(errors.New("Failed to fetch margin balances"), CaptureStack()).Error(),
			"exception",
		)
		return 0
	}

	freeBalance := GetFreeBalanceForMarginAsset(res, strat.BaseAsset)
	return RoundToPrecision(
		math.Min(freeBalance, strat.QuantityOnTradeOpen),
		int32(strat.TradeQuantityPrecision),
	)
}

func GetStrategyCloseTradeSize(bc *BinanceClient, strat Strategy) float64 {
	if strat.IsShortSellingStrategy {
		return getShortSellingStrategyCloseSize(bc, strat)
	} else {
		return getLongStrategyCloseSize(bc, strat)
	}
}

func GetStrategyAvailableBetsizeUSDT(bc *BinanceClient, strat Strategy, account Account) float64 {
	accUSDTValue, err := bc.GetAccountNetValueUSDT()
	if err != nil {
		CreateCloudLog(
			NewFmtError(err, CaptureStack()).Error(),
			"exception",
		)
		return 0.0
	}

	if accUSDTValue == 0.0 {
		CreateCloudLog(
			NewFmtError(err, CaptureStack()).Error(),
			"exception",
		)
		return 0.0
	}

	if strat.IsShortSellingStrategy {
		return calculateShortStratBetSizeUSDT(bc, account, strat, accUSDTValue)
	} else {
		return calculateLongStratBetSizeUSDT(bc, strat, accUSDTValue)
	}
}

func OpenLongTrade(strat Strategy, bc *BinanceClient, sizeUSDT float64) {
	price, err := bc.FetchLatestPrice(strat.Symbol)
	if err != nil {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			"exception",
		)
		return
	}

	quantity := GetBaseQuantity(sizeUSDT, price, int32(strat.TradeQuantityPrecision))

	res := bc.NewMarginOrder(strat.Symbol, quantity, "BUY", "MARKET", strat)

	UpdatePredServerAfterTradeOpen(strat, res, DIRECTION_LONG)
}

func OpenShortTrade(strat Strategy, bc *BinanceClient, sizeUSDT float64) {
	price, err := bc.FetchLatestPrice(strat.Symbol)
	if err != nil {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			"exception",
		)
		return
	}

	quantity := GetBaseQuantity(sizeUSDT, price, int32(strat.TradeQuantityPrecision))

	err = bc.TakeMarginLoan(strat.BaseAsset, quantity)

	fmt.Println(err)

	if err == nil {
		res := bc.NewMarginOrder(strat.Symbol, quantity, "SELL", "MARKET", strat)
		UpdatePredServerAfterTradeOpen(
			strat, res, DIRECTION_SHORT,
		)
	}
}

func EnterStrategyTrade(bc *BinanceClient, strat Strategy, account Account) {
	sizeUSDT := GetStrategyAvailableBetsizeUSDT(bc, strat, account)

	if strat.IsShortSellingStrategy {
		OpenShortTrade(strat, bc, sizeUSDT)
	} else {
		OpenLongTrade(strat, bc, sizeUSDT)
	}
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
		strategies := predServClient.FetchStrategies()
		account, _ := predServClient.FetchAccount(accountName)

		for _, strat := range strategies {

			if account.PreventAllTrading ||
				GetNumFailedCallsToPredServer() >= FAILED_CALLS_TO_UPDATE_STRAT_STATE_LIMIT {
				continue
			}

			if strat.IsInPosition && ShouldCloseTrade(binanceClient, strat) {
				CloseStrategyTrade(binanceClient, strat)
			} else if !strat.IsInPosition && ShouldEnterTrade(strat) {
				EnterStrategyTrade(binanceClient, strat, account)
			}
		}

		predServClient.CreateCloudLog("Trading loop completed", "info")
	}
}
