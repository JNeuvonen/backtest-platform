package main

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"time"

	binance_connector "live/trading-client/src/thirdparty/binance-connector-go"
)

func shouldStopLossClose(strat Strategy, price float64) bool {
	if !strat.UseStopLossBasedClose || strat.ShouldCalcStopsOnPredServ {
		return false
	}

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
		currTimeMs := int64(GetTimeInMs())
		klineSizeInMS := int64(strat.KlineSizeMs)
		maxHoldTimeInKlines := int64(strat.MaximumKlinesHoldTime)
		tradeOpenTime := strat.TimeOnTradeOpenMs

		return !strat.ShouldEnterTrade &&
			currTimeMs >= tradeOpenTime+klineSizeInMS*maxHoldTimeInKlines
	}
	return false
}

func shouldProfitBasedClose(strat Strategy, price float64) bool {
	if !strat.UseProfitBasedClose || strat.ShouldCalcStopsOnPredServ {
		return false
	}

	profitThreshold := 1 - (strat.TakeProfitThresholdPerc / 100)

	if strat.IsShortSellingStrategy {
		return price < strat.PriceOnTradeOpen*profitThreshold
	} else {
		return price > strat.PriceOnTradeOpen*(1+(strat.TakeProfitThresholdPerc/100))
	}
}

func ShouldCloseTrade(bc *BinanceClient, strat Strategy) bool {
	if strat.ShouldEnterTrade {
		return false
	}

	if (strat.UseProfitBasedClose || strat.UseStopLossBasedClose) &&
		!strat.ShouldCalcStopsOnPredServ {

		price, err := bc.FetchLatestPrice(strat.Symbol)
		if err != nil {
			return false
		}

		if shouldProfitBasedClose(strat, price) {
			return true
		}

		if shouldStopLossClose(strat, price) {
			return true
		}

	}

	return shouldTimebasedClose(strat) || strat.ShouldCloseTrade
}

func ShouldEnterTrade(strat Strategy) bool {
	currTimeMs := int64(GetTimeInMs())
	if currTimeMs-strat.TimeOnTradeOpenMs >= int64(strat.MinimumTimeBetweenTradesMs) {
		return strat.ShouldEnterTrade
	}
	return false
}

func closeLoanWithAvailableBalance(bc *BinanceClient, asset string, precision int32) {
	go func() {
		time.Sleep(10 * time.Second)
		res := bc.FetchMarginBalances()
		freeBalance := GetFreeBalanceForMarginAsset(res, asset)
		bc.RepayMarginLoan(asset, RoundToPrecisionCloseLoan(freeBalance, precision))
	}()
}

func handleRepaymentOfShortMarginLoan(
	bc *BinanceClient,
	strat Strategy,
	execOrderRes *binance_connector.MarginAccountNewOrderResponseFULL,
) {
	if execOrderRes != nil {
		bc.RepayMarginLoan(strat.BaseAsset, ParseToFloat64(execOrderRes.ExecutedQty, 0.0))
	}
}

func getQuoteLoanToCloseShortTrade(
	bc *BinanceClient,
	strat Strategy,
	price float64,
	currMaxCloseQuantity float64,
	stratLiabilities float64,
) {
	quoteLoanSize := RoundToPrecision(
		(stratLiabilities-currMaxCloseQuantity)*price,
		0,
	) + USDT_QUOTE_BUFFER

	err := bc.TakeMarginLoan(strat.QuoteAsset, quoteLoanSize, nil)

	if err == nil {
		res := bc.NewMarginOrder(
			strat.Symbol,
			RoundToPrecision(
				stratLiabilities*CLOSE_SHORT_FEES_COEFF,
				int32(strat.TradeQuantityPrecision),
			),
			ORDER_BUY,
			MARKET_ORDER,
		)

		handleRepaymentOfShortMarginLoan(bc, strat, res)
		UpdatePredServerOnTradeClose(strat, res)
		return
	} else {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			LOG_EXCEPTION,
		)
	}
}

func closeShortTrade(bc *BinanceClient, strat Strategy) {
	marginBalancesRes := bc.FetchMarginBalances()

	stratLiabilities := (strat.RemainingPositionOnTrade + GetInterestInAsset(
		marginBalancesRes,
		strat.BaseAsset,
	)) * CLOSE_SHORT_FEES_COEFF
	freeUSDT := GetFreeBalanceForMarginAsset(marginBalancesRes, strat.QuoteAsset)

	price, err := bc.FetchLatestPrice(strat.Symbol)
	if err != nil {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			LOG_EXCEPTION,
		)
		return
	}

	currMaxCloseQuantity := SafeDivide(freeUSDT, price)

	if currMaxCloseQuantity >= stratLiabilities {
		res := bc.NewMarginOrder(
			strat.Symbol,
			RoundToPrecision(stratLiabilities, int32(strat.TradeQuantityPrecision)),
			ORDER_BUY,
			MARKET_ORDER,
		)
		closeLoanWithAvailableBalance(bc, strat.BaseAsset, int32(strat.TradeQuantityPrecision))
		UpdatePredServerOnTradeClose(strat, res)
		return
	}

	// current quote asset was not enough to close short trade, so taking loan in quote asset
	getQuoteLoanToCloseShortTrade(bc, strat, price, currMaxCloseQuantity, stratLiabilities)
}

func closeLongTrade(
	bc *BinanceClient,
	strat Strategy,
) *binance_connector.MarginAccountNewOrderResponseFULL {
	marginBalancesRes := bc.FetchMarginBalances()

	freeBaseAsset := GetFreeBalanceForMarginAsset(marginBalancesRes, strat.BaseAsset)
	closeQuantity := RoundToPrecision(
		math.Min(strat.RemainingPositionOnTrade, freeBaseAsset),
		int32(strat.TradeQuantityPrecision),
	)

	res := bc.NewMarginOrder(
		strat.Symbol,
		closeQuantity,
		ORDER_SELL,
		MARKET_ORDER,
	)
	UpdatePredServerOnTradeClose(strat, res)

	return res
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
			LOG_INFO,
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

	freeUSDT := GetFreeBalanceForMarginAsset(marginBalances, ASSET_USDT)

	if freeUSDT == 0.0 {
		CreateCloudLog(
			NewFmtError(
				errors.New("Strategy tried to long but free USDT was 0"),
				CaptureStack(),
			).Error(),
			LOG_INFO,
		)
	}

	maxAllocatedUSDTValue := (strat.AllocatedSizePerc / 100) * accUSDTValue
	return math.Min(freeUSDT, maxAllocatedUSDTValue)
}

func getShortSellingAvailableCloseSize(bc *BinanceClient, strat Strategy) float64 {
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
			LOG_EXCEPTION,
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
			LOG_EXCEPTION,
		)
		return 0
	}

	freeBalance := GetFreeBalanceForMarginAsset(res, strat.BaseAsset)
	return RoundToPrecision(
		math.Min(freeBalance, strat.QuantityOnTradeOpen),
		int32(strat.TradeQuantityPrecision),
	)
}

func GetStrategyAvailableBetsizeUSDT(bc *BinanceClient, strat Strategy, account Account) float64 {
	accUSDTValue, err := bc.GetAccountNetValueUSDT()
	if err != nil {
		CreateCloudLog(
			NewFmtError(err, CaptureStack()).Error(),
			LOG_EXCEPTION,
		)
		return 0.0
	}

	if accUSDTValue == 0.0 {
		CreateCloudLog(
			NewFmtError(err, CaptureStack()).Error(),
			LOG_EXCEPTION,
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
			LOG_EXCEPTION,
		)
		return
	}

	quantity := GetBaseQuantity(sizeUSDT, price, int32(strat.TradeQuantityPrecision))

	res := bc.NewMarginOrder(strat.Symbol, quantity, ORDER_BUY, MARKET_ORDER)

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
			LOG_EXCEPTION,
		)
		return
	}

	quantity := GetBaseQuantity(sizeUSDT, price, int32(strat.TradeQuantityPrecision))

	err = bc.TakeMarginLoan(strat.BaseAsset, quantity, nil)

	if err == nil {
		res := bc.NewMarginOrder(strat.Symbol, quantity, ORDER_SELL, MARKET_ORDER)
		UpdatePredServerAfterTradeOpen(
			strat, res, DIRECTION_SHORT,
		)
	}
}

func EnterStrategyTrade(bc *BinanceClient, strat Strategy, account Account) {
	sizeUSDT := GetStrategyAvailableBetsizeUSDT(bc, strat, account)

	if sizeUSDT <= USDT_MIN_SIZE_FOR_POS {
		return
	}

	if strat.IsShortSellingStrategy {
		OpenShortTrade(strat, bc, sizeUSDT)
	} else {
		OpenLongTrade(strat, bc, sizeUSDT)
	}
}

func GetRatioOfLongsToNav(bc *BinanceClient) float64 {
	accNetValue, _ := bc.GetAccountNetValueUSDT()
	totalLongsUSDT := GetTotalLongsUSDT(bc)
	ratioOfLongsToNav := SafeDivide(totalLongsUSDT, accNetValue)
	return ratioOfLongsToNav
}

func closeOffAllExcessiveLongLeverage(
	bc *BinanceClient,
	assetUSDTPrice float64,
	strat Strategy,
	requiredToSellValueUSDT float64,
) {
	sellOffQuantity := RoundToPrecision(
		requiredToSellValueUSDT/assetUSDTPrice,
		int32(strat.TradeQuantityPrecision),
	)
	res := bc.NewMarginOrder(
		strat.Symbol,
		sellOffQuantity,
		ORDER_SELL,
		MARKET_ORDER,
	)
	UpdatePredServerOnTradeClose(strat, res)
	marginAssetsRes := bc.FetchMarginBalances()

	totalLiabilitiesUSDT := GetTotalLiabilitiesInAsset(marginAssetsRes, ASSET_USDT)
	freeUSDT := GetFreeBalanceForMarginAsset(marginAssetsRes, ASSET_USDT)

	if freeUSDT >= totalLiabilitiesUSDT {
		bc.RepayMarginLoan(
			ASSET_USDT,
			RoundToPrecision(totalLiabilitiesUSDT, int32(strat.TradeQuantityPrecision)),
		)
	} else {
		bc.RepayMarginLoan(
			ASSET_USDT,
			RoundToPrecision(freeUSDT, int32(strat.TradeQuantityPrecision)),
		)
	}
}

func RemoveRiskFromLongStrats(
	bc *BinanceClient,
	strategies []Strategy,
	ratioOfLongsToNav float64,
	account Account,
	recursionCount int32,
) {
	if ratioOfLongsToNav < account.MaxRatioOfLongsToNav || recursionCount >= 10 {
		return
	}

	strategiesCopy := make([]Strategy, len(strategies))
	copy(strategiesCopy, strategies)

	sort.Slice(strategiesCopy, func(i, j int) bool {
		return strategiesCopy[i].Priority > strategiesCopy[j].Priority
	})

	usdtPrices, err := getAllUSDTPrices()
	if err != nil {
		return
	}

	accNetValueUSDT, err := bc.GetAccountNetValueUSDT()
	if err != nil {
		return
	}

	requiredToSellValueUSDT := (ratioOfLongsToNav - account.MaxRatioOfLongsToNav) * accNetValueUSDT

	if requiredToSellValueUSDT <= MINIMUM_USDT_QUANT_FOR_TRADE {
		return
	}

	for _, strat := range strategiesCopy {
		if strat.IsInPosition && !strat.IsShortSellingStrategy {
			symbolInfo := FindListItem[SymbolInfoSimple](
				usdtPrices,
				func(i SymbolInfoSimple) bool { return i.Symbol == strat.BaseAsset+ASSET_USDT },
			)
			assetUSDTPrice := ParseToFloat64(symbolInfo.Price, 0.0)
			remainingPosUSDTVal := assetUSDTPrice * strat.RemainingPositionOnTrade

			if remainingPosUSDTVal >= requiredToSellValueUSDT {
				closeOffAllExcessiveLongLeverage(bc, assetUSDTPrice, strat, requiredToSellValueUSDT)
				break
			} else {
				closeLongTrade(bc, strat)
				marginAssetsRes := bc.FetchMarginBalances()
				freeUSDT := GetFreeBalanceForMarginAsset(marginAssetsRes, ASSET_USDT)
				bc.RepayMarginLoan(
					ASSET_USDT,
					RoundToPrecision(freeUSDT, int32(strat.TradeQuantityPrecision)),
				)
				newLongRatio := GetRatioOfLongsToNav(bc)
				CreateCloudLog(
					NewFmtError(
						errors.New(fmt.Sprintf("Recursively calling %s to reduce long leverage", GetCurrentFunctionName())),
						CaptureStack(),
					).Error(), LOG_INFO)
				RemoveRiskFromLongStrats(bc, FetchStrategies(), newLongRatio, account, recursionCount+1)
			}
		}
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
		time.Sleep(time.Minute)
	}
}
