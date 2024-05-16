package main

import (
	"encoding/json"
	"errors"
	"math"

	binance_connector "live/trading-client/src/thirdparty/binance-connector-go"
)

func getLongShortEnterUSDTSize(
	bc *BinanceClient,
	account *Account,
	maxAllocationRatio float64,
) float64 {
	accUSDTValue, _ := bc.GetAccountNetValueUSDT()

	if accUSDTValue == 0.0 {
		return 0.0
	}

	accDebtRatio := bc.GetAssetDebtRatioUSDT()

	if accDebtRatio+maxAllocationRatio > account.MaxDebtRatio {
		CreateCloudLog(
			NewFmtError(
				errors.New(
					"Acc debt ratio would surpass account maximum, aborting long/short enter",
				),
				CaptureStack(),
			).Error(),
			LOG_INFO,
		)
		return 0.0
	}

	allocatedRatio := math.Min(account.MaxDebtRatio-maxAllocationRatio, maxAllocationRatio)
	return allocatedRatio * accUSDTValue
}

func initPairTradeShort(
	bc *BinanceClient,
	usdtEnterSize float64,
	pair *LongShortPair,
) (*binance_connector.MarginAccountNewOrderResponseFULL, error, float64) {
	price, err := bc.FetchLatestPrice(pair.SellSymbol)
	if err != nil {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			LOG_EXCEPTION,
		)
		return nil, err, 0.0
	}

	quantity := GetBaseQuantity(usdtEnterSize, price, int32(pair.SellQtyPrecision))

	err = bc.TakeMarginLoan(pair.SellBaseAsset, quantity)

	if err == nil {
		res := bc.NewMarginOrder(pair.SellSymbol, quantity, ORDER_SELL, MARKET_ORDER)
		if res != nil {
			return res, nil, quantity
		}
		UpdatePairTradeEnterError(pair.ID)
		return nil, errors.New(
			"Failed to submit sell order for long/short strategy. Symbol: " + pair.SellSymbol,
		), 0.0
	}

	UpdatePair(pair.ID, map[string]interface{}{
		"is_no_loan_available_err": true,
	})
	CreateCloudLog(
		"Failed to take loan for long/short strategy. Symbol: "+pair.SellSymbol,
		LOG_EXCEPTION,
	)
	return nil, errors.New("Failed to take loan for long/short strategy"), 0.0
}

func initPairTradeLong(
	bc *BinanceClient,
	usdtEnterSize float64,
	pair *LongShortPair,
) (*binance_connector.MarginAccountNewOrderResponseFULL, error) {
	price, _ := bc.FetchLatestPrice(pair.BuySymbol)

	quantity := GetBaseQuantity(usdtEnterSize, price, int32(pair.BuyQtyPrecision))

	res := bc.NewMarginOrder(pair.BuySymbol, quantity, ORDER_BUY, MARKET_ORDER)

	if res != nil {
		return res, nil
	}

	UpdatePairTradeEnterError(pair.ID)

	CreateCloudLog(
		"Failed to enter long position for long/short strategy. Symbol: "+pair.SellSymbol,
		LOG_EXCEPTION,
	)
	return nil, errors.New("Failed to enter to long position for long/short strategy")
}

func closeLong(
	bc *BinanceClient,
	pair *LongShortPair,
) *binance_connector.MarginAccountNewOrderResponseFULL {
	marginBalancesRes := bc.FetchMarginBalances()

	freeBaseAsset := GetFreeBalanceForMarginAsset(marginBalancesRes, pair.BuyBaseAsset)
	closeQuantity := RoundToPrecision(
		math.Min(pair.BuyOpenQtyInBase, freeBaseAsset),
		int32(pair.BuyQtyPrecision),
	)
	res := bc.NewMarginOrder(
		pair.BuySymbol,
		closeQuantity,
		ORDER_SELL,
		MARKET_ORDER,
	)

	return res
}

func getQuoteLoanToClosePairTrade(
	bc *BinanceClient,
	pair *LongShortPair,
	price float64,
	currMaxCloseQuantity float64,
	stratLiabilities float64,
) *binance_connector.MarginAccountNewOrderResponseFULL {
	quoteLoanSize := RoundToPrecision(
		(stratLiabilities-currMaxCloseQuantity)*price,
		0,
	) + USDT_QUOTE_BUFFER

	err := bc.TakeMarginLoan(pair.SellQuoteAsset, quoteLoanSize)

	if err == nil {
		res := bc.NewMarginOrder(
			pair.SellSymbol,
			RoundToPrecision(stratLiabilities, int32(pair.SellQtyPrecision)),
			ORDER_BUY,
			MARKET_ORDER,
		)

		if res != nil {
			bc.RepayMarginLoan(pair.SellBaseAsset, ParseToFloat64(res.ExecutedQty, 0.0))
		}

		return res
	}

	return nil
}

func closeShort(
	bc *BinanceClient,
	pair *LongShortPair,
) *binance_connector.MarginAccountNewOrderResponseFULL {
	marginBalancesRes := bc.FetchMarginBalances()

	stratLiabilities := (pair.DebtOpenQtyInBase + GetInterestInAsset(
		marginBalancesRes,
		pair.SellBaseAsset,
	)) * CLOSE_SHORT_FEES_COEFF
	freeUSDT := GetFreeBalanceForMarginAsset(marginBalancesRes, pair.SellQuoteAsset)

	price, err := bc.FetchLatestPrice(pair.SellSymbol)
	if err != nil {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			LOG_EXCEPTION,
		)
		return nil
	}

	currMaxCloseQuantity := SafeDivide(freeUSDT, price)

	if currMaxCloseQuantity >= stratLiabilities {
		res := bc.NewMarginOrder(
			pair.SellSymbol,
			RoundToPrecision(stratLiabilities, int32(pair.SellQtyPrecision)),
			ORDER_BUY,
			MARKET_ORDER,
		)
		if res != nil {
			closeLoanWithAvailableBalance(bc, pair.SellBaseAsset, int32(pair.SellQtyPrecision))
			return res
		}
		CreateCloudLog(
			NewFmtError(
				errors.New(
					"Failed to submit buy order to close short for long/short strategy. Symbol: "+pair.SellSymbol,
				),
				CaptureStack(),
			).Error(),
			LOG_EXCEPTION,
		)
		return nil
	}

	return getQuoteLoanToClosePairTrade(bc, pair, price, currMaxCloseQuantity, stratLiabilities)
}

func updatePredServOnLongShortEnter(
	predServClient *HttpClient,
	pair *LongShortPair,
	sellSideRes *binance_connector.MarginAccountNewOrderResponseFULL,
	buySideRes *binance_connector.MarginAccountNewOrderResponseFULL,
	debtQuantity float64,
) {
	reqBody := map[string]interface{}{
		"long_side_order":  buySideRes,
		"short_side_order": sellSideRes,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return
	}
	predServClient.Post(GetLongShortEnterTradeEndpoint(pair.ID), APPLICATION_JSON, jsonBody)
}

func enterLongShortTrade(
	bc *BinanceClient,
	predServClient *HttpClient,
	group *LongShortGroup,
	pair *LongShortPair,
) {
	accountName := GetAccountName()
	account, _ := predServClient.FetchAccount(accountName)
	maxAllocationRatio := group.MaxLeverageRatio / float64(group.MaxSimultaneousPositions)
	usdtEnterSize := getLongShortEnterUSDTSize(bc, &account, maxAllocationRatio)

	sellSideRes, err, debtQuantity := initPairTradeShort(bc, usdtEnterSize, pair)

	if err == nil {
		sellOrderQuoteQty := ParseToFloat64(sellSideRes.CumulativeQuoteQty, 0.0)
		buySideRes, err := initPairTradeLong(bc, sellOrderQuoteQty, pair)

		if err == nil {
			updatePredServOnLongShortEnter(
				predServClient,
				pair,
				sellSideRes,
				buySideRes,
				debtQuantity,
			)
		}
	}
}

func exitLongShortTrade(
	bc *BinanceClient,
	predServClient *HttpClient,
	pair *LongShortPair,
) {
	closeLongRes := closeLong(bc, pair)
	closeShortRes := closeShort(bc, pair)

	reqBody := map[string]interface{}{
		"long_side_order":  closeLongRes,
		"short_side_order": closeShortRes,
	}
	jsonBody, err := json.Marshal(reqBody)

	if err == nil {
		predServClient.Post(
			GetLongShortExitTradeEndpoint(pair.ID), APPLICATION_JSON, jsonBody,
		)
	} else {
		CreateCloudLog("Failed to unmarshal request body while trying to close long/short trade", LOG_EXCEPTION)
	}
}

func findLsTicker(id int, lsTickers []LongShortTicker) *LongShortTicker {
	for _, ticker := range lsTickers {
		if ticker.ID == id {
			return &ticker
		}
	}
	return nil
}

func getTradeCurrNetResult(
	openPrice float64,
	currPrice float64,
	openQtyBase float64,
	isShort bool,
) float64 {
	openQuoteQty := openQtyBase * openPrice
	currQuoteQty := openQtyBase * currPrice

	if isShort {
		return openQuoteQty - currQuoteQty
	}
	return currQuoteQty - openQuoteQty
}

func lsGetPairCurrNetResult(
	bc *BinanceClient,
	pair *LongShortPair,
	sellSymbolPrice float64,
	buySymbolPrice float64,
) float64 {
	longNetResult := getTradeCurrNetResult(
		pair.BuyOpenPrice,
		buySymbolPrice,
		pair.BuyOpenQtyInBase,
		false,
	)

	shortNetResult := getTradeCurrNetResult(
		pair.SellOpenPrice,
		sellSymbolPrice,
		pair.DebtOpenQtyInBase,
		false,
	)

	return longNetResult + shortNetResult
}

func shouldLsProfitBasedClose(bc *BinanceClient, group *LongShortGroup, pair *LongShortPair) bool {
	if !group.UseProfitBasedClose {
		return false
	}

	sellSymbolPrice, fetchSellSymbolErr := bc.FetchLatestPrice(pair.SellSymbol)
	buySymbolPrice, fetchBuySymbolErr := bc.FetchLatestPrice(pair.BuySymbol)

	if fetchSellSymbolErr != nil || fetchBuySymbolErr != nil {
		return false
	}
	currNetResult := lsGetPairCurrNetResult(bc, pair, sellSymbolPrice, buySymbolPrice)
	percResult := SafeDivide(currNetResult, pair.BuyOpenQtyInQuote+pair.SellOpenQtyInQuote) * 100

	if percResult > 0 {
		return percResult >= group.TakeProfitThresholdPerc
	}
	return false
}

func shouldLsStopLossBasedClose(
	bc *BinanceClient,
	group *LongShortGroup,
	pair *LongShortPair,
) bool {
	if !group.UseStopLossBasedClose {
		return false
	}

	sellSymbolPrice, fetchSellSymbolErr := bc.FetchLatestPrice(pair.SellSymbol)
	buySymbolPrice, fetchBuySymbolErr := bc.FetchLatestPrice(pair.BuySymbol)

	if fetchSellSymbolErr != nil || fetchBuySymbolErr != nil {
		return false
	}

	currNetResult := lsGetPairCurrNetResult(bc, pair, sellSymbolPrice, buySymbolPrice)
	percResult := SafeDivide(currNetResult, pair.BuyOpenQtyInQuote+pair.SellOpenQtyInQuote) * 100

	if percResult < 0 {
		return math.Abs(percResult) >= group.StopLossThresholdPerc
	}

	return false
}

func shouldLsTimeBasedClose(
	group *LongShortGroup,
	pair *LongShortPair,
) bool {
	if !group.UseTimeBasedClose {
		return false
	}
	currTimeMs := GetTimeInMs()
	maxKlines := group.KlinesUntilClose
	if currTimeMs >= int64(pair.BuyOpenTime)+(int64(group.KlineSizeMs)*int64(maxKlines)) {
		return true
	}
	return false
}

func shouldClosePairTrade(
	bc *BinanceClient,
	group *LongShortGroup,
	pair *LongShortPair,
	lsTickers []LongShortTicker,
) bool {
	sellSideTicker := findLsTicker(pair.SellTickerID, lsTickers)
	buySideTicker := findLsTicker(pair.BuyTickerID, lsTickers)

	if sellSideTicker.IsValidSell && buySideTicker.IsValidBuy {
		return false
	}

	return shouldLsTimeBasedClose(group, pair) || shouldLsProfitBasedClose(bc, group, pair) ||
		shouldLsStopLossBasedClose(bc, group, pair)
}

func ProcessLongShortGroup(bc *BinanceClient, predServClient *HttpClient, group *LongShortGroup) {
	pairs := predServClient.FetchLongShortPairs(group.ID)
	lsTickers := predServClient.FetchLongShortTickers(group.ID)

	for _, pair := range pairs {
		if pair.InPosition && !pair.IsTradeFinished &&
			shouldClosePairTrade(bc, group, &pair, lsTickers) {
			exitLongShortTrade(bc, predServClient, &pair)
		}

		if !pair.InPosition && !pair.IsTradeFinished && !pair.ErrorInEntering {
			// enterLongShortTrade(bc, predServClient, group, &pair)
		}
	}
}
