package main

func TradingLoop() {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)

	for {
		_ = predServClient.FetchStrategies()
		predServClient.CreateCloudLog("Trading loop completed", "info")
	}
}
