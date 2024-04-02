package main

import (
	"time"
)

func TradingLoop() {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)

	for {
		_, err := predServClient.FetchStrategies()
		if err != nil {
			time.Sleep(5 * time.Second)
			continue
		}
	}
}
