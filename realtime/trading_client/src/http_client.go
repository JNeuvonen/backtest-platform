package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
)

type HttpClient struct {
	BaseURL string
	Headers map[string]string
}

func NewHttpClient(baseURL string, headers map[string]string) *HttpClient {
	return &HttpClient{BaseURL: baseURL, Headers: headers}
}

func (client *HttpClient) makeRequest(
	method, endpoint string,
	body []byte,
) (*http.Response, error) {
	url := client.BaseURL + endpoint
	req, err := http.NewRequest(method, url, bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}

	for key, value := range client.Headers {
		req.Header.Set(key, value)
	}

	return http.DefaultClient.Do(req)
}

func (client *HttpClient) Put(endpoint string, body []byte) ([]byte, int, error) {
	resp, err := client.makeRequest(METHOD_PUT, endpoint, body)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, err
	}

	return bodyBytes, resp.StatusCode, nil
}

func (client *HttpClient) Get(endpoint string) ([]byte, error) {
	resp, err := client.makeRequest(METHOD_GET, endpoint, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func (client *HttpClient) Post(
	endpoint string,
	contentType string,
	data []byte,
) (*http.Response, error) {
	client.Headers["Content-Type"] = contentType
	return client.makeRequest(METHOD_POST, endpoint, data)
}

func (client *HttpClient) FetchStrategies() []Strategy {
	response, err := client.Get(PRED_SERV_V1_STRAT)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return []Strategy{}
	}
	if response != nil && len(response) > 0 {
		var strategyResponse StrategyResponse
		err := json.Unmarshal(response, &strategyResponse)
		if err != nil {
			CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
			return []Strategy{}
		}
		return strategyResponse.Data
	} else {
		return []Strategy{}
	}
}

func (client *HttpClient) FetchLongShortStrategies() []LongShortGroup {
	response, err := client.Get(PRED_SERV_V1_LONGSHORT)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return []LongShortGroup{}
	}

	if response != nil && len(response) > 0 {
		var strategyResponse LongShortGroupResponse
		err := json.Unmarshal(response, &strategyResponse)
		if err != nil {
			CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
			return []LongShortGroup{}
		}
		return strategyResponse.Data
	} else {
		return []LongShortGroup{}
	}
}

func (client *HttpClient) FetchLongShortTickers(id int) []LongShortTicker {
	response, err := client.Get(GetLongShortTickersEndpoint(id))
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return []LongShortTicker{}
	}

	if response != nil && len(response) > 0 {
		var strategyResponse LongShortTickerResponse
		err := json.Unmarshal(response, &strategyResponse)
		if err != nil {
			CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
			return []LongShortTicker{}
		}
		return strategyResponse.Data
	} else {
		return []LongShortTicker{}
	}
}

func (client *HttpClient) FetchLongShortPairs(id int) []LongShortPair {
	response, err := client.Get(GetLongShortTickersEndpoint(id))
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return []LongShortPair{}
	}

	if response != nil && len(response) > 0 {
		var strategyResponse LongShortPairResponse
		err := json.Unmarshal(response, &strategyResponse)
		if err != nil {
			CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
			return []LongShortPair{}
		}
		return strategyResponse.Data
	} else {
		return []LongShortPair{}
	}
}

func FetchStrategies() []Strategy {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)
	return predServClient.FetchStrategies()
}

func (client *HttpClient) FetchAccount(accountName string) (Account, error) {
	endpoint := PRED_SERV_V1_ACC + "/" + accountName
	response, err := client.Get(endpoint)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return Account{}, err
	}

	if response != nil {
		var accountByNameResponse AccountByNameResponse

		err := json.Unmarshal(response, &accountByNameResponse)
		if err != nil {

			CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
			return Account{}, err
		}
		return accountByNameResponse.Data, nil
	}
	return Account{}, nil
}

func (client *HttpClient) UpdateStrategy(fieldsToUpdate map[string]interface{}) bool {
	jsonBody, err := json.Marshal(fieldsToUpdate)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return false
	}

	endpoint := PRED_SERV_V1_STRAT
	_, statusCode, err := client.Put(endpoint, jsonBody)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return false
	}

	return statusCode == 200
}

func UpdateStrategy(fieldsToUpdate map[string]interface{}) bool {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)
	return predServClient.UpdateStrategy(fieldsToUpdate)
}

func (client *HttpClient) UpdateStrategyOnTradeClose(
	strat Strategy,
	fieldsToUpdate map[string]interface{},
) bool {
	jsonBody, err := json.Marshal(fieldsToUpdate)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return false
	}

	_, statusCode, err := client.Put(GetCloseTradeEndpoint(strat.ID), jsonBody)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return false
	}
	return statusCode == 200
}

func UpdateStrategyOnTradeClose(strat Strategy, fieldsToUpdate map[string]interface{}) bool {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)
	return predServClient.UpdateStrategyOnTradeClose(strat, fieldsToUpdate)
}

func (client *HttpClient) CreateTradeEntry(fields map[string]interface{}) *int32 {
	jsonBody, err := json.Marshal(fields)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return nil
	}

	res, err := client.Post(PRED_SERV_V1_TRADE, APPLICATION_JSON, jsonBody)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return nil
	}
	defer res.Body.Close()

	bodyBytes, err := io.ReadAll(res.Body)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return nil
	}

	id, err := strconv.Atoi(string(bodyBytes))
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return nil
	}

	tradeID := int32(id)
	return &tradeID
}

func CreateTradeEntry(fields map[string]interface{}) *int32 {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)

	return predServClient.CreateTradeEntry(fields)
}

func CreateCloudLog(msg string, level string) {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)
	predServClient.CreateCloudLog(msg, level)
}

func (client *HttpClient) CreateCloudLog(msg string, level string) error {
	body := CloudLogBody{
		Message:       msg,
		Level:         level,
		SourceProgram: 1,
	}

	jsonData, err := json.Marshal(body)
	if err != nil {
		return err
	}

	resp, err := client.Post(PRED_SERV_V1_LOG, APPLICATION_JSON, jsonData)
	if err != nil {
		return err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("server returned non-success status code: %d", resp.StatusCode)
	}

	return nil
}
