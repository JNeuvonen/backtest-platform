package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
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

func (client *HttpClient) Get(endpoint string) ([]byte, error) {
	resp, err := client.makeRequest("GET", endpoint, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func (client *HttpClient) Post(endpoint string, contentType string, data []byte) ([]byte, error) {
	client.Headers["Content-Type"] = contentType
	resp, err := client.makeRequest("POST", endpoint, data)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return io.ReadAll(resp.Body)
}

func FetchStrategies() ([]Strategy, error) {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	client := NewHttpClient(predServConfig.URI, headers)

	response, err := client.Get(PRED_SERV_V1_STRAT)
	if err != nil {
		return nil, err
	}

	if response != nil && len(response) > 0 {
		var strategyResponse StrategyResponse
		err := json.Unmarshal(response, &strategyResponse)
		if err != nil {
			return nil, err
		}

		return strategyResponse.Data, nil
	} else {
		return []Strategy{}, nil
	}
}
