package main

import (
	"bytes"
	"encoding/json"
	"fmt"
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

func (client *HttpClient) Post(
	endpoint string,
	contentType string,
	data []byte,
) (*http.Response, error) {
	client.Headers["Content-Type"] = contentType
	return client.makeRequest("POST", endpoint, data)
}

func (client *HttpClient) FetchStrategies() ([]Strategy, error) {
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

func (client *HttpClient) CreateCloudLog(msg string, level string) error {
	body := CloudLogBody{
		Message: msg,
		Level:   level,
	}

	jsonData, err := json.Marshal(body)
	if err != nil {
		return err
	}

	resp, err := client.Post(PRED_SERV_V1_LOG, "application/json", jsonData)
	if err != nil {
		return err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("server returned non-success status code: %d", resp.StatusCode)
	}

	return nil
}
