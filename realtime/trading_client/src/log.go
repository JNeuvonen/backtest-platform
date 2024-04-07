package main

func LogAndRetFallback[T any](err error, defaultValue T) T {
	CreateCloudLog(
		NewFmtError(err, CaptureStack()).Error(),
		LOG_EXCEPTION,
	)
	return defaultValue
}
