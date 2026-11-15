# Brightline API

## Rate limits
The REST API allows 60 requests per minute on Starter, 600 requests per minute on Team and 3000 requests per minute on Business.

## API keys
API keys are created in Settings, then Developers. Keys expire after 90 days unless they are rotated.

## Uploads
A single upload request can carry up to 5 GB. Larger files must use multipart upload with 64 MB parts; multipart upload has no total size limit.

## Webhooks
Failed webhook deliveries are retried 5 times with exponential backoff.
