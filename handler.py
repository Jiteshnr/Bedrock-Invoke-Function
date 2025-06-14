

import json
import boto3
import os

# Create Bedrock runtime client
bedrock = boto3.client("bedrock-runtime")

# Set this to your model (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

def lambda_handler(event, context):
    try:
        # Determine if API Gateway sent the event (string body) or test invocation
        if "body" in event:
            body = json.loads(event["body"])
        else:
            body = event

        prompt = body.get("prompt", "")
        if not prompt:
            return response(400, {"error": "Missing 'prompt' in request"})

        # Prepare model input
        model_input = {
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": 300,
            "temperature": 0.7,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman:"]
        }

        # Call Bedrock model
        response_model = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(model_input)
        )

        model_output = json.loads(response_model["body"].read())
        result_text = model_output.get("completion", "No output returned from model")

        return response(200, {"response": result_text})

    except Exception as e:
        return response(500, {"error": str(e)})

def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body)
    }

