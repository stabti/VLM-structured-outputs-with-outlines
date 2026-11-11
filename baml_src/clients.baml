// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

// Using the new OpenAI Responses API for enhanced formatting
client<llm> CustomGPT5 {
  provider openai-responses
  options {
    model "gpt-5"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> CustomGPT5Mini {
  provider openai-responses
  retry_policy Exponential
  options {
    model "gpt-5-mini"
    api_key env.OPENAI_API_KEY
  }
}

// Openai with chat completion
client<llm> CustomGPT5Chat {
  provider openai
  options {
    model "gpt-5"
    api_key env.OPENAI_API_KEY
  }
}

// Latest Anthropic Claude 4 models
client<llm> CustomOpus4 {
  provider anthropic
  options {
    model "claude-opus-4-1-20250805"
    api_key env.ANTHROPIC_API_KEY
  }
}

client<llm> CustomSonnet4 {
  provider anthropic
  options {
    model "claude-sonnet-4-20250514"
    api_key env.ANTHROPIC_API_KEY
  }
}

client<llm> CustomHaiku {
  provider anthropic
  retry_policy Constant
  options {
    model "claude-3-5-haiku-20241022"
    api_key env.ANTHROPIC_API_KEY
  }
}

// Example Google AI client (uncomment to use)
// client<llm> CustomGemini {
//   provider google-ai
//   options {
//     model "gemini-2.5-pro"
//     api_key env.GOOGLE_API_KEY
//   }
// }

// Example AWS Bedrock client (uncomment to use)
// client<llm> CustomBedrock {
//   provider aws-bedrock
//   options {
//     model "anthropic.claude-sonnet-4-20250514-v1:0"
//     region "us-east-1"
//     // AWS credentials are auto-detected from env vars
//   }
// }

// Example Azure OpenAI client (uncomment to use)
// client<llm> CustomAzure {
//   provider azure-openai
//   options {
//     model "gpt-5"
//     api_key env.AZURE_OPENAI_API_KEY
//     base_url "https://MY_RESOURCE_NAME.openai.azure.com/openai/deployments/MY_DEPLOYMENT_ID"
//     api_version "2024-10-01-preview"
//   }
// }

// Example Vertex AI client (uncomment to use)
// client<llm> CustomVertex {
//   provider vertex-ai
//   options {
//     model "gemini-2.5-pro"
//     location "us-central1"
//     // Uses Google Cloud Application Default Credentials
//   }
// }

// Local open-weights VLM (LFM2.5-VL) served by ollama, via its OpenAI-compatible endpoint.
client<llm> CustomOllama {
  provider openai-generic
  options {
    base_url "http://localhost:11434/v1"
    model "hf.co/LiquidAI/LFM2.5-VL-1.6B-GGUF:BF16"
    default_role "user" // Most local models prefer the user role
    temperature 0       // greedy: mirrors the outlines run (one generation per image)
    max_tokens 512      // headroom so the JSON isn't truncated mid-object
    // No API key needed for local Ollama
  }
}

// Same model served by vLLM (bf16 HF weights, OpenAI-compatible server), started from a
// separate env (~/envs/vllm): `vllm serve LiquidAI/LFM2.5-VL-1.6B --max-model-len 8192
// --max-num-batched-tokens 4096 --gpu-memory-utilization 0.6` with VLLM_USE_FLASHINFER_SAMPLER=0. Selected at call time by compare_baml.py (ENGINE="vllm").
client<llm> CustomVllm {
  provider openai-generic
  options {
    base_url "http://localhost:8000/v1"
    model "LiquidAI/LFM2.5-VL-1.6B"
    default_role "user"
    temperature 0       // greedy, same as CustomOllama
    max_tokens 512
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/round-robin
client<llm> CustomFast {
  provider round-robin
  options {
    // This will alternate between the two clients
    strategy [CustomGPT5Mini, CustomHaiku]
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/fallback
client<llm> OpenaiFallback {
  provider fallback
  options {
    // This will try the clients in order until one succeeds
    strategy [CustomGPT5Mini, CustomGPT5]
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/retry
retry_policy Constant {
  max_retries 3
  strategy {
    type constant_delay
    delay_ms 200
  }
}

retry_policy Exponential {
  max_retries 2
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}