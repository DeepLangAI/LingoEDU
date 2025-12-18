# for model in Gemini2.0-Flash Gemini2.5-Flash DeepSeek-V3 GPT-4o O4-mini O3 Claude3.5-Sonnet DeepSeek-R1 Claude3.7-think Qwen3-235B-thinking Qwen3-235B-no-thinking  Qwen2.5-Max DeepSeekV3.1  GPT5 Gemini-2.5-pro
for model in  Gemini-3-pro
do
    echo "now running ${model}"
    python deepsearch.py --model $model
    wait
done

