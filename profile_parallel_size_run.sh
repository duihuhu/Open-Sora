python scripts/inference.py configs/opensora-v1-2/inference/sample_config.py   --num-frames 2s --resolution 144p    --layernorm-kernel False --flash-attn False --batch-size 1 --prompt ["a beautiful waterfall"]
sleep 5

torchrun --nproc_per_node 2 scripts/inference.py configs/opensora-v1-2/inference/sample_config.py   --num-frames 2s --resolution 144p   --layernorm-kernel False --flash-attn False --batch-size 1 --prompt ["a beautiful waterfall"]
sleep 5

torchrun --nproc_per_node 4 scripts/inference.py configs/opensora-v1-2/inference/sample_config.py   --num-frames 2s --resolution 144p   --layernorm-kernel False --flash-attn False --batch-size 1 --prompt ["a beautiful waterfall"]
sleep 5


torchrun --nproc_per_node 8 scripts/inference.py configs/opensora-v1-2/inference/sample_config.py   --num-frames 2s --resolution 144p   --layernorm-kernel False --flash-attn False --batch-size 1 --prompt ["a beautiful waterfall"]
