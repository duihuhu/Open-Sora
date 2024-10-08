torchrun --nproc_per_node 8 scripts/inference.py configs/opensora-v1-2/inference/sample_config.py   --num-frames 4s --resolution 360p   --layernorm-kernel False --flash-attn False --batch-size 1 --prompt ["a beautiful waterfall"]
sleep 5

torchrun --nproc_per_node 8 scripts/inference.py  configs/opensora-v1-2/inference/sample_config.py   --num-frames 4s --resolution 360p   --layernorm-kernel False --flash-attn False --batch-size 2 --prompt ["a beautiful waterfall", "a beautiful waterfall"]
sleep 5

torchrun --nproc_per_node 8 scripts/inference.py configs/opensora-v1-2/inference/sample_config.py   --num-frames 4s --resolution 360p   --layernorm-kernel False --flash-attn False --batch-size 4 --prompt ["a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall"]
sleep 5


torchrun --nproc_per_node 8 scripts/inference.py configs/opensora-v1-2/inference/sample_config.py   --num-frames 4s --resolution 360p   --layernorm-kernel False --flash-attn False --batch-size 8 --prompt ["a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall"]
