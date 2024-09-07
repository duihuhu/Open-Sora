torchrun --nproc_per_node 1 scripts/inference.py configs/opensora-v1-2/inference/sample_config.py   --num-frames 2s --resolution 480p   --layernorm-kernel False --flash-attn False --batch-size 1 --prompt ["a beautiful waterfall"]
sleep 5

torchrun --nproc_per_node 1 scripts/inference.py  configs/opensora-v1-2/inference/sample_config.py   --num-frames 2s --resolution 480p   --layernorm-kernel False --flash-attn False --batch-size 2 --prompt ["a beautiful waterfall", "a beautiful waterfall"]
sleep 5

torchrun --nproc_per_node 1 scripts/inference.py configs/opensora-v1-2/inference/sample_config.py   --num-frames 2s --resolution 480p   --layernorm-kernel False --flash-attn False --batch-size 4 --prompt ["a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall"]
sleep 5


torchrun --nproc_per_node 1 scripts/inference.py configs/opensora-v1-2/inference/sample_config.py   --num-frames 2s --resolution 480p   --layernorm-kernel False --flash-attn False --batch-size 8 --prompt ["a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall", "a beautiful waterfall"]
