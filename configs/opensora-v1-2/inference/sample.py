resolution = "240p"
aspect_ratio = "9:16"
num_frames = 51
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/home/jovyan/models/OpenSora-STDiT-v3/models/snapshots/9a8583918505ee93bd9fae8dd5ce32e1f9334c71/",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="/home/jovyan/models/OpenSora-VAE-v1.2/models/snapshots/33d153e9b5a9f771a8a84f98bd3f46458a8ed0bf/",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="/home/jovyan/models/t5-v1_1-xxl/models/snapshots/c9c625d2ec93667ec579ede125fd3811d1f81d37/",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None
