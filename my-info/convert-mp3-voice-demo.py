import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

ckpt_converter = 'checkpoints_v2/converter'
# Check if CUDA is available and print device information
is_cuda_available = torch.cuda.is_available()
device = "cuda:0" if is_cuda_available else "cpu"
print(f"CUDA available: {is_cuda_available}")
if is_cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
else:
    print(f"CUDA is not available. Using CPU for inference.")
    print(f"Using device: {device}")
output_dir = 'outputs_v2'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

# Reference voice to clone (target voice)
# reference_speaker = 'reference-mp3/mayun_zh.wav'
reference_speaker = 'reference-mp3/情感女声.wav'
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

# Source audio file (voice to be converted)
# source_audio_path = 'melo-mp3/20250401074538.wav'
source_audio_path = 'melo-mp3/1746619459342-melo-1.mp3'
# source_audio_path = 'melo-mp3/20250401074538.wav'

# Extract source embedding
source_se, _ = se_extractor.get_se(source_audio_path, tone_color_converter, vad=True)

# Output path for converted audio
output_path = f'{output_dir}/converted_to_mayun.wav'

# Measure tone color conversion time
start_time = torch.cuda.Event(enable_timing=True)
end_time = torch.cuda.Event(enable_timing=True)

start_time.record()
encode_message = "@MyShell"
tone_color_converter.convert(
    audio_src_path=source_audio_path, 
    src_se=source_se, 
    tgt_se=target_se, 
    output_path=output_path,
    tau=0.1,
    message=encode_message)
end_time.record()

# Wait for GPU synchronization
torch.cuda.synchronize()
conversion_time_ms = start_time.elapsed_time(end_time)
print(f"Tone color conversion time: {conversion_time_ms:.2f} ms")
print(f"Converted audio saved to: {output_path}")
