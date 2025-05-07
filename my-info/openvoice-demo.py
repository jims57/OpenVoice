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


# reference_speaker = 'resources/example_reference.mp3' # This is the voice you want to clone
# reference_speaker = 'resources/prompt_audio_mono.mp3' # SparkTTS sample voice(male voice)
reference_speaker = 'resources/雷军.wav' # SparkTTS sample voice(male voice)
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

from melo.api import TTS

texts = {
    'EN_NEWEST': "Did you ever hear a folk tale about a giant turtle?",  # The newest English base speaker model
    'EN': "Did you ever hear a folk tale about a giant turtle?",
    'ES': "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante.",
    'FR': "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante.",
    'ZH': "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
    'JP': "彼は毎朝ジョギングをして体を健康に保っています。",
    'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
}


src_path = f'{output_dir}/tmp.wav'

# Speed is adjustable
speed = 1.0

for language, text in texts.items():
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        
        source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        if torch.backends.mps.is_available() and device == 'cpu':
            torch.backends.mps.is_available = lambda: False
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
        save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
            message=encode_message)
