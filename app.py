from typing import Any, Dict

import gradio as gr
import librosa
import numpy as np
import torch
import os, shutil
from datetime import datetime
from transformers import WavLMForSequenceClassification


def feature_extract_simple(
    wav,
    sr=16_000,
    win_len=15.0,
    win_stride=15.0,
    do_normalize=False,
) -> np.ndarray:
    """Simple feature extraction for WavLM.
    Parameters
    ----------
    wav : str or array-like
        path to the wav file, or array-like
    sr : int, optional
        sample rate, by default 16_000
    win_len : float, optional
        window length, by default 15.0
    win_stride : float, optional
        window stride, by default 15.0
    do_normalize: bool, optional
        whether to normalize the input, by default False.
    Returns
    -------
    np.ndarray
        batched input to WavLM
    """
    if type(wav) == str:
        signal, _ = librosa.core.load(wav, sr=sr)
    else:
        try:
            signal = np.array(wav).squeeze()
        except Exception as e:
            print(e)
            raise RuntimeError
    batched_input = []
    stride = int(win_stride * sr)
    l = int(win_len * sr)
    if len(signal) / sr > win_len:
        for i in range(0, len(signal), stride):
            if i + int(win_len * sr) > len(signal):
                # padding the last chunk to make it the same length as others
                chunked = np.pad(signal[i:], (0, l - len(signal[i:])))
            else:
                chunked = signal[i : i + l]
            if do_normalize:
                chunked = (chunked - np.mean(chunked)) / (np.std(chunked) + 1e-7)
            batched_input.append(chunked)
            if i + int(win_len * sr) > len(signal):
                break
    else:
        if do_normalize:
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-7)
        batched_input.append(signal)
    return np.stack(batched_input)  # [N, T]


def infer(model, inputs) -> torch.Tensor:
    output = model(inputs)
    probs = torch.sigmoid(torch.Tensor(output.logits))
    return probs


def predict(audio_file) -> Dict[str, Any]:
    if audio_file is None:
        return {"No prediction available": 0.0}
    else:
        save_dir = "recordings"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(save_dir, f"{timestamp}.wav")
        shutil.copy(audio_file, dest)
        print(f"Audio saved to {dest}")

    try:
        input_np = feature_extract_simple(audio_file, sr=16000, do_normalize=True)
        input_pt = torch.Tensor(input_np)

        probs = infer(model, input_pt)
        probs_list = probs.reshape(-1, len(labels)).detach().tolist()

        # Create a results dictionary
        if len(probs_list) > 0:
            first_segment_probs = probs_list[0]
            results = {
                label: float(prob) for label, prob in zip(labels, first_segment_probs)
            }

            # If there are multiple segments, include that information in the results
            if len(probs_list) > 1:
                results["Note"] = (
                    f"Audio contains {len(probs_list)} segments. Showing first segment only."
                )
        else:
            results = {"Error": "No segments detected in audio"}

        # Sort by confidence score
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

        return sorted_results
    except Exception as e:
        return {"Error": str(e)}


if __name__ == "__main__":
    model_path = "Roblox/voice-safety-classifier-v2"
    labels = [
        "Discrimination",
        "Harassment",
        "Sexual",
        "IllegalAndRegulated",
        "DatingAndRomantic",
        "Profanity",
    ]

    model = WavLMForSequenceClassification.from_pretrained(
        model_path, num_labels=len(labels)
    )
    model.eval()

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Audio(type="filepath", label="Upload or record audio"),
        outputs=gr.Label(num_top_classes=6, label="Classification Results"),
        title="Voice Safety Classifier",
        description="""This app uses the Roblox Voice Safety Classifier v2 model to identify potentially unsafe content in audio.
    Upload or record an audio file to get started. The model classifies audio into categories including Discrimination,
    Harassment, Sexual, IllegalAndRegulated, DatingAndRomantic, and Profanity.

    The model processes audio in 15-second chunks and returns probability scores for each category.""",
        examples=[],
        flagging_mode="never",
    )

    demo.launch(share=True)