
## ONNX Model Conversion Scripts

A collection of PyTorch-based models converted to the ONNX format, with ready-to-use ONNX Runtime inference scripts.
All generated .onnx models stored in [Hugging Face ONNX Community](https://huggingface.co/onnx-community)

This project aims to make it easier to:

  * Convert PyTorch models to ONNX format

  * Run efficient inference using ONNX Runtime

  * Learn by example, with minimal boilerplate

📂 Project Structure

    ├── README.md
    ├── model_name_1/
    │   ├── model_name_1_to_onnx_conversion_script.py  # Script to export PyTorch model to ONNX
    │   ├── model_name_1_inference_script.py  # Script to run inference with ONNX Runtime
    ├── model_name_2/
    │   ├── model_name_2_to_onnx_conversion_script.py 
    │   ├── model_name_2_inference_script.py 
    └── ...

🚀 Getting Started

### 1. Clone the repository

    git clone https://github.com/<your-username>/onnx_conversion_scripts.git
    cd <your-repo>

### 2. Install dependencies
    You'll need Python 3.9+ and pip:
    See dependencies and versions info in scripts respectively

📚 Supported Models

| Model Name                | Source                    | Notes                                  | Original repo                                            | ONNX repo
| -----------------         | ------------------------  | -------------------------------------  | ---------------                                          | ------------
| Chatterbox                | resemble-ai/chatterbox-tts| Text to Speech, Speech to Speech       | [GitHub Link](https://github.com/resemble-ai/chatterbox) | [HF Link](https://huggingface.co/onnx-community/chatterbox-ONNX)
| Chatterbox Multilingual   | resemble-ai/chatterbox-tts| Text to Speech, Speech to Speech       | [GitHub Link](https://github.com/resemble-ai/chatterbox) | [HF Link](https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX)
| Perth Watermarker         | resemble-ai/Perth         | Audio Watermarking                     | [GitHub Link](https://github.com/resemble-ai/Perth)      | [HF Link](https://huggingface.co/onnx-community/perth-watermarker-ONNX)
| ...                       | ...                       | ...                                    | ...                                                      | ...

(More models coming soon!)


📜 License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/)

### Project author: Vladislav Bronzov

### Email: vladislav.bronzov@gmail.com


