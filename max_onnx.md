*Run an ONNX model with Python*

MAX 엔진은 AI 모델의 추론 속도를 향상시키는 도구다. 특히 PyTorch 모델을 ONNX 형식으로 변환하여 MAX 엔진을 사용하면, 모델을 수정하지 않고도 CPU나 GPU에서 더 빠르게 실행할 수 있다.

**핵심 내용:**

1.  **속도 향상:** MAX 엔진은 AI 모델의 실행 속도를 높여준다.
2.  **간단한 사용법:** 파이썬 API를 통해 쉽게 사용할 수 있다. (단 3줄의 코드로 추론 실행 가능)
3.  **다양한 하드웨어 지원:** CPU, GPU 등 다양한 환경에서 작동한다.
4.  **ONNX 형식 지원:** Hugging Face의 PyTorch 모델을 ONNX 형식으로 변환하여 사용한다.

**사용 방법:**

1.  **MAX 설치:** MAX 엔진을 설치하고 가상 환경을 설정한다.
2.  **모델 변환:** Hugging Face에서 PyTorch 모델을 가져와 ONNX 형식으로 변환한다.
3.  **추론 실행:** MAX 엔진을 사용하여 변환된 모델로 추론을 실행한다.

**요약:**

MAX 엔진은 AI 모델의 성능을 최적화하여 더 빠르고 효율적으로 실행할 수 있게 해주는 도구라고 생각하면 된다. 특히, 모델을 직접 수정할 필요 없이 간단한 코드를 통해 속도 향상을 얻을 수 있다는 장점이 있다.

## MAX 엔진 사용 실습 과정 (쉽게 따라하기)

**목표:** Hugging Face의 ResNet-50 모델을 MAX 엔진으로 실행하여 이미지 분류하기

**준비물:**

*   macOS 또는 Ubuntu Linux (Magic CLI 설치 필요)

**1단계: 개발 환경 설정**

1.  **Magic CLI 설치:** 터미널에 다음 명령어 입력 후 실행
    ```bash
    curl -ssL https://magic.modular.com/deb1ad68-5a03-43ac-9ae5-021f8244ffe5 | bash
    ```
    터미널에 출력된 `source` 명령어를 실행한다.
2.  **새 프로젝트 생성:** 터미널에 다음 명령어 입력 후 실행
    ```bash
    magic init max-onnx-resnet --format pyproject && cd max-onnx-resnet
    ```
3.  **PyTorch 채널 추가:** 터미널에 다음 명령어 입력 후 실행
    ```bash
    magic project channel add pytorch --prepend
    ```
4.  **필요 패키지 설치:** 터미널에 다음 명령어 입력 후 실행
    ```bash
    magic add "max~=24.6" "pytorch==2.4.0" "numpy<2.0" "onnx==1.16.0" \
      "transformers==4.40.1" "datasets==2.18" "pillow"
    ```
5.  **가상 환경 실행:** 터미널에 다음 명령어 입력 후 실행
    ```bash
    magic shell
    ```

**2단계: ONNX 모델 다운로드**

1.  `max-onnx-resnet` 폴더 안에 `download-model.py` 파일 생성 후 다음 코드 복사 & 붙여넣기
    ```python
    import torch
    from transformers import ResNetForImageClassification
    from torch.onnx import export

    # The Hugging Face model name and exported file name
    HF_MODEL_NAME = "microsoft/resnet-50"
    MODEL_PATH = "resnet50.onnx"

    def main():
        # Load the ResNet model from Hugging Face in evaluation mode
        model = ResNetForImageClassification.from_pretrained(HF_MODEL_NAME)
        model.eval()

        # Create random input for tracing, then export the model to ONNX
        dummy_input = torch.randn(1, 3, 224, 224)
        export(model, dummy_input, MODEL_PATH, opset_version=11,
              input_names=['pixel_values'], output_names=['output'],
              dynamic_axes={'pixel_values': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

        print(f"Model saved as {MODEL_PATH}")

    if __name__ == "__main__":
        main()
    ```
2.  터미널에 다음 명령어 입력 후 실행
    ```bash
    python3 download-model.py
    ```
    `resnet50.onnx` 파일이 생성되었는지 확인한다.

**3단계: MAX 엔진으로 추론 실행**

1.  `max-onnx-resnet` 폴더 안에 `run.py` 파일 생성 후 다음 코드 복사 & 붙여넣기
    ```python
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from datasets import load_dataset
    import numpy as np
    from max import engine

    # The Hugging Face model name and exported file name
    HF_MODEL_NAME = "microsoft/resnet-50"
    MODEL_PATH = "resnet50.onnx"

    def main():
        dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
        image = dataset["test"]["image"][0]
        # optionally, save the image to see it yourself:
        # image.save("cat.png")

        image_processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
        inputs = image_processor(image, return_tensors="np")

        print("Keys:", inputs.keys())
        print("Shape:", inputs['pixel_values'].shape)

        session = engine.InferenceSession()
        model = session.load(MODEL_PATH)
        outputs = model.execute_legacy(**inputs)

        print("Output shape:", outputs['output'].shape)

        predicted_label = np.argmax(outputs["output"], axis=-1)[0]
        hf_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_NAME)
        predicted_class = hf_model.config.id2label[predicted_label]
        print(f"Prediction: {predicted_class}")

    if __name__ == "__main__":
        main()
    ```
2.  터미널에 다음 명령어 입력 후 실행
    ```bash
    python3 run.py
    ```
    터미널에 "Prediction: tiger cat"이 출력되는지 확인한다. (처음 실행 시 모델 컴파일 시간이 다소 소요될 수 있음)

**축하합니다! MAX 엔진을 사용하여 이미지 분류를 성공적으로 수행했습니다.**

**참고:**

*   에러 발생 시, 각 단계별 코드를 꼼꼼히 확인하고, 필요한 패키지가 제대로 설치되었는지 확인한다.
*   MAX 엔진 버전(`max~=24.6`)과 PyTorch 버전(`pytorch==2.4.0`)을 명확하게 지정하여 설치한다.
*   터미널 명령어 실행 시, 오타가 없는지 확인한다.
