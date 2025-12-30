# 2025 SCPC AI Challenge Solution (Ricepunchb)
본 레포지토리는 2025 SCPC AI 챌린지 ("일상 사진을 풀어내는 멀티모달 AI 모델")의 솔루션 코드를 포함하고 있습니다. **KOSMOS-2**를 이용한 이미지 캡셔닝과 **Flan-T5-Large**를 이용한 질의응답(QA)을 결합한 파이프라인 방식을 사용하여, 제한된 하드웨어 환경(RTX 3090)에서도 효율적인 추론이 가능하도록 설계되었습니다.

## 프로젝트 개요
- **팀명/참가자**: Ricepunchb (배수현)

- **핵심 전략**: End-to-End 파이프라인

    1. Image Captioning: Vision-Language Model (KOSMOS-2)이 이미지를 상세하게 묘사하는 텍스트(Description)를 생성

    2. Reasoning & QA: Large Language Model (Flan-T5-Large)이 생성된 묘사, 질문, 선택지를 읽고 최종 정답을 추론

- **최적화**: QLoRA (4-bit quantization), Flash Attention, `torch.compile` 등을 활용하여 추론 속도 및 메모리 효율성 극대화

## 개발 및 실행 환경
이 코드는 다음 환경에서 테스트되었습니다.

- **OS**: Debian GNU/Linux 12 (bookworm)

- **CPU**: AMD Ryzen 9 5900X

- **GPU**: NVIDIA GeForce RTX 3090 (VRAM 24GB)

- **CUDA**: 12.8 (Driver 570.133.07)

- **Python**: 3.11.12

- **Library**: PyTorch 2.7.1+cu128, Transformers 4.54.1

## 디렉토리 구조
프로젝트의 전체 디렉토리 구조는 다음과 같습니다.
```
.
├── datasets/                 # 데이터셋 저장 경로
│   ├── Realworld/            # RealworldQA
│   ├── stanford_img_para_caption/
│   ├── visual7w/             # Visual 7w (setup 시 다운로드)
│   └── VMC/                  # VMCBench
├── eg/                       # 대회용 데이터 (Train/Test csv, Images)
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── train_aug.jsonl
│   └── (images...)
├── Model/                    # 학습된 모델 가중치 저장 경로
│   ├── flan-t5/
│   └── kosmos2/
├── results/                  # 추론 결과(submission.csv) 저장 경로
├── src/                      # 소스 코드
│   ├── flan-t5/              # Flan-T5 학습 코드
│   ├── kosmos2/              # KOSMOS-2 학습 코드
│   ├── augment.py            # 데이터 증강 (Captioning 생성)
│   ├── customdatasets.py     # 데이터셋 로더 및 전처리
│   ├── inference.py          # 최종 추론 및 제출 파일 생성
│   ├── init_setup.sh         # 초기 환경 설정 스크립트
│   └── requirements.txt      # 필수 패키지 목록
└── README.md
```

## 설치 및 실행 방법
1. **초기 환경 설정 (Setup)**

필수 라이브러리 설치 및 Visual 7w 등 필요한 데이터셋을 다운로드합니다.

```Bash
bash src/init_setup.sh
```

위 스크립트는 `torch`, `cmake` 등 시스템 의존성을 설치하고 `requirements.txt`에 명시된 파이썬 패키지를 설치합니다.

DACON 데이터셋(`open.zip`)(대회용 구글 클라우드 폴더. 현재 비활성화)은 `eg/` 폴더에 위치해야 하며, 스크립트 실행 시 자동으로 압축이 해제됩니다.

2. **데이터 증강 (Data Augmentation)**
로컬 LLM (`LiAutoAD/Ristretto-3B`)을 사용하여 학습 데이터셋에 이미지 캡셔닝(Description)을 추가합니다.

```Bash
python src/augment.py
```

`datasets/` 및 `eg/` 경로에 있는 데이터셋들에 대해 캡션이 추가된 파일(`.jsonl` 등)이 생성됩니다.

3. **모델 학습 (Training)**
**주의:** 모듈 경로 문제로 인해 반드시 프로젝트 루트 디렉토리에서 `python -m` 명령어를 사용하여 실행해야 합니다.

- **Flan-T5-Large 학습**

```Bash
python -m src.flan-t5.rep_t5
```

- **KOSMOS-2 학습**

```Bash
python -m src.kosmos2.rep_kosmos2
```

4. **추론 및 결과 생성 (Inference)**

학습된 모델을 로드하여 `test.csv`에 대한 추론을 수행하고 제출용 `submission.csv`를 생성합니다.

```Bash
python src/inference.py
```

- 결과 파일은 `results/submission.csv`에 저장됩니다.

- **참고:** 제공된 `Model/` 폴더에 가중치 파일이 있다면 학습 과정을 건너뛰고 바로 추론을 실행할 수 있습니다.

## 문제 해결 (Troubleshooting)
`transformers` 라이브러리의 KOSMOS-2 모듈 실행 중 에러가 발생할 경우, 소스 코드에 몽키 패치(Monkey Patch)가 필요할 수 있습니다. `rep_kosmos2.py` 또는 관련 모델링 코드에 다음 수정 사항이 반영되어 있는지 확인해 주세요.

### **수정 위치:** `transformers.models.kosmos2.modeling_kosmos2.Kosmos2TextTransformer.forward_embedding`

```Python
# inputs_embeds가 None일 때 처리 부분 아래에 다음 코드 확인
if image_embeds is not None:
    inputs_embeds = inputs_embeds.clone() # !! HOTFIX !! 이 줄이 추가되어야 함
    inputs_embeds[img_input_mask.to(dtype=torch.bool)] = image_embeds.to(inputs_embeds.device).view(-1, image_embeds.size(-1))
```

- 이슈가 지속될 경우 `inputs_embeds.clone()`을 통해 텐서 복사를 명시적으로 수행해야 오류를 방지할 수 있습니다.
    - 자세한 내용은 코드 설명 자료를 참고해 주시기 바랍니다.

## License & References
- **A-OKVQA:** Apache-2.0

- **Visual 7w / VMC:** MIT License

- **RealworldQA:** CC BY-ND 4.0

- **Stanford Image Paragraph Captioning:** CC0 (Public Domain)

- **Recap-COCO-30K:** CC-BY-4.0

- **DACON Dataset:** SCPC 대회 규정 준수

자세한 내용은 발표자료 및 코드 설명 자료를 참고해 주시기 바랍니다.
