![image](https://github.com/user-attachments/assets/430d607d-4d0c-45d1-bb58-2460f0c1debd)

# Lane Detection with Rust + OpenCV

Rust 언어와 OpenCV를 활용한 차선(Lane) 검출 데모 프로젝트입니다.

슬라이딩 윈도우(Sliding Window)와 투시 변환(Perspective Transform) 기법을 통해 **영상(실시간 웹캠 또는 동영상 파일)** 에서 차선을 검출하고, 검출된 차선 정보를 바탕으로 **조향각(steering angle)** 을 시각화합니다.

---

## 목차

- [프로젝트 특징](#프로젝트-특징)  
- [개요](#개요)  
- [프로젝트 구조](#프로젝트-구조)  
- [요구 사항](#요구-사항)  
- [설치 & 빌드](#설치--빌드)  
- [내용 설명](#내용-설명)  
  - [파이프라인(Pipeline) 구조체](#파이프라인pipeline-구조체)  
  - [핵심 알고리즘 설명](#핵심-알고리즘-설명)  
- [실행 화면 예시](#실행-화면-예시)  
- [사용 방법](#사용-방법)  
- [라이선스](#라이선스)

---

## 프로젝트 특징

- **Rust의 안전성**  
  빌림(Ownership) 규칙으로 인해 C/C++ 대비 런타임 에러와 메모리 누수 위험이 낮습니다.

- **OpenCV Crate**  
  Rust에서 C++의 OpenCV 함수를 안전하게 호출 가능합니다.

- **슬라이딩 윈도우 & 투시 변환**  
  동적/정적 영상 속 차선을 검출하는 데 효과적인 알고리즘 사용.

- **조향각 계산**  
  검출된 차선 정보를 기반으로 차량의 조향각을 추정하고, 실시간 시각화(영상 위 표시)까지 구현합니다.

---

## 개요

- Rust 환경에서 영상 처리를 수행하기 위해 [`opencv` 크레이트](https://crates.io/crates/opencv)를 사용합니다.

- **`Pipeline`** 구조체를 통해 전체 차선 검출 프로세스(ROI 설정, 투시 변환, 슬라이딩 윈도우, 조향각 산출 등)를 관리합니다.

- `main.rs` 실행 시, 웹캠(기본 장치 0) 또는 `./video/challenge.mp4` 동영상 파일을 열어 매 프레임마다 차선을 인식하고 시각화합니다.

---

## 프로젝트 구조

```
Lane_Detection_Rust/
├── Cargo.toml
├── README.md                  // 본 파일
├── src/
│   ├── main.rs               // 메인 함수 (start_detection 호출)
│   └── lane_detection.rs     // Pipeline 구조체 및 차선 인식 로직
└── video/
    └── challenge.mp4         // 샘플 테스트 영상
```

---

## 요구 사항

- **Rust 설치**  
  - 예시 버전: Rust 1.87 (nightly)  
  - [Rust 공식 문서](https://www.rust-lang.org/learn/get-started)에 따라 설치

- **OpenCV 라이브러리**  
  - OpenCV C++ 라이브러리가 미리 설치되어 있어야 함 (예: 4.11버전)  
  - 시스템에 설치된 OpenCV를 `opencv` 크레이트가 FFI로 호출

- **플랫폼별 OpenCV 설치 예시**:
  - **Ubuntu**  
    ```bash
    sudo apt-get install libopencv-dev
    ```
  - **Windows**  
    [OpenCV.org 공식 빌드](https://opencv.org/releases/)나 [vcpkg](https://github.com/microsoft/vcpkg) 등을 통해 설치  
    설치 후, 환경변수 `OPENCV_DIR`(또는 `OpenCV_DIR`) 설정이 필요할 수 있음

---

## 설치 & 빌드

### 1. 레포지토리 클론

```bash
git clone https://github.com/sangho007/Lane_Detection_Rust.git
cd Lane_Detection_Rust
```

### 2. Cargo 빌드 & 실행

```bash
cargo run
```

- 빌드 후 자동으로 `./video/challenge.mp4` 영상을 열어 차선을 검출합니다.  
- 웹캠을 사용하고 싶다면 `start_detection` 함수의 인자를 `0`에서`1`로 변경한 뒤 `cargo run`을 재실행하세요.

---

## 내용 설명

### 파이프라인(Pipeline) 구조체

```rust
pub struct Pipeline {
    width: i32,
    height: i32,
    window_margin: i32,
    vertices: Vec<Point>,
    transform_matrix: Mat,
    inv_transform_matrix: Mat,
    // ...
    exit_flag: bool,
}
```

- **width, height**: 입력(카메라/영상) 해상도 (기본 1280×720)
- **vertices**: 관심 영역(ROI) 설정을 위한 꼭짓점 4개
- **transform_matrix, inv_transform_matrix**: 투시 변환(탑뷰) 및 역투시 변환(원근뷰)을 위한 3×3 행렬
- **sliding window 관련 파라미터**(`window_margin`, `leftx_base` 등)와, 이전/현재 프레임의 조향각(`prev_angle`, `steering_angle`)을 관리
- **exit_flag**: 루프 종료 플래그 (키보드 `q` 입력 시 `true`)

#### 주요 메서드

1. **new()**  
   - Pipeline 인스턴스 초기화(ROI 정점, 투시 변환 행렬 계산, 파라미터 기본값 설정 등)

2. **start_detection(mode: i32)**  
   - `mode`를 기준으로 웹캠(기본 0번) 또는 `./video/challenge.mp4` 파일을 열어 프레임 입력  
   - 메인 루프에서 매 프레임마다 `processing()` 호출  
   - 키보드 `q` 입력 시 `exit_flag = true`로 종료

3. **processing(frame: &Mat) -> Result<Mat>**  
   - 단일 프레임을 받아 아래 단계를 수행 후, 결과 영상을 반환:
     1. 그레이 변환 → 블러 → 캐니 엣지 → 모폴로지 닫힘 → ROI → Bird’s-eye 변환  
     2. 슬라이딩 윈도우로 차선 픽셀 추출, 2차 곡선 피팅  
     3. 조향각 계산  
     4. 역투시 변환 후 원본 영상에 차선 정보 합성  
     5. FPS, 조향각 텍스트 표시  
     6. 최종 영상 리턴

4. **슬라이딩 윈도우(sliding_window)**  
   - 탑뷰 상태의 이진 영상에서 세로 방향으로 분할된 윈도우를 사용하여 좌우 차선 픽셀 추적  
   - 검출된 픽셀을 토대로 2차 곡선 \(ax^2 + bx + c\)를 피팅(Polyfit)

5. **조향각 계산(get_angle_on_lane)**  
   - 검출된 왼/오 차선을 1차 선형 근사해 교점을 구하거나(양쪽)  
   - 한쪽만 인식된 경우 해당 차선 기울기로 조향각을 계산  
   - \(\text{atan}(\text{slope}) \times \frac{180}{\pi}\) 값을 사용해 0° 기준(정면)에 좌우 ±각도로 표현

---

### 핵심 알고리즘 설명

1. **ROI(Region of Interest)**  
   - 하단 및 좌우 측면 위주의 관심 구역만 남겨 불필요한 배경(하늘, 차량 전면부 등)을 제외

2. **투시 변환(Perspective Transform)**  
   - Bird’s-eye View로 변환하여 차선을 “평면화”  
   - 검출 후, Inverse Perspective로 되돌려 원본 영상과 합성

3. **슬라이딩 윈도우 (Sliding Window)**  
   - 영상 하단부터 일정 높이씩 구분해 각 구간(윈도우) 내 차선 픽셀을 추적  
   - 해당 구간의 평균 좌표를 다음 윈도우의 중심으로 이용

4. **다항식(2차) 피팅 (Polyfit)**  
   - Least-Square Method로 픽셀 좌표를 2차 다항식 계수로 피팅  
   - Rust에서는 직접 행렬 연산(크래머 공식 등)으로 구현 가능

5. **조향각(Steering Angle) 계산**  
   - 차선 곡선 또는 1차 근사선의 기울기로부터 \(\text{atan}(\text{slope})\) 계산  
   - 0°를 정면으로 보고, 좌/우를 ± 각도(예: ±20°)로 표현

---

## 실행 화면 예시

아래는 예시 이미지로, 왼쪽은 슬라이딩 윈도우 분석(탑뷰) 결과, 오른쪽은 원본 영상에 차선과 조향각을 시각화한 모습입니다.

| 탑뷰(슬라이딩 윈도우) | 원본영상(차선+조향각) |
<img width="1392" alt="image" src="https://github.com/user-attachments/assets/dc1467c7-858a-408c-9bf4-dd7a160296bb" />




---

## 사용 방법

1. **웹캠 사용**  
   - `CAM_MODE`를 1로 설정(기본값) 후 `cargo run` 실행  
   - 기본적으로 `0번` 카메라를 사용 (필요 시 `start_detection(1)` 내 인자 변경)

2. **동영상 파일 사용**  
   - `CAM_MODE`를 1 이외의 값으로 바꾸거나, `start_detection` 함수 인자로 다른 정수를 전달  
   - `./video/challenge.mp4` 파일이 존재해야 함

3. **종료 조건**  
   - 매 프레임 시청 중 `q` 키 입력 시 `exit_flag`가 `true`가 되어 루프 종료

---

## 라이선스 
- OpenCV 라이브러리는 Apache 2/BSD 라이선스로 배포됩니다. 해당 라이선스를 준수해주세요.
---
