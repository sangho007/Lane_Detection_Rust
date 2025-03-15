Lane Detection with Rust + OpenCV

이 프로젝트는 Rust 언어와 OpenCV를 활용하여 차선(Lane) 검출을 수행하는 예제 코드입니다.

	주요 특징
		•	Rust의 안전성 & 빌림(Ownership) 규칙으로 인해 C/C++ 대비 런타임 에러, 메모리 누수 위험이 낮습니다.
	•	OpenCV crate를 통해 C++의 OpenCV 라이브러리 함수를 안전하게 호출합니다.
	•	슬라이딩 윈도우(Sliding Window)와 투시 변환(Perspective Transform) 기법을 활용하여 동적/정적 영상 속 차선을 검출합니다.
	•	차선 정보를 바탕으로 조향각을 계산하고, 결과를 실시간으로 시각화(영상 위에 표시)합니다.

⸻

개요
	•	Rust에서 영상 처리를 할 때, opencv crate를 사용합니다.
	•	Pipeline 구조체가 차선 검출의 전체 과정을 수행합니다.
	•	main.rs (또는 본 예제 코드) 실행 시, 내부에서 웹캠 혹은 동영상 파일을 열어서 매 프레임마다 차선을 인식하고, 결과를 화면에 표시합니다.

프로젝트 구조

.

	•	lane_detection.rs: Pipeline 구조체 및 그 외 차선 인식 로직(슬라이딩 윈도우, 투시변환, 다항식 피팅, 등)을 모아둔 모듈.
	•	main.rs: 이 모듈을 불러와서 실제 start_detection 함수를 호출해 캡처 장치(또는 영상 파일)를 실행.

⸻

요구 사항
	•	Rust 설치 (사용버전 1.87 Nightly)
	•	Rust 설치 방법(공식)
	•	OpenCV 라이브러리
	•	OpenCV C++ 라이브러리가 설치되어 있어야 하며, 4.11버전 사용(Mac OS)
	•	플랫폼별 설치 방식:
	•	Ubuntu: sudo apt-get install libopencv-dev
	•	Windows: OpenCV.org의 official build 다운로드 & 빌드 or vcpkg 등을 통해 설치
	•	opencv crate가 이를 FFI로 호출하므로, 시스템에 OpenCV 동적/정적 라이브러리가 설치되어 있어야 합니다.
	•	Cargo를 이용해 빌드 시, opencv crate가 자동으로 시스템 OpenCV를 찾습니다.
	•	만약 찾지 못한다면 환경변수 OPENCV_DIR 또는 OpenCV_DIR 등을 설정해야 할 수 있습니다.

⸻

설치 & 빌드
	1.	레포지토리 클론

git clone https://github.com/sangho007/Lane_Detection_Rust.git
cd Lane_Detection_Rust

	2.	Cargo 빌드 & 실행

cargo run

	•	빌드가 완료되면 실행과 동영상 파일(./video/challenge.mp4)을 열어서 차선 검출이 진행됩니다.
	•	만약 동시에 웹캠(기본 0번 장치)을 사용하고 싶으면, start_detection 함수의 인자를 0에서 1로 바꾸면 됩니다.

⸻

내용 설명

파이프라인(Pipeline) 구조체

pub struct Pipeline {
    width: i32,
    height: i32,
    window_margin: i32,
    vertices: Vec<Point>,
    transform_matrix: Mat,
    inv_transform_matrix: Mat,
    ...
    exit_flag: bool,
}

	•	width, height: 입력(카메라/영상) 해상도 (기본 1280×720)
	•	vertices: 관심영역(ROI) 설정을 위한 꼭짓점 4개
	•	transform_matrix, inv_transform_matrix: 투시 변환(탑뷰)과 역투시 변환(원근뷰)을 위한 3×3 행렬
	•	sliding window 파라미터(window_margin, leftx_base 등)를 갖고 있으며, 매 프레임마다 갱신
	•	prev_angle, steering_angle: 이전/현재 프레임의 차선 기반 조향각(deg)
	•	exit_flag: 메인 루프 종료 여부 (키보드 q 입력 시 true)

주요 메서드
	1.	new()
	•	Pipeline 인스턴스를 초기화(ROI 정점, 투시 변환 행렬 계산, 초기 파라미터 설정 등).
	2.	start_detection(mode: i32)
	•	mode가 CAM_MODE(현재는 1)와 동일하면 웹캠을 열고, 그렇지 않으면 ./video/challenge.mp4 파일을 열어서 매 프레임 처리.
	•	메인 루프에서 매 프레임마다 processing() 메서드 호출.
	•	‘q’를 누르면 exit_flag = true로 설정되어 종료.
	3.	processing(frame: &Mat) -> Result<Mat>
	•	단일 프레임을 받아 다음 과정을 수행 후 최종 영상(차선+조향각 시각화)을 반환:
	1.	그레이 변환 → 블러 → 캐니 → 모폴로지 닫힘 → ROI → Bird’s-eye(투시 변환)
	2.	슬라이딩 윈도우로 차선 픽셀 추출, 2차 곡선 피팅
	3.	조향각 계산
	4.	역투시 변환으로 원본 시야에 차선 정보 합성
	5.	FPS, 조향각 텍스트 표시
	6.	최종 영상 리턴
	4.	슬라이딩 윈도우(sliding_window)
	•	Bird’s-eye 상태의 이진 영상에서, 세로 방향으로 여러 윈도우(기본 15개)를 배치하여, 좌우 차선 픽셀을 추적합니다.
	•	추출된 픽셀들로부터 2차 곡선 ax^2 + bx + c를 피팅(Polyfit)하여 차선을 모델링합니다.
	5.	조향각 계산(get_angle_on_lane)
	•	검출된 왼/오 차선을 1차 선형 근사하여 교점을 구하거나(양쪽 인식 시), 한쪽만 인식 시 해당 차선 기울기로 조향각을 계산.
	•	atan(기울기) * (180 / π) → 0°를 정면으로 맞추기 위해 90° offset 조정 (좌측 음수, 우측 양수).

⸻

핵심 알고리즘 설명
	1.	ROI(Region of Interest)
	•	하단 + 좌/우 측부만 보고 상단의 불필요한 영역(하늘, 차선이 보이지 않는 영역)을 제외하기 위함.
	2.	투시 변환(Perspective Transform)
	•	Bird’s-eye View (위에서 내려다본) 형태로 바꿔서 직선 탐색을 쉽게 함.
	•	검출 후 다시 Inverse Perspective로 되돌려 원본 영상과 합성.
	3.	슬라이딩 윈도우 (Sliding Window)
	•	영상의 하단부터 일정 높이씩 구간을 나누고, 각 구간(윈도우) 내에서 차선 픽셀을 검출.
	•	차선 픽셀이 일정 수 이상이면 그 평균 x좌표를 다음 윈도우의 중심으로 설정하여 추적.
	4.	다항식(2차) 피팅 (Polyfit)
	•	검출된 픽셀 좌표를 Least-Square Method로 2차 다항식 계수를 구함.
	•	Rust에선 직접 행렬식을 풀어서(Cramer's Rule) 구현.
	5.	조향각(Steering Angle) 계산
	•	차선 곡선 혹은 1차 선형 근사로부터 기울기(slope) → atan(slope)로 각도 도출.
	•	편의상 0°를 정면(수직)으로 보고, 좌/우 회전에 따라 ±20° 제한.

⸻

실행 화면 예시

	(예시 이미지는 직접 캡처한 것을 첨부할 수 있습니다.)

	•	왼쪽: 슬라이딩 윈도우 분석 결과 (탑뷰)
	•	오른쪽: 원본 영상 + 차선 시각화 + 조향각 표시

두 이미지를 hconcat(가로합치기) 하여 하나의 창에 표시합니다.

⸻

사용 방법
	1.	웹캠 사용
	•	CAM_MODE 상수를 1로 유지 → cargo run 실행
	•	기본적으로 0번 카메라를 사용
	2.	동영상 파일 사용
	•	CAM_MODE를 1 외의 값으로 바꾸거나, start_detection 함수에 인자로 0이 아닌 다른 정수를 전달
	•	challenge.mp4 파일이 있는지 확인 (또는 직접 경로 변경)
	3.	매 프레임 종료 조건
	•	q 키 입력 시 exit_flag가 true가 되어 루프 종료

