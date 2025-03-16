// lane_detection 모듈 불러오기 (같은 crate 안에 있다고 가정)
mod lane_detection;
use lane_detection::{Pipeline,LaneDetectionResult};

fn main() -> LaneDetectionResult<()> {
    // 파이프라인 생성
    let mut drive = Pipeline::new()?;
    let _ = drive.start_detection(0,true);   // 0 : video , 1 : cam, true : visible, false : invisible

    Ok(())
}
