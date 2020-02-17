use minifb::{Key, Window, WindowOptions};

//Create an instance a return a mutable reference to a buffer
//that gets drawn to screen

pub fn new(width: usize, height: usize, frame_limit: u32) -> Window {
    let mut proto = Window::new(
        "Test",
        width,
        height,
        WindowOptions::default(),
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });
    let mic = ((1.0 / frame_limit as f64) * 1000.0) as u64;
    proto.limit_update_rate(Some(std::time::Duration::from_micros(mic)));
    proto
}