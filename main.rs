extern crate minifb;
use minifb::{Key, Window, WindowOptions};

use core::ops::{Mul, Add, Div};

use nalgebra::{Vector3, Vector4, Matrix4};

mod display_wrap;

use std::fmt::Display;

fn color_0() -> Color<f32> {
    Color::<f32>::new(0.0, 0.0, 0.0)
}

fn color_x(x: f32, y: f32, z: f32) -> Color<f32> {
    Color::<f32>::new(x, y, z)
}

fn vec3_0() -> Vector3<f64> {
    Vector3::<f64>::new(0.0, 0.0, 0.0)
}

fn vec3_x(x: f64, y: f64, z: f64) -> Vector3<f64> {
    Vector3::<f64>::new(x, y, z)
}

#[derive(Copy, Clone, Debug)]
struct Color<T> {
    red: T,
    blue: T,
    green: T,
}

impl<T> Color<T> {
    fn new<U>(red: U, blue: U, green: U) -> Color<U> {
        Color {
            red,
            blue,
            green,
        }
    }
}

trait ToBuffer<T> {
    fn normalized_to_u32(self) -> u32;
}

impl ToBuffer<f32> for Color <f32> {
    //Use after tone mapping to buffer;
    fn normalized_to_u32(self) -> u32{
        let red = self.red * 255f32;
        let green = self.green * 255f32;
        let blue = self.blue * 255f32;
        u32::from_be_bytes([0x00, red as u8, green as u8, blue as u8])
    }
}

impl Mul<f32> for Color<f32> {
    type Output = Self;
    fn mul(self, rhs: f32) -> Color<f32> {
        Color::<f32>::new(self.red * rhs, self.blue * rhs, self.green * rhs)
    }
}

impl Mul<Self> for Color<f32> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Color<f32> {
        Color::<f32>::new(self.red * rhs.red, self.blue * rhs.blue, self.green * rhs.green)
    }
}



impl Add<Self> for Color<f32> {
    type Output = Self;
    fn add(self, rhs: Self) -> Color<f32> {
        Color::<f32>::new(self.red + rhs.red, self.blue + rhs.blue, self.green + rhs.green)
    }
}

impl Div<f32> for Color<f32> {
    type Output = Self;
    fn div(self, rhs: f32) -> Color<f32> {
        Color::<f32>::new(self.red / rhs, self.blue / rhs, self.green / rhs)
    }
}

struct Ray {
    origin: Vector3<f64>,
    direction: Vector3<f64>,
}

impl Ray {
    fn new(origin: Vector3<f64>, direction: Vector3<f64>) -> Ray {
        Ray {
            origin,
            direction,
        }
    }
}

pub enum RayTerminator {
    Light,
    Sphere,
}

struct PointLightSource {
    origin: Vector3<f64>,
    light: Color<f32>,
}

impl PointLightSource {
    fn new(origin: Vector3<f64>, light: Color<f32>) -> PointLightSource {
        PointLightSource {
            origin,
            light,
        }
    }
}

struct Sphere {
    origin: Vector3<f64>,
    radius: f64,
    color: Color<f32>,
}

pub struct Intersection {
    position: Vector3<f64>,
    normal: Vector3<f64>,
    light: Color<f32>,
    source: RayTerminator,
}

impl Intersection {
    fn new(position: Vector3<f64>, normal: Vector3<f64>, light: Color<f32>, source: RayTerminator) -> Intersection {
        Intersection {
            position,
            normal,
            light,
            source,
        }
    }
}

impl Sphere {
    fn new(origin: Vector3<f64>, radius: f64, color: Color<f32>) -> Sphere {
        Sphere {
            origin,
            radius,
            color,
        }
    }
}

trait Intersectable {
    fn intersect_ray(&self, ray: &Ray) -> Option<Intersection>;
}

impl Intersectable for Sphere {
    
    fn intersect_ray(&self, ray: &Ray) -> Option<Intersection> {
        let delta_origins = self.origin - ray.origin;

        let r_dir = ray.direction.normalize();

        let a = r_dir.norm_squared();
        
        let b = delta_origins.dot(&r_dir);

        let discriminant = b * b - delta_origins.norm_squared() + self.radius * self.radius;

        if discriminant < 0.0 {
            return None;
        } else {
            let t = (- b - discriminant.sqrt()) / (2.0 * a);
            let position = (t + b) * r_dir;
            let normal = self.origin - position;

            let color = self.color;
            //print!("{:?}", color);
            return Some(Intersection::new(position, normal, self.color, RayTerminator::Sphere));
        }
        
    }
}

impl Intersectable for PointLightSource {
    fn intersect_ray(&self, ray: &Ray) -> Option<Intersection> {
        Some(Intersection::new(self.origin - ray.origin, Vector3::<f64>::new(0.0, 0.0, 0.0), self.light, RayTerminator::Light))
    }
}

struct Camera {
    width: usize,
    height: usize,
    position: Vector3<f64>,
    rotation: Vector3<f64>,
    aspect_ratio: f64,
    h_fov: f64,
}

impl Camera {
    fn new(width: usize, height: usize, h_fov: f64, position: Vector3<f64>, rotation: Vector3<f64>) -> Camera {
        let aspect_ratio = (height as f64)/(width as f64);
        Camera {
            width,
            height,
            position,
            rotation,
            aspect_ratio,
            h_fov,
        }
    }

    pub fn new_position(mut self, pos: Vector3<f64>){
        self.position = pos;
    }

    pub fn new_rotation(mut self, dir: Vector3<f64>){
        self.rotation = dir;
    }
}


struct BufferElement {
    color: Color<f32>,
    distance_squared: f64,
}

impl BufferElement {
    fn empty() -> BufferElement {
        BufferElement {
            color: Color::<f32>::new(0.0, 0.0, 0.0),
            distance_squared: std::f64::INFINITY,
        }
    }
    fn update(mut self, in_color: Color<f32>) {
        self.color = self.color + in_color;
    }
}

impl Copy for BufferElement {}

impl Clone for BufferElement {
    fn clone(&self) -> Self {
        *self
    }
}

/* fn main() {
    let camera = Camera::new(600, 400, 90.0, Vector3::<f64>::new(0f64, 0f64, 0f64), Vector3::<f64>::new(0f64,0f64,0f64));

    let mut buffer = vec![Color::<f32>::new(0.0, 0.0, 0.0); camera.width * camera.height];
    let mut buffer_max : f32 = 0.0;

    let mut window = display_wrap::new(camera.width, camera.height, 60);

    let sphere1 = Sphere::new(Vector3::<f64>::new(0.0, 0.0, 30.0), 12.0 , Color::<f32>::new(1.0, 0.0, 0.0));
    let sphere2 = Sphere::new(Vector3::<f64>::new(-10.0, 0.0, 20.0), 4.0 , Color::<f32>::new(0.0, 1.0, 0.0));

    let light1 = PointLightSource::new(Vector3::<f64>::new(10.0, 20.0, 20.0), Color::<f32>::new(5.0, 5.0, 1.0));
    //let light2 = PointLightSource::new(Vector3::<f64>::new(-40.0, -30.0, 40.0), Color::<f32>::new(1.5, 1.5, 1.0));

    let objectArray = vec![&sphere1, &sphere2];
    let lightArray = vec![&light1];

    //println!("V: {}", camera.vert_fov);
    //println!("H: {}", camera.hor_fov);

    for y in 0..camera.height {
        let y_val = ((y as f64/camera.height as f64 * 2.0) - 1.0) * camera.aspect_ratio * (camera.h_fov/2.0).tan() ;
        for x in 0..camera.width {
            let x_val = ((x as f64/camera.width as f64 * 2.0) -1.0) * (camera.h_fov/2.0).tan();
            let ray_dir = Vector3::<f64>::new(x_val, y_val, 1.0);
            let mut pixel_light_rays = Vec::new();
            


            let mut objectArray_index = 0;
            let mut ray = Ray::new(camera.position, ray_dir);
            let mut successHit = BufferElement::empty();
            let mut successInt: Intersection = Intersection::new(Vector3::<f64>::new(0.0, 0.0, 0.0), Vector3::<f64>::new(0.0, 0.0, 0.0), Color::<f32>::new(0.0, 0.0, 0.0), RayTerminator::Sphere);
            let mut was_hit = false;
            let mut lightIntensity = Color::<f32>::new(0.0, 0.0, 0.0);
            for i in 0..objectArray.len() {
                let object = objectArray[i];
                let hit = object.intersect_ray(&ray);
                match hit {
                    Some(intersection) => {
                        //println!("Hit", );
                        was_hit = true;
                        let distance_squared = (intersection.position- object.origin).norm_squared();
                        if distance_squared < successHit.distance_squared {
                            successHit.distance_squared = distance_squared;
                            successInt = intersection;
                            objectArray_index = i;
                        }
                    }
                    None => {}
                }                
            }
            
            if was_hit == true {
                //Calculate lighting               
                //Intersect all lights, add to visibility list

                //Sum intensity
                let mut ray_holder: Intersection = Intersection::new(Vector3::<f64>::new(0.0, 0.0, 0.0), Vector3::<f64>::new(0.0, 0.0, 0.0), Color::<f32>::new(0.0, 0.0, 0.0), RayTerminator::Sphere);
                for i in 0..lightArray.len() {
                    
                    let light = lightArray[i];
                    let sub_ray = Ray::new(successInt.position, light.origin - successInt.position);
                    let hit = light.intersect_ray(&sub_ray);
                    match hit {
                        Some(ray) => {
                            ray_holder = ray;
                        
                        }
                        None => {}
                    }
                    let mut was_ocl = false;
                    for j in 0..objectArray.len() {
                        if j == objectArray_index {
                            continue;
                        }
                        let object = objectArray[i];
                        let occlude = object.intersect_ray(&ray);
                        match occlude {
                            Some (intersection) => {         
                                
                                let distance = (intersection.position.norm_squared());
                                if distance < (successInt.position).norm_squared() {
                                    //Light occluded
                                    println!("Occluded {} {}",x, y );
                                    was_ocl = true;
                                    break;
                                }
                            }
                            None => {}
                        }
                    }
                    if was_ocl != true {
                        let mut light_calc = successInt.normal.normalize().dot(&sub_ray.direction.normalize()); 
                        if light_calc > 0.0 {
                            lightIntensity = lightIntensity + (light.light * successInt.light *  light_calc as f32); 
                        }
                        //println!("{}", light_calc);

                        
                    } else {
                        lightIntensity = Color::<f32>::new(0.0, 0.0, 0.0);
                    }
                    

                }
                lightIntensity = lightIntensity / std::f32::consts::PI;
            } 
            let maxes = vec![lightIntensity.red, lightIntensity.green, lightIntensity.blue];
            for i in 0..maxes.len() {
                let val = maxes[i];
                if val > buffer_max {
                    buffer_max = val;
                }
            }
            buffer[y * camera.width + x] = lightIntensity;//Color::<f32>::new(1.0, 0.0, 0.0);
        }
    }

    let mut final_buffer: Vec<u32> = vec![0 ; camera.width * camera.height];
    let inv_max = 1.0 / buffer_max;
    for i  in 0..final_buffer.len() {
        final_buffer[i] = (buffer[i] * inv_max).normalized_to_u32();
    }

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window
            .update_with_buffer(&final_buffer, camera.width, camera.height)
            .unwrap();
    }
} */

fn update_buffer(val: Color<f32>, x: u32, y: u32, stride: u32, buffer: &mut [Color<f32>], buffer_max: &mut f32) {
    buffer[(y * stride + x )as usize] = val;
    let maxes = vec![val.red, val.green, val.blue];
    for i in 0..maxes.len() {
        let val = maxes[i];
        if val > *buffer_max {
            *buffer_max = val;
        }
    }
}

fn main() {
    let camera = Camera::new(600, 400, 90.0, vec3_x(0f64, 0f64, 0f64), vec3_x(0f64,0f64,0f64));

    let mut buffer = vec![color_0(); camera.width * camera.height];
    let mut buffer_max : f32 = 0.0;

    let mut window = display_wrap::new(camera.width, camera.height, 60);

    let sphere1 = Sphere::new(vec3_x(0.0, 0.0, 30.0), 12.0 , color_x(1.0, 0.0, 0.0));
    let sphere2 = Sphere::new(vec3_x(-10.0, 0.0, 20.0), 4.0 , color_x(0.0, 1.0, 0.0));

    let light1 = PointLightSource::new(vec3_x(10.0, 20.0, 20.0), color_x(5.0, 5.0, 1.0));
    //let light2 = PointLightSource::new(vec3_x(-40.0, -30.0, 40.0), color_x(1.5, 1.5, 1.0));

    let objectArray = vec![&sphere1, &sphere2];
    let lightArray = vec![&light1];
    for y in 0..camera.height {
        let y_val = ((y as f64/camera.height as f64 * 2.0) - 1.0) * camera.aspect_ratio * (camera.h_fov/2.0).tan() ;
        for x in 0..camera.width {
            //Generate ray directions
            let x_val = ((x as f64/camera.width as f64 * 2.0) -1.0) * (camera.h_fov/2.0).tan();
            let ray_dir = vec3_x(x_val, y_val, 1.0);
            let ray = Ray::new(vec3_0(), ray_dir);
            //
            let mut pixel_result = color_0();
            let mut distance_squared = std::f64::INFINITY;


            //Find visble object
            let mut closest_hit: Intersection;
            for i in 0..objectArray.len() {
                let object_hit = objectArray[i].intersect_ray(&ray);
                match object_hit {
                    Some(intersection) => { //Update distance
                        let d = intersection.position.norm_squared();
                        if d < distance_squared {
                            distance_squared = d;
                            closest_hit = intersection;
                            //We have a hit for this pixel
                        }
                        
                    }
                    None => {}                
                }
            }
            let light_accum = color_0();
            match closest_hit {
                Some(intersection) => {
                    //Calculate light for this intersection

                    let mut light_hit_array = vec!();
                    let mut light_hit: Option<Intersection> = None;
                    
                    //
                    for i in 0..lightArray.len() {
                        let light = lightArray[i];
                        let ray = Ray::new(closest_hit.position, closest_hit.position - light.origin);
                        light_hit = light.intersect_ray(&ray);
                        match light_hit {
                            Some(intersection) => {
                                light_accum = light_accum + intersection.normal.dot(&closest_hit.normal) ( intersection.light);
                                
                            }
                            None => {}                
                        }
                    }
                }
                None => {}
            }
            update_buffer(light_accum, x as u32, y as u32, camera.width as u32, &mut buffer, &mut buffer_max);
        }
    }
    let mut final_buffer: Vec<u32> = vec![0 ; camera.width * camera.height];
    let inv_max = 1.0 / buffer_max;
    for i  in 0..final_buffer.len() {
        final_buffer[i] = (buffer[i] * inv_max).normalized_to_u32();
    }
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window
            .update_with_buffer(&final_buffer, camera.width, camera.height)
            .unwrap();
    }
}