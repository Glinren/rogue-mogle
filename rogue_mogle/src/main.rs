#[macro_use]
extern crate vulkano;
extern crate vulkano_win;
extern crate vulkano_shaders;

extern crate cgmath;
extern crate winit;


use std::iter;
use std::sync::Arc;
use std::option::Option;
use std::fmt;
use std::error;

use vulkano::image::attachment::AttachmentImage;
use vulkano::buffer::{CpuAccessibleBuffer,BufferUsage};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::format::ClearValue;
use vulkano::framebuffer::{Subpass, Framebuffer,
                           RenderPassAbstract, FramebufferAbstract};
use vulkano::instance::{Instance,ApplicationInfo,Version,InstanceCreationError};
use vulkano::instance::PhysicalDevice;
use vulkano::image::SwapchainImage;
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError,
                         Surface, Swapchain, SurfaceTransform,
                         PresentMode, SwapchainCreationError};
use vulkano::pipeline::{GraphicsPipeline,GraphicsPipelineAbstract};
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::sync::{GpuFuture,FlushError};
use vulkano::sync;
use vulkano_win::VkSurfaceBuild;
use winit::{EventsLoop, WindowBuilder, Window, Event, WindowEvent };

use cgmath::{ Matrix4, Point3, Vector3, Rad};
fn build_instance () -> Result<Arc<Instance>,InstanceCreationError> {
    let app_info = ApplicationInfo{
        application_name: None,
        application_version: None,
        engine_name: None,
        engine_version: Some(Version{major: 1, minor: 0, patch: 0})};// Vulkan version
    let extensions = vulkano_win::required_extensions();
    Instance::new(Some(&app_info), &extensions, None)
}

fn choose_queue<T>(physical: &PhysicalDevice , surface: &Arc<Surface<T>>)->(Arc<Device>, Arc<Queue>) {

    let queue_family = physical.queue_families()
        .find( |&q|  {q.supports_graphics() && surface.is_supported(q).unwrap_or(false) } )
        .expect("Couldn't find a graphical queue family.");
    
    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions{
            khr_swapchain : true,
            .. vulkano::device::DeviceExtensions::none()
        };
        Device::new(*physical, physical.supported_features(), &device_ext,
                    [(queue_family, 0.5)].iter().cloned())
            .expect("Failed to create logical device.")
            
    };
    (device,queues.next().unwrap())

}


#[derive(Debug, Clone)]
struct Vertex {position: [f32; 3], color: [ f32;3]}


fn create_tetraeder_as_triangle_stripe(device: Arc<Device>) -> Arc<CpuAccessibleBuffer<[Vertex]>>
{
    impl_vertex!(Vertex, position, color);
    CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
        Vertex { position: [ -0.5 , -0.5, -0.5], color: [0.0, 0.0, 0.0]},
        Vertex { position: [ -0.5 ,  0.5,  0.5], color: [0.0, 1.0, 1.0]},
        Vertex { position: [  0.5 , -0.5,  0.5], color: [1.0, 1.0, 0.0]},
        Vertex { position: [  0.5 ,  0.5, -0.5], color: [1.0, 0.0, 0.0]},
        Vertex { position: [ -0.5 , -0.5, -0.5], color: [0.0, 0.0, 0.0]},
        Vertex { position: [ -0.5 ,  0.5,  0.5], color: [0.0, 1.0, 1.0]}
    ].iter().cloned()).unwrap()

}

#[derive(Debug,Clone)]
struct IndeterminableDimensionsError;

impl fmt::Display for IndeterminableDimensionsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Can't determine WindowDimensions")
    }
}

impl error::Error for IndeterminableDimensionsError{
    fn description(&self) -> &str {
        "invalid first item to double"
    }

    fn cause(&self) -> Option<&error::Error> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}

fn get_dimensions(window: &winit::Window) -> Result<[ u32;2],IndeterminableDimensionsError> {
    window.get_inner_size()
        .and_then(|dimensions|  {
            let idimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            Some([idimensions.0, idimensions.1])
        })
        .ok_or( IndeterminableDimensionsError)
}

fn create_swapchain (surface: &Arc<Surface<Window>>,
                     physical: &PhysicalDevice,
                     device: &Arc<Device>,
                     queue: &Arc<Queue>) -> Result<(Arc<Swapchain<Window>>,std::vec::Vec<Arc<SwapchainImage<Window>>>),
                                                   SwapchainCreationError> {

    let caps = surface.capabilities(*physical)
        .expect("Failed to get surface capabilities.");
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let format = caps.supported_formats[0].0;

    let initial_dimension = get_dimensions(surface.window())
        .expect("Couldn't determine window dimensions.");
    
    Swapchain::new(device.clone(),
                   surface.clone(),
                   caps.min_image_count,
                   format,
                   initial_dimension,
                   1,
                   caps.supported_usage_flags,
                   queue,
                   SurfaceTransform::Identity,
                   alpha,
                   PresentMode::Fifo,
                   true,
                   None)
}

fn main() {
    let instance =  build_instance()
        .expect("Failed to create instance");

    let mut events_loop = EventsLoop::new();

    let surface = WindowBuilder::new()
        .build_vk_surface(&events_loop,instance.clone())
        .expect("Failed to create a  Window");
    
    let physical = PhysicalDevice::enumerate(&instance).next()
        .expect("No device available");
    
    let (device,queue) = choose_queue(&physical, &surface);


    let (mut swapchain,images) = create_swapchain(&surface, &physical, &device, &queue)
        .expect("Failed to create swapchain");

    let vertex_buffer = create_tetraeder_as_triangle_stripe(device.clone());
    
    let mut dimension = get_dimensions(surface.window()).unwrap();
    
//    let proj = cgmath::perspective(cgmath::Rad(std::f32::consts::FRAC_PI_2), { dimension[0] as f32 / dimension[1] as f32 }, 0.01, 100.0);
//    let view = cgmath::Matrix4::look_at(cgmath::Point3::new(-1.0,  0.0, 0.0), cgmath::Point3::new(0.0, 0.0, 0.0), cgmath::Vector3::new(0.0, -1.0, 0.0));
    let scale = cgmath::Matrix4::from_scale(0.4);

    let uniform_buffer = vulkano::buffer::cpu_pool::CpuBufferPool::<vs::ty::ViewDescr>
        ::new(device.clone(), vulkano::buffer::BufferUsage::all());

    
    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();
    println!("{:?}", swapchain.format());
    let render_pass = Arc::new(single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            },
            depth: { 
                load: Clear,
                store: DontCare,
                format: Format::D32Sfloat,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    ).unwrap());


    
    let (mut pipeline, mut framebuffers) = window_size_dependent_setup(device.clone(), &vs, &fs, &images,
                                                                       render_pass.clone());


    let mut recreate_swapchain = false;
    let mut previous_frame_end = Box::new(sync::now(device.clone())) as  Box<GpuFuture>;
    
    let rotation_start = std::time::Instant::now();
    loop {
        previous_frame_end.cleanup_finished();
        
        if recreate_swapchain {
            dimension = get_dimensions(surface.window())
                .expect("Couldn't determine window dimensions.");
            
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimension){
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}",err)
            };
            
            swapchain = new_swapchain;

            let ( new_pipeline, new_framebuffers) = window_size_dependent_setup(device.clone(), &vs, &fs, &new_images,
                                                                                render_pass.clone());
            pipeline = new_pipeline;
            framebuffers = new_framebuffers;
            
            recreate_swapchain = false;
            
        }
        
        let uniform_buffer_subbuffer = {
            let elapsed  = rotation_start.elapsed();
            let rotation = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
            let rotation = cgmath::Matrix3::from_angle_y(cgmath::Rad(rotation as f32));
            
            let aspect_ratio = dimension[0] as f32 / dimension[1] as f32;
            let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
            let view = Matrix4::look_at(Point3::new(0.3, 0.3, 1.0), Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
            
            let uniform_data = vs::ty::ViewDescr {
                world: cgmath::Matrix4::from(rotation).into(),
                view: (view * scale).into(),
                proj: proj.into(),
            };
            
            uniform_buffer.next(uniform_data).unwrap()
        };
        
        let set = Arc::new(vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(pipeline.clone(), 0)
                           .add_buffer(uniform_buffer_subbuffer).unwrap()
                           .build().unwrap()
        );
        
        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                },
                Err(err) => panic!("{:?}",err)
            };
        let clear_values = vec![[0.0, 0.5, 0.5, 1.0].into(),ClearValue::Depth(1.0)];

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            .draw(pipeline.clone(), &DynamicState::none(), vec!(vertex_buffer.clone()), set.clone(), () )
            .unwrap()
            .end_render_pass()
            .unwrap()
            .build()
            .unwrap();
        
        let future = previous_frame_end
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {previous_frame_end = Box::new(future) as Box<_>;},
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new( sync::now(device.clone() ) ) as Box<_>;
            }
            Err(e) => {
               println!("{:?}",e);
                previous_frame_end = Box::new( sync::now(device.clone() ) ) as Box<_>;
            }
        }
        
        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                Event::WindowEvent {event: WindowEvent::CloseRequested, ..} => done = true,
                Event::WindowEvent {event: WindowEvent::Resized(_), ..} => recreate_swapchain = true,               
                _ => ()
            }
        });
        if done {return ;}
    }
}


fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &vs::Shader,
    fs: &fs::Shader,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPassAbstract+Send+Sync>)
    ->(Arc<GraphicsPipelineAbstract+ Send+ Sync>, Vec<Arc<FramebufferAbstract + Send + Sync> >)
{
    let dimensions = images[0].dimensions();
    let depth_buffer = AttachmentImage::new(device.clone(), dimensions, Format::D32Sfloat).unwrap();
    let framebuffers = images.iter().map(|image|{
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .add(depth_buffer.clone()).unwrap()
                .build().unwrap()
        )as Arc<FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>();
    
    let pipeline = Arc::new(GraphicsPipeline::start()
                            .vertex_input(SingleBufferDefinition::<Vertex>::new())
                            .vertex_shader(vs.main_entry_point(), ())
                            .triangle_strip()
                            .viewports_dynamic_scissors_irrelevant(1)
                            .viewports(iter::once(Viewport {
                                origin: [0.0, 0.0 ],
                                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                                depth_range: 0.0 .. 1.0, }
                            ))
                            .fragment_shader(fs.main_entry_point(), ())
                            .depth_stencil_simple_depth()
                            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                            .build(device.clone())
                            .expect("Pipeline creation failed")) as Arc<GraphicsPipelineAbstract+ Send+ Sync>;
    (pipeline, framebuffers)
}




mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/shader/vert.glsl"
    }
}

mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/shader/frag.glsl" 
    }
}
