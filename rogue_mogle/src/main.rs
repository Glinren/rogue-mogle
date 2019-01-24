#[macro_use]
extern crate vulkano;

extern crate vulkano_win;
extern crate winit;
extern crate vulkano_shaders;


use std::sync::Arc;
use std::option::Option;

use vulkano::buffer::{CpuAccessibleBuffer,BufferUsage};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, Queue};
use vulkano::framebuffer::{Subpass, Framebuffer,
                           RenderPassAbstract, FramebufferAbstract};
use vulkano::instance::{Instance,ApplicationInfo,Version,InstanceCreationError};
use vulkano::instance::PhysicalDevice;
use vulkano::image::SwapchainImage;
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError,
                         Surface, Swapchain, SurfaceTransform,
                         PresentMode, SwapchainCreationError};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::sync::{GpuFuture,FlushError};
use vulkano::sync;
use vulkano_win::VkSurfaceBuild;
use winit::{EventsLoop, WindowBuilder, Window, Event, WindowEvent };


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

fn create_swapchain<T> (surface: &Arc<Surface<T>>,
                        physical: &PhysicalDevice,
                        device: &Arc<Device>,
                        queue: &Arc<Queue>,
                        window: &winit::Window) -> (Arc<Swapchain<T>>,std::vec::Vec<Arc<SwapchainImage<T>>>) {
    let caps = surface.capabilities(*physical)
        .expect("Failed to get surface capabilities.");
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let format = caps.supported_formats[0].0;

    let initial_dimension = if let Some(dimensions) = window.get_inner_size() {
        let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
        [dimensions.0, dimensions.1]
    } else {
        [1024,1024]
    };
//    println!("{:?}", initial_dimension);
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
        .expect("Failed to create swapchain")
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

    let window = surface.window();

    let (mut swapchain,images) = create_swapchain(&surface, &physical, &device, &queue, window);

    let vertex_buffer = {
        #[derive(Debug, Clone)]
        struct Vertex {position: [f32; 3]}
        impl_vertex!(Vertex, position);
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
            Vertex { position: [ 0.0 ,  0.0 , 0.0]},
            Vertex { position: [ 0.0 ,  1.0 , 1.0]},
            Vertex { position: [ 1.0,   1.0 , 0.0]},
            Vertex { position: [ 1.0 ,  0.0 , 1.0]}
        ].iter().cloned()).unwrap()
    };

    let render_pass = Arc::new(single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap());


    

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();
      
    
    let pipeline = Arc::new(GraphicsPipeline::start()
                            .vertex_input_single_buffer()
                            .vertex_shader(vs.main_entry_point(), ())
                            .triangle_list()
                            .viewports_dynamic_scissors_irrelevant(1)
                            .fragment_shader(fs.main_entry_point(),())
                            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                            .build(device.clone())
                            .expect("Pipeline creation failed"));

    let mut dynamic_state = DynamicState{line_width: None, viewports:None, scissors:None };
    
    let mut framebuffers = window_size_dependent_setup( &images, render_pass.clone(), &mut dynamic_state);


    let mut recreate_swapchain = false;

    let mut previous_frame_end = Box::new(sync::now(device.clone())) as  Box<GpuFuture>;
    
    loop {
        previous_frame_end.cleanup_finished();
        if recreate_swapchain {
            let dimension = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(
                    window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return ;
            };
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimension){
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}",err)
            };
            swapchain = new_swapchain;

            framebuffers = window_size_dependent_setup(
                &new_images,
                render_pass.clone(),
                &mut dynamic_state);
            
            recreate_swapchain = false;
        }

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                },
                Err(err) => panic!("{:?}",err)
            };
        
        
        let clear_values = vec!([1.0, 1.0, 1.0, 1.0].into());

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
            device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
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
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPassAbstract+Send+Sync>,
    dynamic_state: &mut DynamicState)
    -> Vec<Arc<FramebufferAbstract + Send + Sync> >
{
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0 ],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));
        
    images.iter().map(|image|{
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap()
        )as Arc<FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
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
