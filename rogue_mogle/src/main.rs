#[macro_use]
extern crate vulkano;
extern crate vulkano_win;
extern crate vulkano_shaders;

extern crate cgmath;
extern crate winit;


mod resources;


use std::sync::Arc;
use std::option::Option;
use std::fmt;
use std::error;


use vulkano::buffer::{CpuAccessibleBuffer,BufferUsage};
use vulkano::device::{Device, Queue};
//use vulkano::format::Format;
use vulkano::instance::{Instance,ApplicationInfo,Version,InstanceCreationError};
use vulkano::instance::PhysicalDevice;
use vulkano::image::SwapchainImage;
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError,
                         Surface, Swapchain, SurfaceTransform,
                         PresentMode, SwapchainCreationError};
//use vulkano::pipeline::viewport::Viewport;
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

// TODO Glinren hide the CpuAccessible buffer in the renderer
#[derive(Debug, Clone)]
pub struct Vertex {position: [f32; 3], color: [ f32;3]}


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

fn get_dimensions(surface: &Arc<Surface<winit::Window>>) -> Result<[ u32;2],IndeterminableDimensionsError> {
    let window = surface.window();
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
                     queue: &Arc<Queue>) -> Result<(Arc<Swapchain<Window>>,
                                                    std::vec::Vec<Arc<SwapchainImage<Window>>>),
                                                   SwapchainCreationError> {

    let caps = surface.capabilities(*physical)
        .expect("Failed to get surface capabilities.");
    let alpha = caps.supported_composite_alpha.iter().next().unwrap();
    let format = caps.supported_formats[0].0;

    let initial_dimension = get_dimensions(&surface)
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


mod render {
    use resources;


    use cgmath::Matrix4 ;
    use std::sync::Arc;
    use vulkano::device::{Device, Queue};
    use std::fmt;
    use std::error;
    use std::iter;
    
    use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
    use vulkano::format::ClearValue;
    use vulkano::memory::pool::StdMemoryPool;
    use vulkano::image::ImageAccess;
    use vulkano::format::Format;

    use vulkano::swapchain::Surface;
    use vulkano::sync::GpuFuture;
    use vulkano::pipeline::viewport::Viewport;
    use vulkano::pipeline::{GraphicsPipeline,GraphicsPipelineAbstract};
    use vulkano::pipeline::vertex::SingleBufferDefinition;
    pub use Vertex;
    use vulkano::image::attachment::AttachmentImage;
    use vulkano::framebuffer::{Subpass, Framebuffer,
                               RenderPassAbstract, FramebufferAbstract};
    use resources::vs;
    use resources::fs;
    use vulkano::image::SwapchainImage;
    use winit::Window;
    // use main::scope::CustomRenderPassDescr;
    // #[derive(Debug)]
    // struct Shader {
    //     vs: vs::Shader;
    // }
    #[derive(Debug,Clone)]
    pub struct RendererInitializationError;

    impl fmt::Display for RendererInitializationError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Couldn't initialize the Renderer.")
        }
    }

    impl error::Error for RendererInitializationError{
    fn description(&self) -> &str {
        "couldn't initialize the Renderer"
    }

    fn cause(&self) -> Option<&error::Error> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}

    pub struct RendererBuilder {
        graphics_queue: Arc<Queue>,
    }
        
    
    fn check_enabled_extensions(device: &Arc<Device>)-> bool {
        let required_ext = vulkano::device::DeviceExtensions{
            khr_swapchain : true,
            .. vulkano::device::DeviceExtensions::none()
        };
        device.loaded_extensions().intersection( &required_ext) == required_ext
    }
    

    fn check_graphics_queue_properties(queue: &Arc<Queue>)-> bool{
        queue.family().supports_graphics()
    }

    // TODO(Glinren) Builder can be eliminated
    impl RendererBuilder {
        pub fn new(graphics_queue: Arc<Queue>)->Result<RendererBuilder, RendererInitializationError> {
            // TODO(Glinren) be more specific with the error
            // The error should contain enough information to understand what precondition was violated
            if  check_enabled_extensions(graphics_queue.device())
                && check_graphics_queue_properties(&graphics_queue){
                    Ok(RendererBuilder{
                        graphics_queue: graphics_queue,
                    })
                } else { Err(RendererInitializationError)}
        }


        // pub fn build_for_render_targets<T: ImageAccess + Sync + Send>
        pub fn build_for_render_targets
            (self, images: std::vec::Vec<Arc<SwapchainImage<Window>>>) -> Result<Renderer, RendererInitializationError>{
                Renderer::new(self.graphics_queue, images)
        }
    }

    fn create_render_pass(device: Arc<Device>,
                          format: Format)-> Result<Arc<RenderPassAbstract+Send+Sync>,
                                                   RendererInitializationError>{
        Ok(Arc::new(single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: format,
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
        ).unwrap()))
    }

    //fn setup_pipeline<T: ImageAccess + Sync + Send+ SafeDeref>(
    fn setup_pipeline(
        device: Arc<Device>,
        vs: &resources::vs::Shader,
        fs: &resources::fs::Shader,
        images: &std::vec::Vec<Arc<SwapchainImage<Window>>>,
        render_pass: &Arc<RenderPassAbstract+Send+Sync>)
        ->(Arc<GraphicsPipelineAbstract+ Send+ Sync>, Vec<Arc<FramebufferAbstract + Send + Sync> >)
    {
        let dimensions = images[0].dimensions();
        let dimensions = [dimensions.width(), dimensions.height()];
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



    pub struct Renderer {
        graphics_queue: Arc<Queue>,
        render_pass: Arc<RenderPassAbstract+Send+Sync>,
        vertex_shader: resources::vs::Shader,
        fragment_shader: resources::fs::Shader,
        format: Format,
        pub pipeline:  Arc<GraphicsPipelineAbstract+ Send+ Sync>,
        pub framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
        uniform_buffer: vulkano::buffer::CpuBufferPool<vs::ty::ViewDescr>
    }

    
    impl Renderer{
        //        pub fn new<T: ImageAccess + Sync + Send>
        pub fn new
            (queue: Arc<Queue>,
             images: std::vec::Vec<Arc<SwapchainImage<Window>>>
            ) ->Result<Renderer,RendererInitializationError> {
                let device = queue.device().clone();
                let vs = vs::Shader::load(device.clone()).unwrap() ;
                let fs = fs::Shader::load(device.clone()).unwrap() ;
                if let Ok(render_pass ) = create_render_pass(device.clone(), images[0].format()){
                    let (pipeline, framebuffers ) = setup_pipeline(device.clone(),&vs, &fs,&images, &render_pass); 
                    Ok(Renderer{
                        graphics_queue: queue,
                        vertex_shader: vs,
                        fragment_shader: fs,
                        format: images[0].format(),
                        render_pass: render_pass,
                        pipeline: pipeline,
                        framebuffers: framebuffers,
                        uniform_buffer: vulkano::buffer::cpu_pool::CpuBufferPool::<vs::ty::ViewDescr>::new(device.clone(), vulkano::buffer::BufferUsage::all()),
                    })
                } else { Err(RendererInitializationError)}

            }
        
        pub fn render(&self,
                      world: Matrix4<f32>,
                      view: Matrix4<f32>,
                      proj: Matrix4<f32>,
                      target_num: usize,
                      vertex_buffer: &Arc<vulkano::buffer::CpuAccessibleBuffer<[Vertex]>>,
                      future: vulkano::sync::JoinFuture<Box<dyn vulkano::sync::GpuFuture>, vulkano::swapchain::SwapchainAcquireFuture<winit::Window>>)
                      ->vulkano::command_buffer::CommandBufferExecFuture<
                vulkano::sync::JoinFuture<
                        Box<dyn vulkano::sync::GpuFuture>,
                    vulkano::swapchain::SwapchainAcquireFuture<winit::Window>
                        >,
            vulkano::command_buffer::AutoCommandBuffer>
        {
            let set = Arc::new(
                vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(
                    self.pipeline.clone(),0)
                    .add_buffer(self.create_subbuffer(world,view,proj).unwrap() ).unwrap()
                    .build().unwrap()
            );

            let command_buffer = {
                let clear_values = vec![[0.0, 0.5, 0.5, 1.0].into(),ClearValue::Depth(1.0)];
                
                AutoCommandBufferBuilder::primary_one_time_submit(
                    self.graphics_queue.device().clone(), self.graphics_queue.family()).unwrap()
                    .begin_render_pass(self.framebuffers[target_num].clone(), false, clear_values)
                    .unwrap()
                    .draw(self.pipeline.clone(),
                          &DynamicState::none(),
                          vec!(vertex_buffer.clone()), set.clone(), () )
                    .unwrap()
                    .end_render_pass()
                    .unwrap()
                    .build()
                    .unwrap()
            };
            future
              .then_execute(self.graphics_queue.clone(), command_buffer)
                .unwrap()
        }
        
        fn create_subbuffer(&self, world: Matrix4<f32>, view: Matrix4<f32>, proj: Matrix4<f32>)
                       -> Result<vulkano::buffer::cpu_pool::CpuBufferPoolSubbuffer<resources::vs::ty::ViewDescr,Arc<StdMemoryPool>>, vulkano::memory::DeviceMemoryAllocError>
         {
             let uniform_data = vs::ty::ViewDescr {
                 world: world.into(),
                 view: view.into(),
                 proj: proj.into(),
                
             };
             self.uniform_buffer.next(uniform_data)
         }

    

        pub fn can_render_to<T>(&self, surface: &Arc<Surface<T>> )-> bool {
            surface.is_supported(self.graphics_queue.family()).unwrap_or(false) 
        }

        pub fn  set_render_targets(
            &mut self, images: std::vec::Vec<Arc<SwapchainImage<Window>>>)
            ->Result<(),RendererInitializationError >{
            if self.format != images[0].format() {
                self.render_pass = create_render_pass(self.graphics_queue.device().clone(), images[0].format())
                    .expect(" problem creating render pass");
                self.format = images[0].format();
            }
            let (pipeline, framebuffers ) = setup_pipeline(
                self.graphics_queue.device().clone(),
                &self.vertex_shader, &self.fragment_shader, &images, &self.render_pass);
            self.pipeline = pipeline;
            self.framebuffers = framebuffers;
            Ok(())
        }
    }
}

fn main() {
    let instance =  build_instance()
        .expect("Failed to create instance");

    let mut events_loop = EventsLoop::new();

    let surface = WindowBuilder::new()
        .build_vk_surface(&events_loop,instance.clone())
        .expect("Failed to create a  Window");
    
    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("No device available");
    
    let (device,queue) = choose_queue(&physical, &surface);


    let vertex_buffer = create_tetraeder_as_triangle_stripe(device.clone());
    
    let mut dimension = get_dimensions(&surface).unwrap();

    
    let (mut swapchain,images) = create_swapchain(&surface, &physical, &device, &queue)
        .expect("Failed to create swapchain");

    let mut renderer = render::RendererBuilder::new(queue.clone())
        .unwrap()
        .build_for_render_targets(images)
        .unwrap();

    if ! renderer.can_render_to(&surface) {
        return
    }


    let scale = cgmath::Matrix4::from_scale(0.8);



    let mut recreate_swapchain = false;
    let mut previous_frame_end = Box::new(sync::now(device.clone())) as  Box<GpuFuture>;
    
    let rotation_start = std::time::Instant::now();

    loop {
        previous_frame_end.cleanup_finished();
        
        if recreate_swapchain {
            dimension = get_dimensions(&surface)
                .expect("Couldn't determine window dimensions.");
            
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimension){
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}",err)
            };
            
            swapchain = new_swapchain;
            renderer.set_render_targets(new_images).expect("Error");
            
            recreate_swapchain = false;
            
        }
        
        let elapsed  = rotation_start.elapsed();
        let rotation = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
        let rotation = cgmath::Matrix3::from_angle_y(cgmath::Rad(rotation as f32));
            
        let aspect_ratio = dimension[0] as f32 / dimension[1] as f32;
        let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
        let view = Matrix4::look_at(Point3::new(1.0, 1.0, 1.0), Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0));
            

            
        
        
        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                },
                Err(err) => panic!("{:?}",err)
            };
        

        
        let future = renderer.render(cgmath::Matrix4::from(rotation).into(),
                                             (view * scale).into(),
                                             proj.into(),
                                             image_num,
                                         &vertex_buffer,
                                         previous_frame_end.join(acquire_future))
            .then_swapchain_present(queue.clone(),
                                    swapchain.clone(), image_num)
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
