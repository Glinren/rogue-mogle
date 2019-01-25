#[macro_use]


pub mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/resources/shader/vert.glsl"
    }
}

pub mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/resources/shader/frag.glsl" 
    }
}
