
#version 450

layout (set=0, binding = 0) uniform ViewDescr{
     mat4 world;
     mat4 view;
     mat4 proj;     
} view_descr;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec4 vertex_color;

void main(){
  mat4  worldview =  view_descr.view *  view_descr.world;
  vec4 pos = vec4(position.x*0.8-0.4,
		  position.y*0.8-0.4,
		  position.z*0.8-0.4, 1.0);
  gl_Position = view_descr.proj * worldview * pos ;

  vertex_color = vec4(color, 1.0);
}
