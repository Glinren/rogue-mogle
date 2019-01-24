
#version 450

layout(location = 0) in vec3 position;

layout(location = 0) out vec4 vertex_color;

void main(){
  gl_Position = vec4(position.x*0.8-0.4,
		     position.y*0.8-0.4,
		     position.z*0.8-0.4, 1.0);
  vertex_color = vec4(position, 1.0);
}
