#version 450

layout(location = 0) out vec2 outUV;


// Reference: Full screen large triangle : 
// Sascha (2016). Vulkan tutorial on rendering a fullscreen quad without buffers. [online] Sascha Willems. 
// Available at: https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/ 
void main()
{
	outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(outUV * 2.0f + -1.0f, 0.0f, 1.0f);
}
