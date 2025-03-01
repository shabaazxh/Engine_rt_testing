
#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D renderedScene;

void main()
{
	vec4 renderedScene = texture(renderedScene, uv);
	fragColor = vec4(vec3(renderedScene), 1.0);
}