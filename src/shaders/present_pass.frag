
#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform PostProcessSettings
{
	bool Enable;
}ppSettings;

layout(set = 0, binding = 1) uniform sampler2D composited_result; // this is the shading result with gamma correction
layout(set = 0, binding = 2) uniform sampler2D accumulated_result; // this is temporal accumulated result to provide a ground truth

void main()
{
	// If enabled use the accumulated result
	if(ppSettings.Enable)
	{
		vec3 scene = texture(accumulated_result, uv).rgb;
		vec3 ldrColor = scene.rgb / (scene.rgb + vec3(1.0));
		vec3 gammaCorrectedColor = pow(ldrColor, vec3(1.0 / 2.2));
		fragColor = vec4(gammaCorrectedColor, 1.0);
	} else
	{
		vec3 scene = texture(composited_result, uv).rgb;
		vec3 ldrColor = scene.rgb / (scene.rgb + vec3(1.0));
		vec3 gammaCorrectedColor = pow(ldrColor, vec3(1.0 / 2.2));
		fragColor = vec4(gammaCorrectedColor, 1.0);
	}

}

/// I have removed the directional light source in Renderer.cpp, add it back in and remove 1 point light to get it back