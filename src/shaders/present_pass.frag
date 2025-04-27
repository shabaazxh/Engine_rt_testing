
#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform PostProcessSettings
{
	bool Enable;
}ppSettings;

layout(set = 0, binding = 1) uniform sampler2D renderedScene;
layout(set = 0, binding = 2) uniform sampler2D nonTemporalScene;

// Reference: Implementation and Learnings based on:
// https://www.geeks3d.com/20101029/shader-library-pixelation-post-processing-effect-glsl/
const vec2 pixelSize = vec2(1.0 / 1280.0, 1.0 / 720.0);

void main()
{
	if(ppSettings.Enable) {
		vec3 color = texture(renderedScene, uv).rgb;
		vec4 denoised = vec4(0.0);
		float kernel[9] = float[](
			1.0/16.0, 2.0/16.0, 1.0/16.0,  // Row 1
			2.0/16.0, 4.0/16.0, 2.0/16.0,  // Row 2
			1.0/16.0, 2.0/16.0, 1.0/16.0   // Row 3
		);
		vec2 offsets[9] = vec2[](
			vec2(-pixelSize.x, -pixelSize.y), vec2(0.0, -pixelSize.y), vec2(pixelSize.x, -pixelSize.y),
			vec2(-pixelSize.x, 0.0),          vec2(0.0, 0.0),          vec2(pixelSize.x, 0.0),
			vec2(-pixelSize.x, pixelSize.y),  vec2(0.0, pixelSize.y),  vec2(pixelSize.x, pixelSize.y)
		);
		for (int i = 0; i < 9; i++) {
			denoised += texture(renderedScene, uv + offsets[i]) * kernel[i];
		}
		fragColor = vec4(denoised.xyz, 1.0);

	} else {
		vec3 scene = texture(nonTemporalScene, uv).rgb;

		vec3 ldrColor = scene.rgb / (scene.rgb + vec3(1.0));
		vec3 gammaCorrectedColor = pow(ldrColor, vec3(1.0 / 2.2));
		fragColor = vec4(gammaCorrectedColor, 1.0);
	}

}

/// I have removed the directional light source in Renderer.cpp, add it back in and remove 1 point light to get it back