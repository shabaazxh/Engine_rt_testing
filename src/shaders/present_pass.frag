
#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform PostProcessSettings
{
	bool Enable;
}ppSettings;

layout(set = 0, binding = 1) uniform sampler2D renderedScene;

// Reference: Implementation and Learnings based on:
// https://www.geeks3d.com/20101029/shader-library-pixelation-post-processing-effect-glsl/

void main()
{	
	if(ppSettings.Enable) {
		vec2 texelSize = 1.0 / textureSize(renderedScene, 0); // get size of a single texel
		vec2 blockSize = vec2(5,3); // block size in pixels
		vec2 blockSizeInUV = blockSize * texelSize; // size of the block in textured-uv space

		// uv to "block space"
		vec2 blockSpaceUV = uv / blockSizeInUV;
		vec2 snappedBlockSpaceUV = ceil(blockSpaceUV);
		vec2 blockInUVSpace = snappedBlockSpaceUV * blockSizeInUV;

		// flip signs if snapping to nearest block instead of next block
		blockInUVSpace -= 0.5 * blockSizeInUV; // get the center 

		vec3 sampledColor = texture(renderedScene, blockInUVSpace).rgb;

		fragColor = vec4(sampledColor, 1.0);
	} else {

		vec3 color = texture(renderedScene, uv).rgb;
		fragColor = vec4(color, 1.0);
	}

}