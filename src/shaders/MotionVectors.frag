#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D depth_buffer;

layout(set = 0, binding = 1) uniform CurrentCameraTransform
{
	mat4 model;
	mat4 view;
	mat4 projection;
    vec4 cameraPosition;
    vec2 viewportSize;
	float fov;
	float nearPlane;
	float farPlane;
} current_camera_transform;

layout(set = 0, binding = 2) uniform PreviousCameraTransform
{
	mat4 model;
	mat4 view;
	mat4 projection;
    vec4 cameraPosition;
    vec2 viewportSize;
	float fov;
	float nearPlane;
	float farPlane;
} previous_camera_transform;


vec3 depth_to_world_position()
{
	float depth = texture(depth_buffer, uv).r;
	vec4 clip_space_pos = vec4(uv * 2.0 - 1.0, depth, 1.0);
	// Convert from clip space to view space
	vec4 view_space_pos = inverse(current_camera_transform.projection) * vec4(clip_space_pos.xyz, 1.0);
	view_space_pos.xyz /= view_space_pos.w;
	vec4 world_space_pos = inverse(current_camera_transform.view) * vec4(view_space_pos.xyz, 1.0);
	return world_space_pos.xyz;
}

vec2 world_to_screen_space_uv(vec3 world_pos, mat4 projection, mat4 view)
{
	vec4 project = projection * view * vec4(world_pos, 1.0);
	project.xyz /= project.w;
	return project.xy * 0.5 + 0.5;
}

void main()
{
	//vec3 current_world_pos = texture(WorldHitPositions, uv).xyz;

	vec3 current_world_pos = depth_to_world_position();
	vec2 prev_uv = world_to_screen_space_uv(current_world_pos, previous_camera_transform.projection, previous_camera_transform.view);

	vec2 curr_uv = uv;
	vec2 motion_vector = curr_uv - prev_uv;

	// @TODO: should store this in a RG texture not RGBA
	fragColor = vec4(motion_vector, 0.0, 1.0);
}

// This should now be using past depth buffer or world pos from raster
// not from the ray tracing