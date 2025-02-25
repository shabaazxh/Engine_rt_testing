#version 460
#extension GL_EXT_ray_tracing : enable

struct RayPayLoad
{
    vec3 pos;
    vec3 dir;
    vec3 colour;
    vec3 hitnormal;
    vec3 hitpos;
    float hit;
};

layout(location = 2) rayPayloadInEXT RayPayLoad rayPayLoad;

void main()
{
    rayPayLoad.colour = vec3(0.5, 0.75, 1.0);
    rayPayLoad.hit = -1.0f;
    rayPayLoad.pos = vec3(0.0);
    rayPayLoad.hitpos = vec3(0.0);
    rayPayLoad.hitnormal = vec3(0.0);
    rayPayLoad.dir = vec3(0.0);
}