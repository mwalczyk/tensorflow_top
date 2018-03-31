#pragma once

static const char* vertShaderSrc =
R"(#version 430 core								
		
vec2 positions[6] = vec2[]( vec2(-1.0, -1.0),	// lower-left
                            vec2( 1.0, -1.0),	// lower-right
							vec2( 1.0,  1.0),	// upper-right
			
							vec2(-1.0, -1.0),	// lower-left
                            vec2( 1.0,  1.0),	// upper-right
							vec2(-1.0,  1.0));	// upper-left
		
out VS_OUT 
{
	vec2 uv;
} vs_out;

void main()									
{			
	vs_out.uv = positions[gl_VertexID] * 0.5 + 0.5;
									
	gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
})";

static const char* fragShaderSrc =
R"(#version 430 core	
		
layout(binding = 0) uniform sampler2D u_input;
		
in VS_OUT 
{
	vec2 uv;
} fs_in;

layout(location = 0) out vec4 o_color;		
						
void main()									
{												
	o_color = texture(u_input, fs_in.uv);			
})";