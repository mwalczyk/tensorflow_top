/* Shared Use License: This file is owned by Derivative Inc. (Derivative) and
* can only be used, and/or modified for use, in conjunction with 
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement (which
* also govern the use of this file).  You may share a modified version of this
* file with another authorized licensee of Derivative's TouchDesigner software.
* Otherwise, no redistribution or sharing of this file, with or without
* modification, is permitted.
*/

#include "TensorFlowTOP.h"

extern "C"
{
	DLLEXPORT TOP_PluginInfo GetTOPPluginInfo(void)
	{
		TOP_PluginInfo info;
		info.apiVersion = TOPCPlusPlusAPIVersion;
		info.executeMode = TOP_ExecuteMode::OpenGL_FBO;
		return info;
	}

	DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
	{
		return new TensorFlowTOP(info, context);
	}

	DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
	{
		delete (TensorFlowTOP*)instance;
	}
};

Status convertPixelsToTensor(std::vector<Tensor>* out_tensors,
							 uint8_t* pixels,
							 int pixels_width, 
							 int pixels_height, 
							 int pixels_channels,
							 const int expected_height, 
							 const int expected_width, 
							 const int expected_channels = 3,
							 const float expected_mean = 128,
							 const float expected_standard_dev = 128) 
{
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops; 

	// Create the input tensor.
	tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({pixels_height, pixels_width, expected_channels}));
	auto input_tensor_mapped = input_tensor.tensor<float, 3>();

	// Copying the pixel data into the tensor.
	for (int y = 0; y < pixels_height; ++y) 
	{
		const uint8_t* source_row = pixels + (y * pixels_width * pixels_channels);
		for (int x = 0; x < pixels_width; ++x) 
		{
			const uint8_t* source_pixel = source_row + (x * pixels_channels);

			// OpenGL texture data is read as BGRA, so we want to 
			// ignore the alpha channel and reorder the other 3
			// color channels
			for (int c = 0; c < expected_channels; ++c)
			{
				const uint8_t* source_value = source_pixel + ((expected_channels - 1) - c);
				float pixel = static_cast<float>(*source_value);

				input_tensor_mapped(y, x, c) = pixel;
			}
		}
	}

	const std::string output_name = "normalized";
	auto float_caster = Cast(root.WithOpName("float_caster"), input_tensor, tensorflow::DT_FLOAT);
	auto dims_expander = ExpandDims(root, float_caster, 0);
	auto resized = ResizeBilinear(root, dims_expander, Const(root.WithOpName("size"), {expected_height, expected_width}));
	Div(root.WithOpName(output_name), Sub(root, resized, {expected_mean}), {expected_standard_dev});

	// Run the graph.
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));

	return Status::OK();
}

GLuint createGlslProgram(const std::string& vertSrc, const std::string& fragSrc)
{
	// Vertex shader
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	const char* vertSrcPtr = vertSrc.c_str();
	glShaderSource(vertexShader, 1, &vertSrcPtr, nullptr);
	glCompileShader(vertexShader);

	// Check for compile time errors
	GLint success;
	GLchar infoLog[512];
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "Error compiling vertex shader:\n" << infoLog << std::endl;
	}

	// Fragment shader
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	const char* fragSrcPtr = fragSrc.c_str();
	glShaderSource(fragmentShader, 1, &fragSrcPtr, NULL);
	glCompileShader(fragmentShader);

	// Check for compile time errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "Error compiling fragment shader:\n" << infoLog << std::endl;
	}

	// Link shaders
	GLuint program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);

	// Check for linking errors
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) 
	{
		glGetProgramInfoLog(program, 512, NULL, infoLog);
		std::cout << "Error linking program:\n" << infoLog << std::endl;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return program;
}

TensorFlowTOP::TensorFlowTOP(const OP_NodeInfo* info, TOP_Context* context)
{
#ifdef WIN32
	static bool needGLEWInit = true;
	if (needGLEWInit)
	{
		needGLEWInit = false;
		context->beginGLCommands();
		glewInit();
		context->endGLCommands();
	}
#endif

	const GLchar* vertexShaderSource =
	R"(
		#version 430 core								
		
		vec2 positions[6] = vec2[]( vec2(-1.0, -1.0),	// lower-left
                                    vec2( 1.0, -1.0),	// lower-right
									vec2( 1.0,  1.0),	// upper-right
			
									vec2(-1.0, -1.0),	// lower-left
                                    vec2( 1.0,  1.0),	// upper-right
									vec2(-1.0,  1.0));	// upper-left
		
		out vec2 uv;

		void main()									
		{			
			uv = positions[gl_VertexID] * 0.5 + 0.5;
									
			gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
		}
	)";

	const GLchar* fragmentShaderSource =
	R"(
		#version 430 core	
		
		layout(binding = 0) uniform sampler2D u_input;
		in vec2 uv;

		layout(location = 0) out vec4 o_color;		
						
		void main()									
		{												
			o_color = texture(u_input, uv);			
		}
	)";

	program = createGlslProgram(vertexShaderSource, fragmentShaderSource);

	glCreateVertexArrays(1, &vao);

	std::string graph_path = "C:/Users/michael.walczyk/Desktop/tensorflow_top/models/inception.pb";
	std::cout << "Attempting to load graph file...\n";

	tensorflow::GraphDef graph_def;
	if (!ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def).ok()) 
	{
		std::cout << "Failed to read protobuffer file\n";
	}
	std::cout << "Attempting to start session...\n";

	tensorflow::SessionOptions sess_opt;
	sess_opt.config.mutable_gpu_options()->set_allow_growth(true);
	(&session)->reset(tensorflow::NewSession(sess_opt));

	if (!session->Create(graph_def).ok()) {
		std::cout << "Failed to create graph\n";
	}
}

TensorFlowTOP::~TensorFlowTOP()
{
}

void TensorFlowTOP::getGeneralInfo(TOP_GeneralInfo* ginfo)
{
	ginfo->cookEveryFrame = false;
	ginfo->cookEveryFrameIfAsked = false;
}

bool TensorFlowTOP::getOutputFormat(TOP_OutputFormat* format)
{
	return false;
}

void TensorFlowTOP::execute(const TOP_OutputFormatSpecs* outputFormat, OP_Inputs* inputs, TOP_Context *context)
{
	auto topInput = inputs->getInputTOP(0);

	if (topInput)
	{	
		// Draw the input texture.
		context->beginGLCommands();
		{
			glViewport(0, 0, outputFormat->width, outputFormat->height);
			glClearColor(0.0, 0.0, 0.0, 0.0);
			glClear(GL_COLOR_BUFFER_BIT);

			glBindTextureUnit(0, topInput->textureIndex);

			glUseProgram(program);
			glBindVertexArray(vao);
			glDrawArrays(GL_TRIANGLES, 0, 6);
		}
		context->endGLCommands();

		// Tensors are interpretted top-down, so we need to flip the pixels here.
		OP_TOPInputDownloadOptions options;
		options.verticalFlip = true;

		// Read pixels from GPU -> CPU.
		uint8_t* pixels = static_cast<uint8_t*>(inputs->getTOPDataInCPUMemory(topInput, &options));

		// Per the TouchDesigner documentation, the pointer returned above might be `null` sometimes...
		if (pixels != nullptr)
		{
			// This is specific to the `inception` network and will need to be determined automatically 
			// in the future.
			const int32 expected_dims = 299;

			std::vector<Tensor> inputs;

			if (!convertPixelsToTensor(&inputs, pixels, outputFormat->width, outputFormat->height, 4, expected_dims, expected_dims).ok()) 
			{
				std::cout << "Failed to convert pixels to tensor\n";
			}

			// Run the session and collect output tensors.
			std::vector<Tensor> outputs;
			string input_layer = "Mul";
			string output_layer = "softmax";
			if (!session->Run({{input_layer, inputs[0]}}, {output_layer}, {}, &outputs).ok()) 
			{
				std::cout << "Failed to run model\n";
			}

			// Grab the index of the class with the highest score.
			Eigen::Map<Eigen::VectorXf> pred(outputs[0].flat<float>().data(), outputs[0].NumElements());
			int maxIndex; 
			float maxValue = pred.maxCoeff(&maxIndex);

			std::cout << "Class with highest probability: " << classNames[maxIndex] << ", " << maxValue << "\n";
		}
	}
}

int32_t TensorFlowTOP::getNumInfoCHOPChans()
{
	return 0;
}

void TensorFlowTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan)
{
}

bool TensorFlowTOP::getInfoDATSize(OP_InfoDATSize* infoSize)
{
	return true;
}

void TensorFlowTOP::getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries* entries)
{
}

void TensorFlowTOP::setupParameters(OP_ParameterManager* manager)
{
}

void TensorFlowTOP::pulsePressed(const char* name)
{
}

