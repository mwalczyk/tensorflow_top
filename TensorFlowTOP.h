/* Shared Use License: This file is owned by Derivative Inc. (Derivative) and
 * can only be used, and/or modified for use, in conjunction with 
 * Derivative's TouchDesigner software, and only if you are a licensee who has
 * accepted Derivative's TouchDesigner license or assignment agreement (which
 * also govern the use of this file).  You may share a modified version of this
 * file with another authorized licensee of Derivative's TouchDesigner software.
 * Otherwise, no redistribution or sharing of this file, with or without
 * modification, is permitted.
 */

#include "TOP_CPlusPlusBase.h"

#include <fstream>
#include <vector>
#include <string>
#include <iostream>

#include "Names.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

class TensorFlowTOP : public TOP_CPlusPlusBase
{
public:
	TensorFlowTOP(const OP_NodeInfo *info, TOP_Context *context);
	virtual ~TensorFlowTOP();

	virtual void getGeneralInfo(TOP_GeneralInfo *) override;
	virtual bool getOutputFormat(TOP_OutputFormat*) override;
	virtual void execute(const TOP_OutputFormatSpecs*, OP_Inputs*, TOP_Context *context) override;
	virtual int32_t	getNumInfoCHOPChans() override;
	virtual void getInfoCHOPChan(int32_t index, OP_InfoCHOPChan *chan) override;
	virtual bool getInfoDATSize(OP_InfoDATSize *infoSize) override;
	virtual void getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries *entries) override;
	virtual void setupParameters(OP_ParameterManager *manager) override;
	virtual void pulsePressed(const char *name) override;
	virtual const char* getErrorString() override;

private:

	Status convertPixelsToTensor(std::vector<Tensor>* out_tensors,
								 uint8_t* pixels,
								 int pixels_width,
								 int pixels_height,
								 int pixels_channels,
								 const int expected_height,
								 const int expected_width,
								 const int expected_channels = 3,
								 const float expected_mean = 128,
								 const float expected_standard_dev = 128);

	GLuint createGlslProgram(const std::string& vertSrc, const std::string& fragSrc);
	void loadModel(const std::string& path);

	std::unique_ptr<tensorflow::Session> session;
	GLuint program;
	GLuint vao;
	const char* error;
	bool runGraph;
};
