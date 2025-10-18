#pragma once

#include "TsdfFusion/UIEngine.h"
#include "TsdfFusion/CLIEngine.h"
#include "ITMLib/ITMLibDefines.h"
#include "ITMLib/Core/ITMBasicEngine.h"

#include "dataset_reader.h"
#include "cv_utils.h"
#include "tensor_math.h"

using namespace InfiniTAM::Engine;
using namespace InputSource;
using namespace ITMLib;

CLIEngine *createTsdfEngine(const DatasetReader &data_reader, const YAML::Node &config);