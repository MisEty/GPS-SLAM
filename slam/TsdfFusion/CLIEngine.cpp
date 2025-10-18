#include "TsdfFusion/CLIEngine.h"

#include <string.h>

#include "ORUtils/FileUtils.h"

using namespace InfiniTAM::Engine;
using namespace InputSource;
using namespace ITMLib;

CLIEngine *CLIEngine::instance;

void CLIEngine::Initialise(std::vector<ITMUChar4Image *> rgb_images,
                           std::vector<ITMShortImage *> depth_images,
                           ITMMainEngine *mainEngine)
{
    this->rgb_images = rgb_images;
    this->depth_images = depth_images;
    this->mainEngine = mainEngine;
    this->currentFrameNo = 0;

    bool allocateGPU = true;

    inputRGBImage = new ITMUChar4Image(GetRGBSize(), true, allocateGPU);
    inputRawDepthImage = new ITMShortImage(GetDepthSize(), true, allocateGPU);

    sdkCreateTimer(&timer_instant);
    sdkCreateTimer(&timer_average);
    sdkResetTimer(&timer_average);

    printf("initialised.\n");
}

bool CLIEngine::ProcessFrame()
{
    if (currentFrameNo >= rgb_images.size())
        return false;
    inputRGBImage = rgb_images[currentFrameNo];
    inputRawDepthImage = depth_images[currentFrameNo];

    sdkResetTimer(&timer_instant);
    sdkStartTimer(&timer_instant);
    sdkStartTimer(&timer_average);
    // actual processing on the mailEngine
    mainEngine->ProcessFrame(inputRGBImage, inputRawDepthImage);

    sdkStopTimer(&timer_instant);
    sdkStopTimer(&timer_average);

    float processedTime_inst = sdkGetTimerValue(&timer_instant);
    float processedTime_avg = sdkGetAverageTimerValue(&timer_average);

    // printf("frame %i: time %.2f, avg %.2f\n", currentFrameNo, processedTime_inst, processedTime_avg);

    currentFrameNo++;

    return true;
}

void CLIEngine::Run()
{
    while (true)
    {
        if (!ProcessFrame())
            break;
    }
}

void CLIEngine::Shutdown()
{
    sdkDeleteTimer(&timer_instant);
    sdkDeleteTimer(&timer_average);

    delete inputRGBImage;
    delete inputRawDepthImage;
    delete instance;
}
