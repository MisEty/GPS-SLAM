#pragma once

#include <iostream>
#include "InputSource/ImageSourceEngine.h"
#include "ITMLib/Core/ITMMainEngine.h"
#include "ITMLib/Utils/ITMLibSettings.h"
#include "ORUtils/FileUtils.h"
#include "ORUtils/NVTimer.h"

namespace InfiniTAM
{
    namespace Engine
    {
        class CLIEngine
        {
            static CLIEngine *instance;

            // InputSource::ImageSourceEngine *imageSource;
            std::vector<ITMUChar4Image *> rgb_images;
            std::vector<ITMShortImage *> depth_images;
            // std::vector<ORUtils::SE3Pose> cam_poses;
            // std::vector<ITMLib::ITMIntrinsics> cam_intrincs;
            ITMLib::ITMLibSettings internalSettings;
            ITMLib::ITMMainEngine *mainEngine;

            StopWatchInterface *timer_instant;
            StopWatchInterface *timer_average;

        private:
            ITMUChar4Image *inputRGBImage;
            ITMShortImage *inputRawDepthImage;

            int currentFrameNo;

        public:
            static CLIEngine *Instance(void)
            {
                if (instance == NULL)
                    instance = new CLIEngine();
                return instance;
            }

            float processedTime;

            void Initialise(std::vector<ITMUChar4Image *> rgb_images,
                            std::vector<ITMShortImage *> depth_images,
                            ITMLib::ITMMainEngine *mainEngine);
            void Shutdown();

            void Run();
            bool ProcessFrame();

            Vector2i GetDepthSize()
            {
                return depth_images[0]->noDims;
            }

            Vector2i GetRGBSize()
            {
                return rgb_images[0]->noDims;
            }

            ITMLib::ITMMainEngine *getMainEngine()
            {
                return mainEngine;
            }
        };
    }
}
