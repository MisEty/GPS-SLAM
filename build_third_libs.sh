THIRDLIB_PATH=$PWD/ThirdLibs
INSTALL_PATH=${THIRDLIB_PATH}/install

# INSTALL assimp
cd ${THIRDLIB_PATH}/assimp
mkdir -p build && cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}
make -j && make install

# INSTALL freetype
cd ${THIRDLIB_PATH}/freetype
mkdir -p build && cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}
make -j && make install

# INSTALL indicators
cd ${THIRDLIB_PATH}/indicators
mkdir -p build && cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}
make -j && make install

# INSTALL Pangolin
cd ${THIRDLIB_PATH}/Pangolin
mkdir -p build && cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}
make -j && make install

# INSTALL tensorboard_logger
cd ${THIRDLIB_PATH}/tensorboard_logger
mkdir -p build && cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}
make -j && make install

# INSTALL tinyply
cd ${THIRDLIB_PATH}/tinyply
mkdir -p build && cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}
make -j && make install

# INSTALL yaml-cpp
cd ${THIRDLIB_PATH}/yaml-cpp
mkdir -p build && cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}
make -j && make install

# Build SLAM
# cd ${THIRDLIB_PATH}/..
# mkdir -p build && cd build
# cmake ..
# make -j