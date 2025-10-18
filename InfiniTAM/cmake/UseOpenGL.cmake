###################
# UseOpenGL.cmake #
###################

set(OpenGL_GL_PREFERENCE LEGACY)
FIND_PACKAGE(OpenGL REQUIRED)
MESSAGE(STATUS "opengl libs: ${OPENGL_LIBRARY}")
