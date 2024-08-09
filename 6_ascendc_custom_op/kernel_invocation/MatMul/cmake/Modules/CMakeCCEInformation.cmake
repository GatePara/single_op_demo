include(CMakeCommonLanguageInclude)

set(CMAKE_INCLUDE_FLAG_CCE "-I")

if(UNIX)
  set(CMAKE_CCE_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_CCE_OUTPUT_EXTENSION .obj)
endif()

set(_INCLUDED_FILE 0)
set(CMAKE_SHARED_LIBRARY_CCE_FLAGS -fPIC)
set(CMAKE_SHARED_LIBRARY_CREATE_CCE_FLAGS -shared)
set(CMAKE_LIBRARY_CREATE_CCE_FLAGS "--cce-fatobj-link ${_CMAKE_COMPILE_AS_CCE_FLAG}")

if(NOT CMAKE_CCE_COMPILE_OBJECT)
    set(CMAKE_CCE_COMPILE_OBJECT
      "<CMAKE_CCE_COMPILER> -xcce <DEFINES> <INCLUDES>${__IMPLICIT_INCLUDES} ${_CMAKE_CCE_BUILTIN_INCLUDE_PATH} <FLAGS> ${_CMAKE_COMPILE_AS_CCE_FLAG} ${_CMAKE_CCE_COMPILE_OPTIONS} ${_CMAKE_CCE_COMMON_COMPILE_OPTIONS} <CMAKE_SHARED_LIBRARY_CCE_FLAGS> -pthread -o <OBJECT> -c <SOURCE>")
endif()

if(NOT CMAKE_CCE_CREATE_SHARED_LIBRARY)
  set(CMAKE_CCE_CREATE_SHARED_LIBRARY
      "<CMAKE_CCE_COMPILER> ${CMAKE_LIBRARY_CREATE_CCE_FLAGS} <CMAKE_SHARED_LIBRARY_CCE_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CCE_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS>")
endif()

if(NOT CMAKE_CCE_CREATE_SHARED_MODULE)
  set(CMAKE_CCE_CREATE_SHARED_MODULE ${CMAKE_CCE_CREATE_SHARED_LIBRARY})
endif()

if(NOT CMAKE_CCE_LINK_EXECUTABLE)
  set(CMAKE_CCE_LINK_EXECUTABLE
    "<CMAKE_CCE_COMPILER> ${CMAKE_LIBRARY_CREATE_CCE_FLAGS} <FLAGS> <CMAKE_CCE_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>${__IMPLICIT_LINKS}")
endif()

set(CMAKE_CCE_INFORMATION_LOADED 1)
