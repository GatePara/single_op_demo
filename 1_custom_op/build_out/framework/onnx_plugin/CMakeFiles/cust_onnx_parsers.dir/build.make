# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/wt/code/ascendc/4_op_dev/1_custom_op

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out

# Include any dependencies generated for this target.
include framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/compiler_depend.make

# Include the progress variables for this target.
include framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/progress.make

# Include the compile flags for this target's objects.
include framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/flags.make

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.o: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/flags.make
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.o: ../framework/onnx_plugin/add_plugin.cc
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.o: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.o"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.o -MF CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.o.d -o CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.o -c /root/wt/code/ascendc/4_op_dev/1_custom_op/framework/onnx_plugin/add_plugin.cc

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.i"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/wt/code/ascendc/4_op_dev/1_custom_op/framework/onnx_plugin/add_plugin.cc > CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.i

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.s"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/wt/code/ascendc/4_op_dev/1_custom_op/framework/onnx_plugin/add_plugin.cc -o CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.s

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.o: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/flags.make
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.o: ../framework/onnx_plugin/addn_plugin.cc
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.o: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.o"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.o -MF CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.o.d -o CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.o -c /root/wt/code/ascendc/4_op_dev/1_custom_op/framework/onnx_plugin/addn_plugin.cc

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.i"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/wt/code/ascendc/4_op_dev/1_custom_op/framework/onnx_plugin/addn_plugin.cc > CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.i

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.s"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/wt/code/ascendc/4_op_dev/1_custom_op/framework/onnx_plugin/addn_plugin.cc -o CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.s

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.o: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/flags.make
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.o: ../framework/onnx_plugin/leaky_relu_plugin.cc
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.o: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.o"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.o -MF CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.o.d -o CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.o -c /root/wt/code/ascendc/4_op_dev/1_custom_op/framework/onnx_plugin/leaky_relu_plugin.cc

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.i"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/wt/code/ascendc/4_op_dev/1_custom_op/framework/onnx_plugin/leaky_relu_plugin.cc > CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.i

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.s"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/wt/code/ascendc/4_op_dev/1_custom_op/framework/onnx_plugin/leaky_relu_plugin.cc -o CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.s

# Object files for target cust_onnx_parsers
cust_onnx_parsers_OBJECTS = \
"CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.o" \
"CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.o" \
"CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.o"

# External object files for target cust_onnx_parsers
cust_onnx_parsers_EXTERNAL_OBJECTS =

makepkg/packages/vendors/xdudbgroup/framework/onnx/libcust_onnx_parsers.so: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/add_plugin.cc.o
makepkg/packages/vendors/xdudbgroup/framework/onnx/libcust_onnx_parsers.so: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/addn_plugin.cc.o
makepkg/packages/vendors/xdudbgroup/framework/onnx/libcust_onnx_parsers.so: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/leaky_relu_plugin.cc.o
makepkg/packages/vendors/xdudbgroup/framework/onnx/libcust_onnx_parsers.so: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/build.make
makepkg/packages/vendors/xdudbgroup/framework/onnx/libcust_onnx_parsers.so: /usr/local/Ascend/ascend-toolkit/latest/compiler/include/../lib64/libgraph.so
makepkg/packages/vendors/xdudbgroup/framework/onnx/libcust_onnx_parsers.so: framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library ../../makepkg/packages/vendors/xdudbgroup/framework/onnx/libcust_onnx_parsers.so"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cust_onnx_parsers.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/build: makepkg/packages/vendors/xdudbgroup/framework/onnx/libcust_onnx_parsers.so
.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/build

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/clean:
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin && $(CMAKE_COMMAND) -P CMakeFiles/cust_onnx_parsers.dir/cmake_clean.cmake
.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/clean

framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/depend:
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/wt/code/ascendc/4_op_dev/1_custom_op /root/wt/code/ascendc/4_op_dev/1_custom_op/framework/onnx_plugin /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : framework/onnx_plugin/CMakeFiles/cust_onnx_parsers.dir/depend

