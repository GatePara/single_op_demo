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
include cpukernel/CMakeFiles/cust_aicpu_kernels.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include cpukernel/CMakeFiles/cust_aicpu_kernels.dir/compiler_depend.make

# Include the progress variables for this target.
include cpukernel/CMakeFiles/cust_aicpu_kernels.dir/progress.make

# Include the compile flags for this target's objects.
include cpukernel/CMakeFiles/cust_aicpu_kernels.dir/flags.make

cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.o: cpukernel/CMakeFiles/cust_aicpu_kernels.dir/flags.make
cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.o: ../cpukernel/impl/add_block_cust_kernels.cc
cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.o: cpukernel/CMakeFiles/cust_aicpu_kernels.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.o"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel && /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ $(CXX_DEFINES) -D__FILE__=\"add_block_cust_kernels.cc,\" $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.o -MF CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.o.d -o CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.o -c /root/wt/code/ascendc/4_op_dev/1_custom_op/cpukernel/impl/add_block_cust_kernels.cc

cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.i"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel && /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ $(CXX_DEFINES) -D__FILE__=\"add_block_cust_kernels.cc,\" $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/wt/code/ascendc/4_op_dev/1_custom_op/cpukernel/impl/add_block_cust_kernels.cc > CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.i

cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.s"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel && /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ $(CXX_DEFINES) -D__FILE__=\"add_block_cust_kernels.cc,\" $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/wt/code/ascendc/4_op_dev/1_custom_op/cpukernel/impl/add_block_cust_kernels.cc -o CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.s

cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.o: cpukernel/CMakeFiles/cust_aicpu_kernels.dir/flags.make
cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.o: ../cpukernel/impl/reshape_cust_kernels.cc
cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.o: cpukernel/CMakeFiles/cust_aicpu_kernels.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.o"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel && /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ $(CXX_DEFINES) -D__FILE__=\"reshape_cust_kernels.cc,\" $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.o -MF CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.o.d -o CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.o -c /root/wt/code/ascendc/4_op_dev/1_custom_op/cpukernel/impl/reshape_cust_kernels.cc

cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.i"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel && /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ $(CXX_DEFINES) -D__FILE__=\"reshape_cust_kernels.cc,\" $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/wt/code/ascendc/4_op_dev/1_custom_op/cpukernel/impl/reshape_cust_kernels.cc > CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.i

cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.s"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel && /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ $(CXX_DEFINES) -D__FILE__=\"reshape_cust_kernels.cc,\" $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/wt/code/ascendc/4_op_dev/1_custom_op/cpukernel/impl/reshape_cust_kernels.cc -o CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.s

cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.o: cpukernel/CMakeFiles/cust_aicpu_kernels.dir/flags.make
cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.o: ../cpukernel/impl/unique_cust_kernels.cc
cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.o: cpukernel/CMakeFiles/cust_aicpu_kernels.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.o"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel && /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ $(CXX_DEFINES) -D__FILE__=\"unique_cust_kernels.cc,\" $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.o -MF CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.o.d -o CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.o -c /root/wt/code/ascendc/4_op_dev/1_custom_op/cpukernel/impl/unique_cust_kernels.cc

cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.i"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel && /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ $(CXX_DEFINES) -D__FILE__=\"unique_cust_kernels.cc,\" $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/wt/code/ascendc/4_op_dev/1_custom_op/cpukernel/impl/unique_cust_kernels.cc > CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.i

cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.s"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel && /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ $(CXX_DEFINES) -D__FILE__=\"unique_cust_kernels.cc,\" $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/wt/code/ascendc/4_op_dev/1_custom_op/cpukernel/impl/unique_cust_kernels.cc -o CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.s

# Object files for target cust_aicpu_kernels
cust_aicpu_kernels_OBJECTS = \
"CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.o" \
"CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.o" \
"CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.o"

# External object files for target cust_aicpu_kernels
cust_aicpu_kernels_EXTERNAL_OBJECTS =

makepkg/packages/vendors/xdudbgroup/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels.so: cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/add_block_cust_kernels.cc.o
makepkg/packages/vendors/xdudbgroup/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels.so: cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/reshape_cust_kernels.cc.o
makepkg/packages/vendors/xdudbgroup/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels.so: cpukernel/CMakeFiles/cust_aicpu_kernels.dir/impl/unique_cust_kernels.cc.o
makepkg/packages/vendors/xdudbgroup/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels.so: cpukernel/CMakeFiles/cust_aicpu_kernels.dir/build.make
makepkg/packages/vendors/xdudbgroup/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels.so: /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/aicpu/aicpu_kernel/lib/Ascend/libascend_protobuf.a
makepkg/packages/vendors/xdudbgroup/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels.so: /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/aicpu/aicpu_kernel/lib/Ascend/libcpu_kernels_context.a
makepkg/packages/vendors/xdudbgroup/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels.so: cpukernel/CMakeFiles/cust_aicpu_kernels.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library ../makepkg/packages/vendors/xdudbgroup/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels.so"
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cust_aicpu_kernels.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cpukernel/CMakeFiles/cust_aicpu_kernels.dir/build: makepkg/packages/vendors/xdudbgroup/op_impl/cpu/aicpu_kernel/impl/libcust_aicpu_kernels.so
.PHONY : cpukernel/CMakeFiles/cust_aicpu_kernels.dir/build

cpukernel/CMakeFiles/cust_aicpu_kernels.dir/clean:
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel && $(CMAKE_COMMAND) -P CMakeFiles/cust_aicpu_kernels.dir/cmake_clean.cmake
.PHONY : cpukernel/CMakeFiles/cust_aicpu_kernels.dir/clean

cpukernel/CMakeFiles/cust_aicpu_kernels.dir/depend:
	cd /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/wt/code/ascendc/4_op_dev/1_custom_op /root/wt/code/ascendc/4_op_dev/1_custom_op/cpukernel /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel /root/wt/code/ascendc/4_op_dev/1_custom_op/build_out/cpukernel/CMakeFiles/cust_aicpu_kernels.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cpukernel/CMakeFiles/cust_aicpu_kernels.dir/depend
