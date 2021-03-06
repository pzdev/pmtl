CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

foreach(TARGET multitask-learning )
  ADD_EXECUTABLE(${TARGET} ${TARGET}.cc)
  if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET} rt)
  endif()
#  if (WITH_CUDA_BACKEND)
#	  set(CUDA_SEPARABLE_COMPILATION ON)
#	  list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-std=c++11;-DVERBOSE;-DEIGEN_USE_GPU;-DHAVE_CUDA")
#	  if(CMAKE_COMPILER_IS_GNUCXX)
#	    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.9)
#	      # gcc 4.9 or later versions raise SEGV due to the optimization problem.
	      # Use -O1 instead for now.
#	      list(APPEND CUDA_NVCC_FLAGS "-O1")
#	    else()
#	      list(APPEND CUDA_NVCC_FLAGS "-O2")
#	    endif()
#	  else()
#	    list(APPEND CUDA_NVCC_FLAGS "-O2")
#	  endif()
#	  add_dependencies(${TARGET} gdynet dynetcuda)
#	  target_link_libraries(${TARGET} gdynet dynetcuda)
#   CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
#  else()
#	  add_dependencies(${TARGET} dynet)
#	  target_link_libraries(${TARGET} dynet ${LIBS})
#  endif (WITH_CUDA_BACKEND)
  if (WITH_CUDA_BACKEND)
    target_link_libraries(${TARGET} gdynet ${LIBS})
    CUDA_ADD_CUBLAS_TO_TARGET(${TARGET})
  else()
    target_link_libraries(${TARGET} dynet ${LIBS})
  endif (WITH_CUDA_BACKEND)
  if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET} rt)
  endif()
endforeach()

