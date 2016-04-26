/*
  * $Id: device-query-functions.h,v 1.3 2012/05/01 13:53:22 charliep Exp $
  * 
  * This file is part of BCCD, an open-source live CD for computational science
  * education.
  * 
  * Copyright (C) 2010 Andrew Fitz Gibbon, Paul Gray, Kevin Hunter, Dave 
  *   Joiner, Sam Leeman-Munk, Tom Murphy, Charlie Peck, Skylar Thompson, 
  *   & Aaron Weeden 
  * 
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  * 
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  * 
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
  * CUDA/OpenCL capable device query program, uses the CUDA runtime API.
  *
  * Usage: device-query 
  *
  * charliep	13-April-2011	First pass, based on deviceQuery from NVIDIA.
  * charliep	01-July-2011	Improved error handling, additional characteristics.
  * charliep	03-August-2012	Added CPU count based on documentation and compute version
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/unistd.h>
#include <cuda_runtime_api.h>

//prototypes
void printCudaVersion();
int printDeviceCount();
void printDeviceProperties(int deviceID);

//function definitions

void printCudaVersion(int * driverVersion_out, int * runtimeVersion_out) {
	cudaError_t status = (cudaError_t)0;
	int driverVersion = 0, runtimeVersion = 0;
 
	if ((status = cudaDriverGetVersion(&driverVersion)) != cudaSuccess) {
		fprintf(stderr, "cudaDriverGetVersion() FAILED, status = %d (%s)\n", status, cudaGetErrorString(status));
		exit(1);
	} else {
		printf("CUDA driver version: %d.%d\n", driverVersion / 1000, driverVersion % 100);
	}

	if ((status = cudaRuntimeGetVersion(&runtimeVersion)) != cudaSuccess) {
		fprintf(stderr, "cudaRuntimeGetVersion() FAILED, status = %d (%s)\n", status, cudaGetErrorString(status));
		exit(1); 
	} else {
		printf("CUDA runtime version: %d.%d\n", runtimeVersion / 1000, runtimeVersion % 100);
	}
	if (driverVersion_out != NULL) *driverVersion_out = driverVersion;
	if (runtimeVersion_out != NULL) *runtimeVersion_out = runtimeVersion;
}

int printDeviceCount() {
	cudaError_t status = (cudaError_t)0;
	int deviceCount;
	
	if ((status = cudaGetDeviceCount(&deviceCount)) != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount() FAILED, status = %d (%s)\n", status, cudaGetErrorString(status));
		exit(1);
	}
	
    if (deviceCount == 0) { 
		printf("There are no hardware devices which support CUDA\n");
	} else {
		printf("There %s %d CUDA capable hardware device%s\n", deviceCount == 1 ? "is" : "are", 
		  deviceCount, deviceCount > 1 ? "s" : ""); 
	}
	return deviceCount;
}

void printDeviceProperties(int deviceID) {
	cudaError_t status = (cudaError_t)0;
	struct cudaDeviceProp deviceProperties;
        char hostname[128];

        gethostname(hostname, sizeof(hostname));
        fprintf(stderr, "hostname: %s\n", hostname);

	if ((status = cudaGetDeviceProperties(&deviceProperties, deviceID)) != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceProperties() FAILED, status = %d (%s)\n", status, cudaGetErrorString(status));
		exit(1); 
	}

	printf("Device %d:\n", deviceID); 
	printf("\tname = %s\n", deviceProperties.name);
	printf("\tCUDA capability major.minor version = %d.%d\n", deviceProperties.major, deviceProperties.minor);

	// Compute Capability <= 1.3 --> 8 CUDA Cores per SM
	// CC == 2.0 --> 32 CUDA cores per SM
	// CC == 2.1 --> 48 CUDA cores per SM
	// CC == 3.5 --> 192 CUDA cores per SM

	printf("\tmultiProcessorCount = %d\n", deviceProperties.multiProcessorCount);
	
	if ((deviceProperties.major <= 1) && (deviceProperties.minor <= 3)) 
		printf("\tMultiprocessors x (Cores/MP) = Total Cores : \n\t\t%d (MP) x %d (Cores/MP) = %d (Total Cores)\n", deviceProperties.multiProcessorCount, 8, 8 * deviceProperties.multiProcessorCount);
	
	else if ((deviceProperties.major == 2) && (deviceProperties.minor == 0)) 
		printf("\tMultiprocessors x (Cores/MP) = Total Cores : %d (MP) x %d (Cores/MP) = %d (Cores)\n", deviceProperties.multiProcessorCount, 32, 32 * deviceProperties.multiProcessorCount); 

	else if ((deviceProperties.major == 2) && (deviceProperties.minor == 1)) 
		printf("\tMultiprocessors x (Cores/MP) = Total Cores : %d (MP) x %d (Cores/MP) = %d (Cores)\n", deviceProperties.multiProcessorCount, 48, 48 * deviceProperties.multiProcessorCount); 	
		
	else if (deviceProperties.major == 3)
		printf("\tMultiprocessors x (Cores/MP) = Total Cores : %d (MP) x %d (Cores/MP) = %d (Cores)\n", deviceProperties.multiProcessorCount, 192, 192 *
deviceProperties.multiProcessorCount);

	else if (deviceProperties.major == 5)
		printf("\tMultiprocessors x (Cores/MP) = Total Cores : %d (MP) x %d (Cores/MP) = %d (Cores)\n", deviceProperties.multiProcessorCount, 128, 128 *
deviceProperties.multiProcessorCount);
	
	else 
		printf("\tUnknown CUDA capability\n"); 

	printf("\ttotalGlobalMem = %ld bytes\n", (long)deviceProperties.totalGlobalMem); 
	printf("\tsharedMemPerBlock = %d bytes\n", (int)deviceProperties.sharedMemPerBlock);
	printf("\tregsPerBlock = %d\n", deviceProperties.regsPerBlock);
	printf("\twarpSize = %d\n", deviceProperties.warpSize);
	printf("\tmemPitch = %d bytes\n", (int)deviceProperties.memPitch);
	printf("\tmaxThreadsPerBlock = %d\n", deviceProperties.maxThreadsPerBlock);
	printf("\tmaxThreadsDim = %d x %d x %d\n", deviceProperties.maxThreadsDim[0], 
	  deviceProperties.maxThreadsDim[1], deviceProperties.maxThreadsDim[2]);
	printf("\tmaxGridSize = %d x %d x %d\n", deviceProperties.maxGridSize[0], 
	  deviceProperties.maxGridSize[1], deviceProperties.maxGridSize[2]);
	printf("\n");	
	printf("\tmemPitch = %ld bytes\n", (long)deviceProperties.memPitch);
	printf("\ttextureAlignment = %ld bytes\n", (long)deviceProperties.textureAlignment);
	printf("\tclockRate = %.2f GHz\n", deviceProperties.clockRate * 1e-6f);

#if CUDART_VERSION >= 2000
	printf("\tdeviceOverlap = %s\n", deviceProperties.deviceOverlap ? "Yes" : "No");
#endif

#if CUDART_VERSION >= 2020
	printf("\tkernelExecTimeoutEnabled = %s\n", deviceProperties.kernelExecTimeoutEnabled ? "Yes" : "No");
	printf("\tintegrated = %s\n", deviceProperties.integrated ? "Yes" : "No");
	printf("\tcanMapHostMemory = %s\n", deviceProperties.canMapHostMemory ? "Yes" : "No");
	printf("\tcomputeMode = %s\n", deviceProperties.computeMode == cudaComputeModeDefault ?
	  "Default (multiple host threads can use this device simultaneously)" :
	  deviceProperties.computeMode == cudaComputeModeExclusive ?
	  "Exclusive (only one host thread at a time can use this device)" :
	  deviceProperties.computeMode == cudaComputeModeProhibited ?
	  "Prohibited (no host thread can use this device)" :
	  "Unknown");
#endif

#if CUDART_VERSION >= 3000
	printf("\tconcurrentKernels = %s\n", deviceProperties.concurrentKernels ? "Yes" : "No");
#endif

#if CUDART_VERSION >= 3010
	printf("\tECCEnabled = %s\n", deviceProperties.ECCEnabled ? "Yes" : "No");
#endif

#if CUDART_VERSION >= 3020
	printf("\ttccDriver = %s\n", deviceProperties.tccDriver ? "Yes" : "No");
#endif	

	printf("\n"); 
}
