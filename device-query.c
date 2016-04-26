/*
  * $Id: device-query-local.c,v 1.3 2012/05/01 13:53:22 charliep Exp $
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
*/
#include "device-query-functions.h"

int main(int argc, const char** argv) {
	int deviceCount = 0, device;
	cudaError_t status = (cudaError_t)0; 
	int driverVersion = 0;

	printCudaVersion(&driverVersion, NULL);
	
	if (driverVersion == 0) {
		printf("No CUDA drivers detected--assuming no local CUDA cards.\n");
		return 0;
	}

	deviceCount = printDeviceCount();

	for (device = 0; device < deviceCount; ++device) {
		printDeviceProperties(device);
	}
	
	return 0;
}

