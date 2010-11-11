// OpenCL kernel for experimenting with phylogenic tree looping
// Martin Smith: October 2010
// 
//
 #pragma OPENCL EXTENSION cl_khr_fp64: enable
 
typedef double fpoint;

__kernel void FirstLoop(__global const fpoint* node_cache, __global const fpoint* model, __global fpoint* parent_cache, __local fpoint* nodeScratch, __local fpoint * modelScratch, int nodes, int sites, int characters)
{
    // get index into global data array
    int parentCharGlobal = get_global_id(0);
	int parentCharLocal = get_local_id(0);

//	reimplement these checks to allow for non power of two work sizes. 
    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
//    if (iGID >= nodes)
//    {   
//        return; 
//    }

	nodeScratch[parentCharLocal] = node_cache[parentCharGlobal];
	// broken
	modelScratch[parentCharLocal] = model[parentCharLocal * characters + parentCharLocal];

	barrier(CLK_LOCAL_MEM_FENCE);

 	int siteStartID = parentCharGlobal-parentCharLocal;
	int parentIndex = parentCharLocal * characters;
	fpoint sum = 0.;
	long myChar;
	for (myChar = 0; myChar < characters; myChar++)
	{
//		// sum += node_cache[siteStartID+myChar] * model[parentIndex+myChar];
		sum += nodeScratch[myChar] * modelScratch[myChar]; 
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	parent_cache[parentCharGlobal] *= sum;
//	parent_cache[siteIndex+parentChar] *= sum;
}
