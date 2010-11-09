// OpenCL kernel for experimenting with phylogenic tree looping
// Martin Smith: October 2010
// 
// Issues remaining: As the experimental loop dumped into parentcache
// regardless of node, parallel write issues make this a difficult
// simulation.
//
 #pragma OPENCL EXTENSION cl_khr_fp64: enable
 
typedef double fpoint;

__kernel void FirstLoop(__global const fpoint* node_cache, __global const fpoint* model, __global fpoint* parent_cache, int nodes, int sites, int characters)
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

 	int siteStartID = parentCharGlobal-parentCharLocal;
	int parentIndex = parentCharLocal * characters;
	fpoint sum = 0.;
	long myChar;
	for (myChar = 0; myChar < characters; myChar++)
	{
		sum += node_cache[siteStartID+myChar] * model[parentIndex+myChar];
	}
	parent_cache[parentCharGlobal] *= sum;
//	parent_cache[siteIndex+parentChar] *= sum;
}
