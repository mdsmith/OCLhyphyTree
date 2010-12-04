// OpenCL kernel for experimenting with phylogenic tree looping
// Martin Smith: October 2010, Novemeber 2010
//
//
 #pragma OPENCL EXTENSION cl_khr_fp64: enable

#if defined(__APPLE__)
typedef float fpoint;
#else
typedef double fpoint;
#endif

__kernel void FirstLoop(__global const fpoint* node_cache, __global const fpoint* model, __global fpoint* parent_cache, 
        __local fpoint* nodeScratch, __local fpoint * modelScratch, int nodes, int sites, int characters,
        __global int* scalings, fpoint uflowthresh, fpoint scalar)
{
    // get index into global data array
    int parentCharGlobal = get_global_id(0); // a unique global ID for each parentcharacter
        int parentCharLocal = get_local_id(0); // a local ID unique within the site.

//      reimplement these checks to allow for non power of two work sizes.
    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
//    if (iGID >= nodes)
//    {
//        return;
//    }

        nodeScratch[parentCharLocal] = node_cache[parentCharGlobal];
        modelScratch[parentCharLocal] = model[parentCharLocal * characters + parentCharLocal];

        barrier(CLK_LOCAL_MEM_FENCE);

//      int siteStartID = parentCharGlobal-parentCharLocal;
//      int parentIndex = parentCharLocal * characters;
        fpoint sum = 0.;
        long myChar;
        for (myChar = 0; myChar < characters; myChar++)
        {
//              // sum += node_cache[siteStartID+myChar] * model[parentIndex+myChar];
                sum += nodeScratch[myChar] * modelScratch[myChar];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

    while (parent_cache[parentCharGlobal] < uflowthresh)
    {
        parent_cache[parentCharGlobal] *= scalar;
        scalings[parentCharGlobal] += 1;
    }

        parent_cache[parentCharGlobal] *= sum;
//    if (parent_cache[parentCharGlobal] < .0000001 && scalings[parentCharGlobal] ==0) scalings[parentCharGlobal] += (uflowthresh);
//      parent_cache[siteIndex+parentChar] *= sum;
}
