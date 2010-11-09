// OpenCL kernel for experimenting with phylogenic tree looping
// Martin Smith: October 2010
// 
// Issues remaining: As the experimental loop dumped into parentcache
// regardless of node, parallel write issues make this a difficult
// simulation.
//
 #pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void FirstLoop(__global const double* node_cache, __global const double* model, __global double* parent_cache, int nodes, int sites, int characters)
{
    // get index into global data array
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
//    if (iGID >= nodes)
//    {   
//        return; 
//    }

	long myChar, parentChar, site;
    
        double sum;
        for (site = 0; site < sites; site++)
        {
		for (parentChar = 0; parentChar < characters; parentChar++)
		{
                        parent_cache[(iGID*sites*characters)+(site*characters)+parentChar] = 1.;
			sum = 0.;
			for (myChar = 0; myChar < characters; myChar++)
			{
				sum += node_cache[site*characters+myChar] * model[parentChar*characters+myChar];
			}
			parent_cache[(iGID*sites*characters)+(site*characters)+parentChar] *= sum;
		}
	}
}
