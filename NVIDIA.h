/*
 *  NVIDIA.h
 *  FirstLoopXCode
 *
 *  Created by Martin Smith on 12/3/10.
 *
 */


char * load_program_source(const char *filename, const char * argv, size_t *szKernelLength)
{

	shrLog("oclLoadProgSource (%s)...\n", filename); 
	char* cPathAndName = shrFindFilePath(filename, argv);
	char* cSourceCL = oclLoadProgSource(cPathAndName, "", szKernelLength); 
	return cSourceCL;
}
