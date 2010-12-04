/*
 *  NVIDIA.h
 *  FirstLoopXCode
 *
 *  Created by Martin Smith on 12/3/10.
 *
 */


char * load_program_source(const char *filename)
{

	shrLog("oclLoadProgSource (%s)...\n", cSourceFile); 
	char* cPathAndName = shrFindFilePath(filename, argv[0]);
	char* cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength); 
	return source;
}