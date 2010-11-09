// *********************************************************************
// oclFirstLoop Notes:  
//
// Runs computations with OpenCL on the GPU device and then checks results 
// against basic host CPU/C++ computation.
//
// *********************************************************************

// common SDK header for standard utilities and system libs 
#include <oclUtils.h>
#include <time.h>

// Constants
//**********************************************************************
#define SITES			1024	        //originally 1000
#define	CHARACTERS		64		//originally 61 (codons)
#define NODES			256		//originally 100

// Name of the file with the source code for the computation kernel
// *********************************************************************
const char* cSourceFile = "oclFirstLoop.cl";

// Host buffers for demo
// *********************************************************************
void* Golden;                   // Host buffer for host golden processing cross check
void *node_cache, *parent_cache, *model;

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command queue
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_mem cmNode_cache;		// OpenCL device source for node_cache
cl_mem cmParent_cache;		// OpenCL device source for parent_cache
cl_mem cmModel;			// OpenCL device source for model
size_t szGlobalWorkSize;        // 1D var for Total # of work items
size_t szLocalWorkSize;		// 1D var for # of work items in the work group	
size_t szParmDataBytes;		// Byte size of context information
size_t szKernelLength;		// Byte size of kernel code
cl_int ciErr1, ciErr2;		// Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
char* cSourceCL = NULL;         // Buffer to hold source for compilation 

shrBOOL bNoPrompt = shrFALSE;  

// Forward Declarations
// *********************************************************************
void FirstLoopHost(const double* node_cache, const double* model, double* parent_cache);
void Cleanup (int iExitCode);

// Main function 
// *********************************************************************
int main(int argc, char **argv)
{

    // time stuff:
        clock_t dtimer;
        clock_t htimer;



    // get command line arg for quick test, if provided
        bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    
    // start logs 
        shrSetLogFileName ("oclFirstLoop.txt");

    // set and log Global and Local work size dimensions
        szLocalWorkSize = 256;
        szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, NODES);
        shrLog("Global Work Size \t\t= %u\nLocal Work Size \t\t= %u\n# of Work Groups \t\t= %u\n\n", 
           szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 

    // Allocate and initialize host arrays 
	//*************************************************
        shrLog( "Allocate and Init Host Mem...\n"); 

        node_cache	= (void*)malloc (sizeof(cl_double)*CHARACTERS*SITES);
        parent_cache    = (void*)malloc (sizeof(cl_double)*NODES*CHARACTERS*SITES);
        model		= (void*)malloc (sizeof(cl_double)*CHARACTERS*CHARACTERS);
        Golden 		= (void*)malloc (sizeof(cl_double)*NODES*CHARACTERS*SITES);

	long tempindex = 0;
    // initialize the vectors
	for (tempindex = 0; tempindex < (CHARACTERS*SITES); tempindex++)
	{
		((double*)node_cache)[tempindex] = 1./CHARACTERS; // this is just dummy filler
	}

        for (tempindex = 0; tempindex < (NODES*CHARACTERS*SITES); tempindex++)
	{
                ((double*)Golden)[tempindex] = 1.;
		((double*)parent_cache)[tempindex] = 1.;
	}

	// initialize the model
	for (tempindex = 0; tempindex < (CHARACTERS*CHARACTERS); tempindex++)
	{
		((double*)model)[tempindex] = 1./CHARACTERS; // this is just dummy filler
	}

	//**************************************************


       dtimer = clock(); 

    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);

    shrLog("clGetPlatformID...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    //Get the devices
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    shrLog("clGetDeviceIDs...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    //Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    shrLog("clCreateContext...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
    shrLog("clCreateCommandQueue...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
	cmNode_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_double) * CHARACTERS * SITES, NULL, &ciErr1);
	cmModel = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_double) * CHARACTERS * CHARACTERS, NULL, &ciErr2);
    ciErr1 |= ciErr2;
	cmParent_cache = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_double) * NODES * CHARACTERS * SITES, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    shrLog("clCreateBuffer...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    
    // Read the OpenCL kernel in from source file
    shrLog("oclLoadProgSource (%s)...\n", cSourceFile); 
    cPathAndName = shrFindFilePath(cSourceFile, argv[0]);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);

    // Create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErr1);
    shrLog("clCreateProgramWithSource...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

	ciErr1 = clBuildProgram(cpProgram, 1, &cdDevice, NULL, NULL, NULL);
    shrLog("clBuildProgram...\n"); 

	// Shows the log
	char* build_log;
	size_t log_size;
	// First call to know the proper size
	clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[log_size+1];	
	// Second call to get the log
	clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[log_size] = '\0';
	shrLog(build_log);
	delete[] build_log;

    if (ciErr1 != CL_SUCCESS)
    {
	
	 shrLog("%i\n", ciErr1); //prints "1"
		  switch(ciErr1)
		  {
		    case   CL_INVALID_PROGRAM: shrLog("CL_INVALID_PROGRAM\n"); break;
		    case   CL_INVALID_VALUE: shrLog("CL_INVALID_VALUE\n"); break;
		    case   CL_INVALID_DEVICE: shrLog("CL_INVALID_DEVICE\n"); break;
		    case   CL_INVALID_BINARY: shrLog("CL_INVALID_BINARY\n"); break;	
		    case   CL_INVALID_BUILD_OPTIONS: shrLog("CL_INVALID_BUILD_OPTIONS\n"); break;
		    case   CL_COMPILER_NOT_AVAILABLE: shrLog("CL_COMPILER_NOT_AVAILABLE\n"); break;
		    case   CL_BUILD_PROGRAM_FAILURE: shrLog("CL_BUILD_PROGRAM_FAILURE\n"); break;
		    case   CL_INVALID_OPERATION: shrLog("CL_INVALID_OPERATION\n"); break;
		    case   CL_OUT_OF_HOST_MEMORY: shrLog("CL_OUT_OF_HOST_MEMORY\n"); break;
		    default: shrLog("Strange error\n"); //This is printed
		  }

        shrLog("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // Create the kernel
    ckKernel = clCreateKernel(cpProgram, "FirstLoop", &ciErr1);
    shrLog("clCreateKernel (FirstLoop)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

		
	int tempNodeCount = NODES;
	int tempSiteCount = SITES;
	int tempCharCount = CHARACTERS;

    // Set the Argument values
    ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmNode_cache);
    ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmModel);
    ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmParent_cache);
    ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_int), (void*)&tempNodeCount);
	ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_int), (void*)&tempSiteCount);
	ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(cl_int), (void*)&tempCharCount);
    shrLog("clSetKernelArg 0 - 5...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmNode_cache, CL_FALSE, 0, sizeof(cl_double) * CHARACTERS * SITES, node_cache, 0, NULL, NULL);
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmModel, CL_FALSE, 0, sizeof(cl_double) * CHARACTERS * CHARACTERS, model, 0, NULL, NULL);
    shrLog("clEnqueueWriteBuffer (node_cache and model)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // Launch kernel
    ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
    shrLog("clEnqueueNDRangeKernel (FirstLoop)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    // Synchronous/blocking read of results, and check accumulated errors
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmParent_cache, CL_TRUE, 0, sizeof(cl_double) * NODES * CHARACTERS * SITES, parent_cache, 0, NULL, NULL);
    shrLog("clEnqueueReadBuffer (Dst)...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        shrLog("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    //--------------------------------------------------------

        printf("%g seconds on device\n", (clock() - dtimer)/(double)CLOCKS_PER_SEC);
        htimer = clock();


    // Compute and compare results for golden-host and report errors and pass/fail
    shrLog("Comparing against Host/C++ computation...\n\n"); 
    FirstLoopHost ((const double*)node_cache, (const double*)model, (double*)Golden);

    printf("%g seconds on host\n", (clock() - htimer)/(double)CLOCKS_PER_SEC);

	bool match = true;
        int unmatching = 0;
        long firstUnmatch = -1;
        long lastUnmatch = -1;
	for (int i = 0; i < NODES*CHARACTERS*SITES; i++)
	{
//                if (i<100)
//                        shrLog("%e, %e\n", ((double*)parent_cache)[i], ((double*)Golden)[i]); 
//                if (((double*)parent_cache)[i] != ((double*)Golden)[i]) match = false;
                if (((double*)parent_cache)[i] != ((double*)Golden)[i])
                {
                        match = false;
                        unmatching++;
                        if (firstUnmatch == -1) firstUnmatch = i;
                        if (lastUnmatch < i) lastUnmatch = i;
                }
        }
//        shrLog("Unmatching: %i, First: %d, Last: %d\n", unmatching, firstUnmatch, lastUnmatch);
	shrLog("%s\n\n", (match) ? "PASSED" : "FAILED");


    // Cleanup and leave
    Cleanup (EXIT_SUCCESS);
}

void Cleanup (int iExitCode)
{
    // Cleanup allocated objects
    shrLog("Starting Cleanup...\n\n");
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cmNode_cache)clReleaseMemObject(cmNode_cache);
    if(cmModel)clReleaseMemObject(cmModel);
    if(cmParent_cache)clReleaseMemObject(cmParent_cache);

    // Free host memory
    free(node_cache); 
    free(model);
    free (parent_cache);
    free(Golden);

    // finalize logs and leave
    if (bNoPrompt)
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "oclVectorAdd.exe Exiting...\n");
    }
    else 
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "Exiting...\nPress <Enter> to Quit\n");
        getchar();
    }
    exit (iExitCode);
}

// "Golden" Host processing vector addition function for comparison purposes
// *********************************************************************
void FirstLoopHost(const double* hnode_cache, const double* hmodel, double* hparent_cache)
{
	long index, myChar, parentChar, site;

        double sum;

//        long i=0;
//        for (i=0;i< CHARACTERS*SITES;i++)
//        {
//                if (i%10000==0)
//                {
//                        printf("Node_cache: %e, Parent_cache: %e\n", ((double*)hnode_cache)[i], ((double*)hparent_cache)[i]);
//                }
//        }

	for (index = 0; index < NODES; index++) // this is meant to simulate looping over tree nodes
        {        
		for (site = 0; site < SITES; site++)
		{
			for (parentChar = 0; parentChar < CHARACTERS; parentChar++)
			{
				sum = 0.;
				for (myChar = 0; myChar < CHARACTERS; myChar++)
				{
					sum += hnode_cache[site*CHARACTERS+myChar] * hmodel[parentChar*CHARACTERS+myChar];
				}
//                                if ((index%100) == 0) printf("index: %ld, sum: %e\n", index, sum);
				hparent_cache[index*SITES*CHARACTERS + site*CHARACTERS+parentChar] *= sum;
//                                if ((index%100) == 0) printf("index: %ld, parent_cache: %e\n", index, hparent_cache[site*CHARACTERS+parentChar]);
			}
		}
        }      

//        for (i=0;i< CHARACTERS*SITES;i++)
//        {
//                if (i%10000==0)
//                {
//                        printf("Node_cache: %e, Parent_cache: %e\n", ((double*)hnode_cache)[i], ((double*)hparent_cache)[i]);
//                }
//        }
}
