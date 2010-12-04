/*
 *  appleOCL.h
 *  FirstLoopXCode
 *
 *  Created by Martin Smith on 12/3/10.
 *
 */

char * load_program_source(const char *filename)
{ 
    
    struct stat statbuf;
    FILE *fh; 
    char *source; 
    
    fh = fopen(filename, "r");
    if (fh == 0)
        return 0; 
    
    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0'; 
    
    return source; 
} 