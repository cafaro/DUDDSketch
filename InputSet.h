/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/

#ifndef __DATAM_H__
#define __DATAM_H__

#include "Base.h"     



const double ALPHA = 0.001;             
const int DEFAULT_BOUND = 1024;         

const int MIN_KEY = pow(2,30);          

const int ELOGFILE = 4;                 
const int EDATA = 7;                    

const int INPUTLEN = 100000;            



enum DISTRIBUTIONS{ UNKNOWN=-1, UNIFORM=1, EXPONENTIAL=2, NORMAL=3, LOGN=4, CHIS=5}; 


typedef struct InputT {

    char filename[FLEN];            
    bool fileflag;                  

    int sketch_bound;               
    double initial_alpha;           

    DISTRIBUTIONS dtype;            
    char dname[NLEN];               
    double x;                       
    double y;                       
    unsigned long inputLen;                  
    
    unsigned long process_seed;     

} InputT;



typedef struct ParamD{
    
    double x;
    double y;
    unsigned long len;

    unsigned long seed;
} PVariate;



typedef struct ItemGenerator{
    std::function<double()> randomizer;
    std::default_random_engine generator;
} ItemGenerator;


void getDistName(int dtype, char *dname, int len);


void initInputData(InputT *data);


void printInputT(InputT *data, FILE *fp, double igamma, double NBOUND);


double *getDataFromBinaryFile(char *binfilename, unsigned long *nitems);

double *generateRandomData(DISTRIBUTIONS dtype, double x, double y, int len, double *NULLBOUND, unsigned long seed);
 

unsigned long computeParamsForPeers(DISTRIBUTIONS dtype, double x, double y, unsigned long inputLen, unsigned long seed, int npeers, PVariate *dparams, double gamma);


void initGenerator(ItemGenerator *h, DISTRIBUTIONS type, double x, double y, unsigned long seed);

#endif // __DATAM_H__

