/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/


#include "InputSet.h"     


void getDistName(int dtype, char *dname, int len) {

    switch (dtype){
        case UNIFORM:
            memcpy(dname, "UNIFORM", sizeof(char)*len);
            break;

        case EXPONENTIAL:
            memcpy(dname, "EXPONENTIAL", sizeof(char)*len);
            break;

        case NORMAL:
            memcpy(dname, "NORMAL", sizeof(char)*len);
            break;

        case LOGN:
            memcpy(dname, "LOGNORMAL", sizeof(char)*len);
            break;
        
        case CHIS:
            memcpy(dname, "CHISQUARED", sizeof(char)*len);
            break;
        
        default:
            memcpy(dname, "UNKNOWN", sizeof(char)*len);
    }//switch
}



void freeInputT(InputT *data) { }


void initInputData(InputT *data) {

    data->sketch_bound = 0;
    data->initial_alpha = 0;
    data->inputLen = INPUTLEN;
    data->dtype = UNKNOWN;
    data->x = 0;
    data->y = 0;
    data->process_seed = 0;
    data->fileflag = false;
}




void printInputT(InputT *data, FILE *fp, double igamma, double NBOUND) {

    FILE *out = fp?fp:stdout;
    
    if (data->fileflag){
        fprintf(out, "Input from binary file: %s\n", data->filename);
    } else {
        fprintf(out, "Input from: %s distribution (param: %.6f,%.6f)\n", data->dname, data->x, data->y);
    }
    fprintf(out, "Number of items: %ld\n", data->inputLen);
    fprintf(out, "Initial sketch memory bound: %d\n", data->sketch_bound);
    fprintf(out, "Initial alpha: %.6f\n", data->initial_alpha);
    fprintf(out, "Initial gamma: %.6f\n", igamma);
    fprintf(out, "Bound for 0 and near-0 items: %.6f\n", NBOUND);
    fprintf(out,"------------------------------------\n");
}




double *getDataFromBinaryFile(char *binfilename, unsigned long *nitems) {
    
    FILE *fp_in = fopen(binfilename, "rb");         
    if (fp_in == NULL) {
        std::cerr << "Error opening " << binfilename << std::endl;
        return NULL;
    }

    fseek(fp_in, 0, SEEK_END);
    int fileSize = ftell(fp_in);                     
    *nitems = fileSize/sizeof(double);              

    double *buffer = (double *)malloc((*nitems)*sizeof(double));

    int offset = 0;                                  
    fseek(fp_in, offset, SEEK_SET);
    fread(buffer, (*nitems)*sizeof(double), 1, fp_in); 
    fclose(fp_in);
    
    return buffer;
}






double *generateRandomData(DISTRIBUTIONS dtype, double x, double y, int len, double *NULLBOUND, unsigned long seed) {
   

    std::uniform_real_distribution<double> udistribution(x,y);  
    std::exponential_distribution<double> edistribution(x);     
    std::normal_distribution<double> ndistribution(x, y);       
    std::lognormal_distribution<double> Ldistribution(x, y);    
    std::chi_squared_distribution<double> Cdistribution(x);     
   
    std::default_random_engine generator;
    generator.seed(seed);
    
    std::function<double()> randomizer;
    
    switch(dtype){

        case UNIFORM:
            randomizer = std::bind(udistribution, generator);
            break;

        case EXPONENTIAL:
            randomizer = std::bind(edistribution, generator);
            break;

        case NORMAL:
            randomizer = std::bind(ndistribution, generator);
            break;
        
        case LOGN:
            randomizer = std::bind(Ldistribution, generator);
            break;
        
        case CHIS:
            randomizer = std::bind(Cdistribution, generator);
            break;
        
        default:
            randomizer = std::bind(ndistribution, generator);    
            break;
    }//switch


    double *buffer = (double *)malloc(sizeof(double)*(len));
    int i = 0;
    while (i < len) {
        buffer[i] = randomizer();

        if (!NULLBOUND || (NULLBOUND && buffer[i] > (*NULLBOUND))) {
            ++i;
        }//fi

    }//wend
   
    return buffer;
}



unsigned long computeParamsForPeers(DISTRIBUTIONS dtype, double x, double y, unsigned long inputLen, unsigned long seed, int npeers, PVariate *dparams, double gamma) {

    
    unsigned long totalLen = 0;

    switch (dtype) {
        
        case UNIFORM:
        {
            for(int id = 0; id < npeers; ++id) { 
                
                dparams[id].x = x + (y - x + 1)*gamma*id;
                dparams[id].y = y + (y - x + 1)*gamma*id;
                
                dparams[id].seed = seed ? (seed + id) : (std::chrono::system_clock::now().time_since_epoch().count() + id);
                
                dparams[id].len = inputLen + (id * 100); 

                totalLen += dparams[id].len;             
            }//for peers
        }
        break;



        case NORMAL:
        {
            for(int id = 0; id < npeers; ++id) { 
                dparams[id].x = x;
                dparams[id].y = y;
                dparams[id].seed = seed ? (seed + id) : (std::chrono::system_clock::now().time_since_epoch().count() + id);
                dparams[id].len = inputLen; 

                totalLen += dparams[id].len;             
            }//for peers
        }
        break;



        case LOGN:
        {
            for(int id = 0; id < npeers; ++id) { 
                dparams[id].x = x;
                dparams[id].y = y;
                dparams[id].seed = seed ? (seed + id) : (std::chrono::system_clock::now().time_since_epoch().count() + id);
                dparams[id].len = inputLen; 

                totalLen += dparams[id].len;             
            }//for peers
        }
        break;



        case EXPONENTIAL:
        {
            for(int id = 0; id < npeers; ++id) { 
                dparams[id].x = x;
                dparams[id].y = 0;
                dparams[id].seed = seed ? (seed + id) : (std::chrono::system_clock::now().time_since_epoch().count() + id);
                dparams[id].len = inputLen; 

                totalLen += dparams[id].len;            
            }//for peers
        }
        break;


        case CHIS:
        {
            for(int id = 0; id < npeers; ++id) { 
                dparams[id].x = x;
                dparams[id].y = 0;
                dparams[id].seed = seed ? (seed + id) : (std::chrono::system_clock::now().time_since_epoch().count() + id);
                dparams[id].len = inputLen; 

                totalLen += dparams[id].len;             
            }//for peers
        }
        break;


        default:
        {
            for(int id = 0; id < npeers; ++id) { 
                dtype = NORMAL;
                dparams[id].x = x; 
                dparams[id].y = y;
                dparams[id].seed = seed ? (seed + id) : (std::chrono::system_clock::now().time_since_epoch().count() + id);
                dparams[id].len = inputLen + (id*100);

                totalLen += dparams[id].len;             
            }//for peers
        }
        break;
    }//switch

return totalLen;
}




void initGenerator(ItemGenerator *h, DISTRIBUTIONS type, double x, double y, unsigned long seed) {


    std::uniform_real_distribution<double> udistribution(x,y); 
    std::exponential_distribution<double> edistribution(x);
    std::normal_distribution<double> ndistribution(x, y);
    std::lognormal_distribution<double> Ldistribution(x, y); 
    std::chi_squared_distribution<double> Cdistribution(x);  
    
    h->generator.seed(seed);

    switch (type)
    {
        case 1:
            h->randomizer = std::bind(udistribution, h->generator);
            break;

        case 2:
            h->randomizer = std::bind(edistribution, h->generator);
            break;

        case 3:
            h->randomizer = std::bind(ndistribution, h->generator);
            break;

        case 4:
            h->randomizer = std::bind(Ldistribution, h->generator);
            break;

        case 5:
            h->randomizer = std::bind(Cdistribution, h->generator);
            break;


        default:
            h->randomizer = std::bind(udistribution, h->generator);
            break;
    }

}

