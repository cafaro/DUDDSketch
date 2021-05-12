/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/

#include "Base.h"


extern char VERSION[];



//************************************************************ Command Line Management

void usage(char *msg) {

    std::cout << "\n\t" << VERSION << std::endl;
   	std::cout << "\nYou have to specify two groups of options:\n";
   	std::cout << "- Options for the P2P network ([]),\n";
   	std::cout << "- Options for the sketch data ({})\n";

   	std::cout << "\nUsage:\n" << msg;
    
    std::cout << " [-G graph-type]";
    std::cout << " [-S graph-seed]";
    
    std::cout << " [-P num-of-peers]"; 
    std::cout << " [-R num-of-rounds-for-convergence]";
    std::cout << " [-O fan-out]";
    
    std::cout << " [-C churning-model]";
    std::cout << " [-F peer-failure-probability (when -C is 1)]";

    std::cout << " {-a initial-alpha} {-b max_sketch_bound}";
    std::cout << " ({-i input-file}) ||";
    std::cout << " ({-d distribution-type} {-x distribution-param1} {-y distribution-param2}";
    std::cout << " {-n items-per-peer} {-h initial-data-seed})";
    std::cout << " " << std::endl;
    
    std::cout << "\n -G can be: \n";
    std::cout << " : Barabasi-Albert random graph\t[1],\n";
    std::cout << " : Erdos-Renyi random graph\t[2],\n";
    std::cout << " : Geometric random graph\t[3],\n";
    std::cout << " : Regular random graph\t\t[4],\n";

    std::cout << "\n -C can be: \n";
    std::cout << " : no churning\t\t\t[0],\n";
    std::cout << " : fail-stop model\t\t[1],\n";
    std::cout << " : Yao model\t\t\t[2],\n";
    std::cout << " : Yao with Exponential rejoin\t[3],\n";

    std::cout << "\n -d can be: \n";
    std::cout << " : Uniform distribution\t\t[1], so -x is for param 'a' and -y for param 'b' of Uniform [a,b)\n";
    std::cout << " : Exponential distribution\t[2], so -x is for 'λ' of Exponential\n";
    std::cout << " : Normal distribution\t\t[3], so -x is for param 'mean' and -y for param 'stddev' of Normal\n";
    std::cout << " : LogNormal distribution\t[4], so -x is for param 'mean' and -y for param 'stddev' of LogNormal\n";
    std::cout << " : Chi-Squared distribution\t[5], so -x is for param 'degree of freedom' of Chi-Squared\n";
    std::cout << " " << std::endl;
}




//************************************* Time evaluation


void startTimer(Timer *t) {
    if (t) {
        gettimeofday(&t->start, NULL);
    }
}


void stopTimer(Timer *t) {
    if (t) {
        gettimeofday(&t->end, NULL);
    }
}


double getElapsedSeconds(Timer *t){
    double res = -1.0;

    if (t) {
        double seconds = (t->end.tv_sec - t->start.tv_sec);
        double microsec = (t->end.tv_usec - t->start.tv_usec);
        res = microsec/1000000.0 + seconds;
    } 
return res;
}



double getElapsedMilliSecs(Timer *t) {
    double res = -1.0;

    if (t) {
        double seconds = (t->end.tv_sec - t->start.tv_sec);
        double microsec = (t->end.tv_usec - t->start.tv_usec);
        res = 1000.0*seconds + microsec/1000.0;
    }

return res;
}




//************************************* Log Files management

unsigned long composeFileName(char *fname, int flen, char *gname, int peers, int rounds, char *dname, int fo, int churntype, double failp) {

    unsigned long timestamp = time(NULL); 
    snprintf(fname, flen-1, "%.3s-%d-%d-%.10s-%d-%d-%.3f-%ld", gname, peers, rounds, dname, fo, churntype, failp, timestamp);

    return timestamp;
}



FILE *openLogFile(char prefix[], int prefixlen) {
    
    char fname[FLEN];
    snprintf(fname, FLEN-1, "Logs/%s.log", prefix);
    
    FILE *fp = NULL;
    fp = fopen(fname, "w");
    if (!fp){
        std::cout << "Error opening "<<fname<<std::endl;
    }//fi

    return fp;
}

