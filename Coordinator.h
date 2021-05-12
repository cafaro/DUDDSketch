/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/


#ifndef __COORDINATOR_H__
#define __COORDINATOR_H__


#include "GraphNet.h"
#include "Quantiles.h"
#include "Event.h"



typedef struct PState {
    
    int peerId;             
    double q;               
    bool hasConverged;      
    
    int epoch;              
    double prev_q;          
    
    double *buffer;         
    unsigned long len;                
    
    SketchT L;              
} PState;



int validityOfData(InputT *data, bool fdist, int dtype, bool fitems);


int getUserParams(int argc, char *argv[], PeersT *p, GraphT *g, InputT *data);



FILE *logStartupConfiguration(PeersT *p, GraphT *g, InputT *data, double gamma, double NBOUND, unsigned long timestamp);

void onClose(PeersT *p, GraphT *g, InputT *data);

 
double GossipSketches(char *prefixName, int *executedRounds, PState *peers, GraphT *g, PeersT *p, ChurnEvent *churnEv, double *Qs, int Ns, InputT *data, unsigned long tSum, PVariate *dparams);


void sampleConvergenceAtRound(int sampledRound, char *PrefixName, double *Qs, int Ns, PeersT *peers, unsigned long tSum, PState *pstates, InputT *data, PVariate *dparams);


#endif // __COORDINATOR_H__
