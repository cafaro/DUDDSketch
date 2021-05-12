/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/


#ifndef __GNET_H__
#define __GNET_H__


#include "Graphs.h"  


const int DEFAULT_ROUNDS = 15;                  
const int DEFAULT_FAN_OUT = 1;                  
const int DEFAULT_PEERS = 5000;                 

const double Peer_Failure_Prob_Th = 0.1;        

const double DEFAULT_BA_A = 1.0;               
const double DEFAULT_BA_m = 5;                  
const double DEFAULT_BA_power = 1.0;            



typedef struct PeersT {
    
    int num_peers;              
    int fan_out;               
    
    int rounds;                 
    
    CHURN_MODEL churntype;      
    char churnname[FLEN];       
    double peer_failure_p;      
    
} PeersT;



void initPeersT(PeersT *p);
void printPeersT(PeersT *p, FILE *fp);



double createP2PGraph(GraphT *g, int kdegree, double radius, double ERparamP, double ERparamM, double BApower, double BAa, int BAm, int *trials);


#endif // __GNET_H__

