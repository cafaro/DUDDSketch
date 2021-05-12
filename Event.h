/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/


#ifndef __EVENT_H__
#define __EVENT_H__


#include "Base.h"




typedef struct ChurnEvent {

    std::function<double()> randomizer; 
    unsigned seed;                      
    double a;                           
    double b;                           

    int npeers;                         
    
    CHURN_MODEL churnmode;              

    bool *PeersFailed;                  
    int failedPeersCount;                  

    int *peerOnOff_period;              
    double *AvgOnDuration;              
    double *AvgOffDuration;             
    
    double alpha;                      
    double betaOn;                      
    double betaOff;                     
    double mu;                          
    double beta;                        

    FILE *fp;                          
}ChurnEvent;





void initChurning(ChurnEvent *event, unsigned aseed, bool generateSeed, int npeers, CHURN_MODEL churnmodel);


void destroyChurning(ChurnEvent *event);



void initChurnModel(ChurnEvent *ev);


int getFailedPeers(ChurnEvent *ev, double pThreshold);


int updatePeersLifetime(ChurnEvent *ev);


int updateNetAtRound(ChurnEvent *ev, int round, double pThreshold);


#endif // __EVENT_H__

