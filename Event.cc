/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/


#include "Event.h"


void initChurning(ChurnEvent *event, unsigned aseed, bool generateSeed, int npeers, CHURN_MODEL churnmodel) {
        
    event->a = 0.0;
    event->b = 1.0;
    std::uniform_real_distribution<double> udistr(event->a, event->b);

    if (generateSeed) {
        std::random_device randomDevice;
        event->seed = randomDevice();
    } else {
        event->seed = (aseed)? aseed : std::chrono::system_clock::now().time_since_epoch().count();
    }//fi seed

    std::mt19937_64 MTengine(event->seed);    
    event->randomizer = std::bind(udistr, MTengine);
    
    
    event->churnmode = churnmodel;

    event->npeers = npeers;                                     // total number of peers
    event->failedPeersCount = 0;                                // total number of failed peers    
    event->PeersFailed = (bool *)malloc(sizeof(bool)*npeers);   
    
    if (event->churnmode > 1 ) {    
        
        event->peerOnOff_period = (int *)malloc(sizeof(int)*npeers); 

        event->AvgOnDuration = (double *)malloc(sizeof(double)*npeers); 
        event->AvgOffDuration = (double *)malloc(sizeof(double)*npeers);

        event->alpha = 3.0;                 
        event->betaOn = 1.0;                
        event->mu = 1.01;                   
        
        event->betaOff = 2.0;               

        event->beta = 3.0;                  
    }//fi churnmode

}



void destroyChurning(ChurnEvent *event) {

    if (event) {

        free(event->PeersFailed);

        if (event->churnmode > 1) {
            free(event->peerOnOff_period);
            free(event->AvgOnDuration);
            free(event->AvgOffDuration);
        }

        if (event->fp){
            fclose(event->fp);
        }

    }//fi event

}



void initChurnModel(ChurnEvent *ev) {
    
    if (ev->churnmode < 2) {
        
        memset(ev->PeersFailed, 0, sizeof(bool)*ev->npeers);
        ev->failedPeersCount = 0;
        
    } else {

        memset(ev->PeersFailed, 1, sizeof(bool)*ev->npeers);
        ev->failedPeersCount = ev->npeers; 

        memset(ev->peerOnOff_period, 0, sizeof(int)*ev->npeers);


        for(int i = 0; i < ev->npeers; ++i) {

            double on = ev->randomizer();

            ev->AvgOnDuration[i] = ev->betaOn/pow(1.0-on,1.0/ev->alpha) - ev->betaOn + ev->mu;
            
            double off = ev->randomizer();

            ev->AvgOffDuration[i] = ev->betaOff/pow(1.0-off,1.0/ev->alpha) - ev->betaOff + ev->mu;    
        }//for peers    
    }
 
}






int getFailedPeers(ChurnEvent *ev, double pThreshold) {
    
    int roundFails = 0;

    for(int i = 0; i < ev->npeers; ++i) 
    {
        if (!ev->PeersFailed[i]) 
        {
            ev->PeersFailed[i] = ( ev->randomizer() < pThreshold) ? true : false;
            
            if (ev->PeersFailed[i]) {
                roundFails +=1;
            } 
        }//fi
    }//for npeers
        
    return roundFails; 
}






int updatePeersLifetime(ChurnEvent *ev) {
    
    int roundFailures = 0;
        
    double alpha = 2.0;
    double beta = 3.0;
    double mu = 0.0;
    double randn, a;


    for(int i = 0; i < ev->npeers; ++i ) {

        if (ev->peerOnOff_period[i] > 0) 
        {
            --ev->peerOnOff_period[i]; 
        }
        else if ( !ev->PeersFailed[i]) 
        {   
            
           
            a = alpha * ev->AvgOffDuration[i];
            
            
            randn = ev->randomizer();

            ev->peerOnOff_period[i] = (int) std::round(beta/pow(1.0-randn,1.0/a) - beta + mu);
            
            if (ev->peerOnOff_period[i] > 0)
            {
                ev->PeersFailed[i] = true;
                roundFailures += 1;
            }

        }
        else if (ev->PeersFailed[i]) 
        {   
             
            if (ev->churnmode==YAOEXP) 
            {   
               
                double exp_lambda = (1.0/ev->AvgOnDuration[i]);
               
                randn = ev->randomizer();

                ev->peerOnOff_period[i] = (int) std::round(-log(randn)/exp_lambda);
            } 
            else 
            {   
                
                a = alpha * ev->AvgOffDuration[i];

                
                randn = ev->randomizer();

                
                ev->peerOnOff_period[i] = (int) std::round(beta/pow(1.0-randn,1.0/alpha) - beta + mu);
            }//fi expChurn

            if (ev->peerOnOff_period[i] > 0) {
                ev->PeersFailed[i] = false;
                roundFailures -= 1;
            }//fi onOff_period

        }//fi ev->PeersFailed[i]
    }//for npeers

    return roundFailures;
}






int updateNetAtRound(ChurnEvent *ev, int round, double pThreshold) {
    
    int failedAtRound = 0;

    if (ev->churnmode==FAILSTOP && round) {
        
        failedAtRound = getFailedPeers(ev, pThreshold);
    }
    else if (ev->churnmode > 1 ) {
        failedAtRound = updatePeersLifetime(ev);
    }  else {
        //no churning
    } 

    //log
    for(int i = 0; i < ev->npeers; ++i) 
    {
        
        if (ev->fp){
            fprintf(ev->fp, "%d,%d,%c\n", round, i, (ev->PeersFailed[i])?'Y':'N');
        }//fi log
    }//for

    ev->failedPeersCount += failedAtRound; 

return failedAtRound;
}


