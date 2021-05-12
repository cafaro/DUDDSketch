/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/


#include "Coordinator.h" 



FILE *logStartupConfiguration(PeersT *p, GraphT *g, InputT *data, double gamma, double NBOUND, unsigned long timestamp) {

    if (!p || !g || !data) {
        std::cout << "Failed to pass Params"<<std::endl;
        return NULL;
    }

    FILE *fp = NULL;

    char fname[FLEN];
    snprintf(fname, FLEN, "Logs/%.3s-%d-%d-%.3s-%ld.log", g->gname, p->num_peers, p->rounds, data->dname, timestamp);
    
    fp = fopen(fname, "w");
    if (!fp){
        std::cout << "Error on opening "<<fname<<std::endl;
        return fp;
    }

    printGraphT(g, fp);
    fprintf(fp, "----------\n");
    printPeersT(p, fp);
    fprintf(fp, "----------\n");
    printInputT(data, fp, gamma, NBOUND);
    fprintf(fp, "----------\n");
    
return fp;
}



void onClose(PeersT *p, GraphT *g, InputT *data) {
    if (g){
        freeGraphT(g);
    }
}



int validityOfData(InputT *data, bool fdist, int dtype, bool fitems) {
    
    int res = 0;

    if (fdist) 
    {
        switch (dtype){
        
            case UNIFORM:
                {
                    data->dtype = UNIFORM;
                    if (data->y <= data->x) {
                        std::cout<<"Error on setting the range [a,b) for Uniform distribution\n" << std::endl;
                        return EPARAMS;
                    }
                    #ifdef POSI
                        if (data->x < 0.0) {
                            std::cout<<"POSITIVE MODE: we only want to deal with positive numbers, set [a,b) in the positive range \n" << std::endl;
                            return EPARAMS;
                        }
                    #endif
                }
                break;

            case EXPONENTIAL:
                {
                    data->dtype = EXPONENTIAL;
                    if (data->x <= 0.0){
                        std::cout<<"Error on setting the lambda (λ) for Exponential distribution (it has to be (λ>0)\n" << std::endl;
                        return EPARAMS;
                    }
                }
                break;

            case NORMAL:
                {
                    data->dtype = NORMAL;
                    if (data->y==0.0) {
                        std::cout<<"Error on setting stddev (σ) for Normal distribution\n" <<std::endl;
                        return EPARAMS;
                    }

                    #ifdef POSI
                        if (data->x <= 0) {
                            std::cout<<"POSITIVE MODE: we only want to deal with positive numbers, set mean (μ) in the positive range \n" << std::endl;
                            return EPARAMS;
                        }
                    #endif
                }
                break;

            case LOGN:
                {
                    data->dtype = LOGN;
                    #ifdef POSI
                        if (data->x < 0) {
                            std::cout<<"POSITIVE MODE: we only want to deal with positive numbers, set mean (μ) in the positive range \n" << std::endl;
                            return EPARAMS;
                        }
                    #endif
                }
                break;
            

            case CHIS:
                {
                    data->dtype = CHIS;
                    if (data->x==0.0){
                        std::cout<<"Error on setting the degree of freedom for Chi-Squared distribution\n" <<std::endl;
                        return EPARAMS;
                    }
                    #ifdef POSI
                        if (data->x < 0) {
                            std::cout<<"POSITIVE MODE: we only want to deal with positive numbers, set degree of freedom in the positive range \n" << std::endl;
                            return EPARAMS;
                        }
                    #endif
                }
                break;

            
            default: 
                data->dtype = UNKNOWN;
                fdist = false;
                std::cout<<"Error on setting input distribution type\n" <<std::endl;
                break;
        }//switch

        getDistName(data->dtype, data->dname, NLEN);     
    }//fi fdist


    if (!fdist && !data->fileflag) {
        std::cout << "You must specify either the input binary file (-i) or the distribution to process (-d)!\n"<<std::endl;
        return EPARAMS;
    }
    
    if (fdist && !fitems) {
        data->inputLen = INPUTLEN;
        std::cout << "\tNumber of items to process invalid or not configured ... Setting to default " << data->inputLen << std::endl;
    }

    if (data->fileflag) {
        std::string name = data->filename;
        std::size_t from_p = name.find_last_of("/"); 
        std::size_t to_p = name.find_last_of("."); 
        memcpy(data->dname, name.substr(from_p+1, to_p-from_p-1).c_str(), sizeof(char)*NLEN);
    }//fi 

    return res;
}





int getUserParams(int argc, char *argv[], PeersT *p, GraphT *g, InputT *data) {

    int res = 0;
    
    if (!p || !g || !data) {
        std::cout << "Failed to pass Params"<<std::endl;
        exit(EPARAMS);
    }
    

    initGraphT(g);
    bool fgraph = false;    
    int gtype = -1;         
    

    initPeersT(p);
    bool fpeers = false;    
    bool frounds = false;   
    int churnmodel = 0;
    
    
    initInputData(data);
    bool falpha = false;    
    bool fbound = false;    
    
    bool fdist = false;     
    int dtype = UNKNOWN;    
    bool fitems = false;    

    
    int c=0;    
    
    while ( (c = getopt(argc, argv, "G:S:R:P:O:M:C:F:a:b:i:d:x:y:n:h:")) != -1) 
    { 
        switch (c) {
            
            case 'G': 
                { 
                    gtype = atoi(optarg); 
                    
                    if (gtype < 1 || gtype > 4) {
                        std::cout << "Invalid graph type specification\n";
                        return EPARAMS;
                    }//fi gtype
                    fgraph = true;  

                    switch (gtype) {
                        case BARABASI:
                            g->graph_type = BARABASI;
                            break;
                        case ERDOS:
                            g->graph_type = ERDOS;
                            break;
                        case GEOMETRIC:
                            g->graph_type = GEOMETRIC;
                            break;
                        case REGULAR:
                            g->graph_type = REGULAR;
                            break;
                        default: 
                            g->graph_type = UNRECOGNIZED;
                            break;
                    }//switch
                    getGraphName(g->graph_type, g->gname, NLEN);
                }
                break; 

            case 'S':
                g->initial_seed = strtoul(optarg, NULL, 10); 
                break;
     
            case 'P':
                p->num_peers = atoi(optarg); 
                if (p->num_peers <= 1) {
                    std::cout << "Invalid number of peers specification (at least 2)\n";
                    return EPARAMS;
                }//fi
                fpeers = true; 
                break;
            
            case 'O':
                p->fan_out = atoi(optarg); 
                if (p->fan_out < 1) {
                    std::cout << "Invalid fan-out number \n";
                    return EPARAMS;
                }
                break;
            
            case 'R':
                p->rounds = atoi(optarg); 
                if (p->rounds < 1) {
                    std::cout << "Invalid number of rounds specification\n";
                    return EPARAMS;
                }
                frounds = true; 
                break;


            case 'C':
                churnmodel = atoi(optarg);
                if (churnmodel < 0 || churnmodel > 3) {
                    std::cout << "Churning model specification invalid, disable churning\n";
                    churnmodel = 0;
                }//fi 
                break;


            case 'F': 
                p->peer_failure_p = strtod(optarg,NULL); 
                break;


            case 'a':
                data->initial_alpha = strtod(optarg, NULL); 
                falpha = true;
                break;


            case 'b':
                data->sketch_bound = atoi(optarg);
                fbound = true;
                break;

            case 'i':
                
                if (strlen(optarg) <= FLEN) {
                    snprintf(data->filename, FLEN-1,"%s", optarg);
                    data->fileflag = true; 
                }
                break;

            case 'd':
                dtype = atoi(optarg); 
                if (dtype < 1 || dtype > 5) {
                    std::cout << "Invalid distribution model specification\n";
                    return EPARAMS;
                }//fi 
                fdist = true; 
                break;        

            case 'x':
                data->x = strtod(optarg, NULL);
                break;
            
            case 'y':
                data->y = strtod(optarg, NULL);
                break;
            
            case 'n':
                data->inputLen = strtoul(optarg, NULL, 10); 
                fitems = true;
                break;


            case 'h':
                data->process_seed = strtoul(optarg, NULL, 10); 
                break;

            default:
                fprintf(stdout, "Incorrect parameter specification for: '%c'\n", optopt);
                break;
        }// switch
    }//wend


    if (!fgraph) {
        std::cout << " You must provide the kind of graph to use, -G option\n" <<std::endl;
        exit(EPARAMS);
    }


    if (!fpeers) {
        p->num_peers = DEFAULT_PEERS;
        std::cout << "\tNumber of peers in the network, -P option invalid or not configured ... Setting to default " <<  p->num_peers << std::endl;
    }


    if (p->fan_out < 1 || p->fan_out >= p->num_peers) {
        p->fan_out = DEFAULT_FAN_OUT;
        
    }

    if (!frounds){
        std::cout << " You must provide the number of gossip rounds, -R option\n" <<std::endl;
        exit(EPARAMS);
    }//fi rounds



    switch (churnmodel)
    {
        case NOCHURN:
            p->churntype = NOCHURN;
            snprintf(p->churnname, FLEN-1, "No churning");
            break;
        
        case FAILSTOP:
            p->churntype = FAILSTOP;
            snprintf(p->churnname, FLEN-1, "Fail-and-Stop churning");
            break;
        
        case YAO:
            p->churntype = YAO;
            snprintf(p->churnname, FLEN-1, "Churning with Yao model");
            break;

        case YAOEXP:
            p->churntype = YAOEXP;
            snprintf(p->churnname, FLEN-1, "Churning with Yao model and Exponential rejoin");
            break;

        default:
            p->churntype = NOCHURN;
            snprintf(p->churnname, FLEN-1, "No churning");
            break;
    }//switch


    
    if (validityOfData(data, fdist, dtype, fitems)){// == EPARAMS) {
        std::cout << "Incorrect input data arguments!\n"<<std::endl;
        return EPARAMS;
    }

    
    if (!falpha){
        data->initial_alpha = ALPHA; 
    }
    
    if (!fbound){
        data->sketch_bound = DEFAULT_BOUND;
    }


return res;
}





double GossipSketches(char *prefixName, int *executedRounds, PState *peers, GraphT *g, PeersT *p, ChurnEvent *churnEv, double *Qs, int Ns, InputT *data, unsigned long tSum, PVariate *dparams) {
    
    Timer gossipTime;                          

    igraph_vector_t neighbours[p->num_peers];   
    long neighboursCount[p->num_peers];         

    int failedPeersAtRound = 0;                 
    int aliveP = 0;                            
    
    int currentRound = 0;                      
    int STEP = 5;
    sampleConvergenceAtRound(currentRound, prefixName, Qs, Ns, p, tSum, peers, data, dparams);

     
    startTimer(&gossipTime);          
    
    while(currentRound < p->rounds)  { 
   
        
        failedPeersAtRound = updateNetAtRound(churnEv, currentRound, p->peer_failure_p);
        aliveP = p->num_peers - churnEv->failedPeersCount;
 
        fprintf(stderr, "Round %d, Failed at round: %d, Total Failed peers %d/%d, Joined %d\n", currentRound, failedPeersAtRound, churnEv->failedPeersCount, p->num_peers, aliveP);

        for(int i = 0; i < p->num_peers; ++i) {
            peers[i].prev_q = peers[i].q; 
        }
        

        for (int peerI = 0; peerI < p->num_peers; ++peerI){

            if (!churnEv->PeersFailed[peerI]) {   

                bool isChurnOn = churnEv->churnmode;
                neighboursCount[peerI] = getPeerNeighbours(peerI, g, &neighbours[peerI], p->fan_out, churnEv->PeersFailed, p->num_peers, isChurnOn);

                assert(neighboursCount[peerI] <= p->fan_out);
                
                for(int j=0; j < neighboursCount[peerI]; ++j) 
                {
                    
                    int peerJ = (int) igraph_vector_e(&neighbours[peerI], j);
 
                    int epoch = (peers[peerI].epoch < peers[peerJ].epoch)?peers[peerJ].epoch:peers[peerI].epoch;
                    peers[peerI].epoch = peers[peerJ].epoch = epoch+1;
                    
                    double mean = (peers[peerI].q + peers[peerJ].q)/2.0;
                    peers[peerI].q = peers[peerJ].q = mean;

                    double mergeTime = 0;
                    mergeSketches(peerI, &(peers[peerI].L), peerJ, &(peers[peerJ].L), &mergeTime, peers[peerJ].hasConverged );
                    
                }//for push-pull with j

                igraph_vector_destroy(&neighbours[peerI]);

            }//fi 

        }//for peerI
    
    ++currentRound;
    if (currentRound%STEP == 0){
        sampleConvergenceAtRound(currentRound, prefixName, Qs, Ns, p, tSum, peers, data, dparams);
    }
 

    }//wend currentRound
    stopTimer(&gossipTime);                     
    

    //*** n. End gossiping
    double seconds = getElapsedSeconds(&gossipTime);
    if (executedRounds) {
        *executedRounds = currentRound;
    }  
   

return seconds;
}





void sampleConvergenceAtRound(int sampledRound, char *PrefixName, double *Qs, int Ns, PeersT *peers, unsigned long tSum, PState *pstates, InputT *data, PVariate *dparams) {

    char outname[FLEN];
    snprintf(outname, FLEN, "CSV/Round-%d-%s.csv", sampledRound, PrefixName);
    FILE *out = fopen(outname, "w");

    if (!out) {
    
        std::cout << "Error opening "<<outname<<std::endl;
    
    } else {
        
        double npeers = (double) peers->num_peers;
       
    
        fprintf(out, "ID,EstPeers,relErrP,Dtype,ParamX,ParamY,PLen,Bound,Collps,IAlpha,FAlpha,EstCount,ExactCount,errCount,");
        logHeader(out, Qs, Ns);
        fprintf(out, "\n");
        
        for(int id = 0; id < peers->num_peers; ++id) {
            
            
            double estP = 0, errP = 0;
            if (pstates[id].q > 0){
               estP = 1.0/pstates[id].q;
            } else {
                estP = 1; 
            }
            errP = std::abs(estP - npeers)/npeers;

            double estItems = (pstates[id].L.negapop+pstates[id].L.posipop)*estP;
            double CountErr = std::abs(estItems - tSum)/(double)tSum;
            
            fprintf(out, "%d,%.2f,%.6f,", pstates[id].peerId, estP, errP);
            
            fprintf(out, "%s,%.6f,%.6f,%ld,", data->dname, dparams[id].x, dparams[id].y, dparams[id].len);
            
            fprintf(out, "%d,%d,%.6f,%.6f,", pstates[id].L.bound, pstates[id].L.collapses, data->initial_alpha, pstates[id].L.alpha);

            fprintf(out, "%.3f,%lu,%.3f,", estItems, tSum, CountErr);
            
            logAvgQuantiles(Qs, Ns, &pstates[id].L, out);

            fprintf(out,"\n");
        }
    fclose(out);
    out = NULL;
    }//fi
}


