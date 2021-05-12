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
#include "Quantiles.h"

//#define NDEBUG
#include <assert.h>



char VERSION[] = "DistributedSketch (v1.0)";        
char IGRAPHVERSION[NLEN];                           
const int PRIME_NUMBER = 28547;                    




int main(int argc, char *argv[]) {

    //\********************************************* Step 1. 
    
    if (argc < MIN_PARAMS) {    // at least parameters -G -P -R -i/-d are necessary
        usage(argv[0]);
        std::cout << "Not enough parameters on the command line\n";
        exit(EPARAMS);
    }

    PeersT peers;               // options related to the P2P net peers
    GraphT g;                   // options related to the underlying Graph
    InputT data;                // options related to the Input Data to process

    
    int res = getUserParams(argc, argv, &peers, &g, &data);
    if (res == EPARAMS) {
        std::cout << "Bad command line configuration!!!"<<std::endl;
        usage(argv[0]);
        onClose(&peers, &g, &data);
        exit(EPARAMS);
    }//fi EPARAMS



    char PrefixName[FLEN]; 
    unsigned long timestamp = composeFileName(PrefixName, FLEN, g.gname, peers.num_peers, peers.rounds, data.dname, peers.fan_out, peers.churntype, peers.peer_failure_p);
   
  
    FILE *logf = openLogFile(PrefixName, strlen(PrefixName));
    if (logf) {        
        fprintf(logf, "\t%s..\n", VERSION);
        
        get_iGraphVersion();
        fprintf(logf, "\tUsing library %s\n\n", IGRAPHVERSION);
        
        printPeersT(&peers, logf);

        #ifdef POSI
        fprintf(logf, "\tPositive only items\n");
        fprintf(logf,"--------------------------------\n");
        #endif

    } else {
        printPeersT(&peers, stdout);
    }//fi logf

    


    //\********************************************* Step 2. 
    
    g.nodes = peers.num_peers;                          
    double npeers = (double)peers.num_peers;            
     
    int kdegree = peers.num_peers - 1;                   
    
    double radius = std::sqrt(100.0/npeers);            

    double BA_power = DEFAULT_BA_power;                 
    int BA_m = DEFAULT_BA_m;                            
    double BA_A = DEFAULT_BA_A;                         
   
    double ER_paramP = (10.0/npeers);                   
    double ER_paramM = std::ceil(pow(npeers,2)/3.0);    


    int trials = 0; 
    double graphElapsedSecs = createP2PGraph(&g, kdegree, radius, ER_paramP, ER_paramM, BA_power, BA_A, BA_m, &trials);

    
    if (logf) fprintf(logf, "\nConnected %s random graph in %.6f seconds and %d trials\n", g.gname, graphElapsedSecs, trials);
    
    getGraphProperties(&g);

    printGraphT(&g, logf);


    //\********************************************* Step 3. 
    
    double initial_gamma = (1.0 + data.initial_alpha)/(1.0 - data.initial_alpha);
    double NULLBOUND = pow(initial_gamma, -MIN_KEY);

    SketchF FullS; 
    initFullSketch(&FullS, data.sketch_bound, data.initial_alpha, initial_gamma, NULLBOUND);

    if (logf) fprintf(logf, "\nInitial gamma %.16f, NULLBOUND %.16f\n", initial_gamma, NULLBOUND);

    double Qs[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}; 
    int Ns = sizeof(Qs)/sizeof(double);         
    
    PState pstates[peers.num_peers];


    for(int id = 0; id < peers.num_peers; ++id) {
        pstates[id].peerId = id;
        pstates[id].epoch = 0;
        pstates[id].q = 0;
        pstates[id].hasConverged = false;   
        initSketchType(&(pstates[id].L), data.sketch_bound, data.initial_alpha, initial_gamma, NULLBOUND);
    }//for peers
    pstates[0].q = 1;                       
    

    unsigned long tSum = 0;                           
    PVariate *dparams = NULL;
    
    ItemGenerator streamer;

    double iTotTime = 0.0, itime = 0.0;     
    Timer iTime;
    std::cout<< time(NULL)-timestamp << ": Local datasets initialization"<<std::endl;
    

    #ifdef BUFFERED
        
        if (data.fileflag) 
        {

            std::string name = data.filename;
            std::size_t p = name.find_last_of("."); 
            char bname[strlen(data.filename)+10];
            
            for(int id = 0; id < peers.num_peers; ++id) {
                sprintf(bname, "%s-%d.%s", name.substr(0, p).c_str(), id, name.substr(p+1).c_str());
                
                startTimer(&iTime);
                pstates[id].buffer = getDataFromBinaryFile(bname, &pstates[id].len);
                stopTimer(&iTime);
                itime = getElapsedMilliSecs(&iTime);    
                
                iTotTime += itime;                      
                tSum += pstates[id].len;                
                
            }//for peers
        } 
        else 
        {
            dparams = (PVariate *)malloc( peers.num_peers * sizeof(PVariate));
            if (!dparams) {
                fprintf(stderr, "Not enough memory to allocate variables: %s \n",strerror(errno));
            }

            tSum = computeParamsForPeers(data.dtype, data.x, data.y, data.inputLen, data.process_seed, peers.num_peers, dparams, initial_gamma);
            
                for(int id = 0; id < peers.num_peers; ++id) {

                    startTimer(&iTime);   

                    #ifdef POSI 
                        
                    pstates[id].buffer = generateRandomData(data.dtype, dparams[id].x, dparams[id].y, dparams[id].len, &NULLBOUND, dparams[id].seed);
                    pstates[id].len = dparams[id].len;
                    #else
                        
                        pstates[id].buffer = generateRandomData(data.dtype, dparams[id].x, dparams[id].y, dparams[id].len, NULL, dparams[id].seed);
                        pstates[id].len = dparams[id].len;
                    #endif
                    
                    stopTimer(&iTime);
                    itime = getElapsedMilliSecs(&iTime);

                    iTotTime += itime;
                }//for peers 
        }//fi
    #else 
        dparams = (PVariate *)malloc( peers.num_peers * sizeof(PVariate));
        if (!dparams) {
            fprintf(stderr, "Not enough memory to allocate variables: %s \n",strerror(errno));
        }

        tSum = computeParamsForPeers(data.dtype, data.x, data.y, data.inputLen, data.process_seed, peers.num_peers, dparams, initial_gamma);
    #endif


    if (logf) fprintf(logf, "\nOverall dataset with %lu items for %d peers loaded in %.6f ms\n", tSum, peers.num_peers, iTotTime);
    

    //\********************************************* Step 4. 
    
    Timer localPTime; 
    double secsToFillSketches = 0.0;    
    double secsToFillFullS = 0.0;       
    unsigned long Tcounter = 0;


    std::cout<< time(NULL)-timestamp << ": Fill local sketches"<<std::endl;
    for(int peerId = 0; peerId < peers.num_peers; ++peerId) 
    {

        #ifdef BUFFERED
            secsToFillSketches += fillSketches(&pstates[peerId].L, pstates[peerId].buffer, pstates[peerId].len);
            
            secsToFillFullS += fillFullSketches(&FullS, pstates[peerId].buffer, pstates[peerId].len);

            free(pstates[peerId].buffer);
            pstates[peerId].buffer = NULL;
            pstates[peerId].len = 0;
        
        #else
            unsigned long localseed = std::chrono::system_clock::now().time_since_epoch().count() + peerId;
            dparams[peerId].seed = localseed;
            initGenerator(&streamer, data.dtype, dparams[peerId].x, dparams[peerId].y, localseed);

            double item = 0.0;
            unsigned long counter = 0;
            

            while (counter < dparams[peerId].len) {
            
                item = streamer.randomizer();
            
                #ifdef POSI
                    if (item > NULLBOUND){
                        ++counter;

                        addItemToSketch(&pstates[peerId].L, item);
                        addItemToFullSketch(&FullS, item);
                    }
                #else 
                    ++counter;
                    
                    addItemToSketch(&pstates[peerId].L, item);
                    addItemToFullSketch(&FullS, item);
                #endif
            }//wend
            Tcounter += dparams[peerId].len;
        #endif 
    }//for peerId
    
    std::cout<< time(NULL)-timestamp << ": All Sketches filled"<<std::endl;
    assert(tSum == Tcounter);

    if (logf) fprintf(logf, "FULL dataset of %lu items: final alpha: %.6f, collapses: %d\n", tSum, FullS.alpha, FullS.collapses);

       
    
    //\********************************************* Step 7. 


    ChurnEvent churnEv;
    initChurning(&churnEv, 0, true, peers.num_peers, peers.churntype);
    
    if (churnEv.churnmode) {
        
        char filename[FLEN];
        snprintf(filename, FLEN-1, "Logs/Peers-%s.log", PrefixName);

        churnEv.fp = fopen(filename, "w");
        if (churnEv.fp==NULL){
            std::cout << "Error opening "<<filename<<std::endl;
        } else {
            fprintf(churnEv.fp, "Round,Peer,isOFF\n");
        }//fi fp
    } else {
        churnEv.fp = NULL;
    }//fi

    initChurnModel(&churnEv);


    int roundsExecuted = 0;                     
    std::cout<< time(NULL)-timestamp << ": Starting Gossip"<<std::endl;
    
    double gossipTime = GossipSketches(PrefixName, &roundsExecuted, pstates, &g, &peers, &churnEv, Qs, Ns, &data, tSum, dparams);
    
    std::cout<< time(NULL)-timestamp << ": Gossip ended"<<std::endl;
   

    if (logf) {
        fprintf(logf, "Completed distributed gossiping in %.16f seconds and %d rounds/%d\n", gossipTime, roundsExecuted, peers.rounds);

    }

    destroyChurning(&churnEv);
    
    
    //\********************************************* Step 8. 


    std::cout<< time(NULL)-timestamp << ": Computing Quantiles over the sequential sketch"<<std::endl;

    if (logf){
        fprintf(logf, "Full sketch quantiles over %.3lu items\n", FullS.posipop+FullS.negapop);
    }
    
    char FullDatafile[FLEN];
    snprintf(FullDatafile, FLEN, "CSV/FullD-%s.csv", PrefixName);
    FILE *Dfile = fopen(FullDatafile, "w");
    if (Dfile) {
        getFullQuantiles(Qs, Ns, &FullS, Dfile);
        fclose(Dfile);
        Dfile = NULL;
    } else {
        std::cout << "Error opening "<<FullDatafile<<std::endl;
    }//fi Dfile
 

    //\********************************************* Step 9. 

    std::cout<< time(NULL)-timestamp << ": Saving local statistics for peers"<<std::endl;


    //\********************************************* Step 10. 
    if (dparams){
        free(dparams);
    }

    
    onClose(&peers, &g, &data);
    
    
    if (logf){
        fclose(logf);
    }

    std::cout<< time(NULL)-timestamp << ": Processing Ended!\n";
    return 0;
}


