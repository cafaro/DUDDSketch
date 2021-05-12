/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/


#include "Graphs.h"        



void get_iGraphVersion() { 
    
    const char *string;
    int major, minor, subminor;

    igraph_version(&string, &major, &minor, &subminor);
    snprintf(IGRAPHVERSION, NLEN-1, "iGraph-%i.%i.%i", major, minor, subminor);
}



void getGraphName(GRAPH_TYPE gtype, char *gname, int len) {

    switch (gtype) {
        case BARABASI:
            memcpy(gname, "Barabasi-Albert", sizeof(char)*len);
            break;
        case ERDOS:
            memcpy(gname, "Erdos-Renyi", sizeof(char)*len);
            break;
        case GEOMETRIC:
            memcpy(gname, "Geometric", sizeof(char)*len);
            break;
        case REGULAR:
            memcpy(gname, "Regular", sizeof(char)*len);
            break;
        default:
            memcpy(gname, "Unrecognized", sizeof(char)*len);
    }//switch
}




void freeGraphT(GraphT *g) {

    if (g && g->hasGraph) {
        std::cout << "Destroying the "<< g->gname <<" graph"<< std::endl;
        igraph_destroy(&(g->graph));
    }//fi g
}



void initGraphT(GraphT *g) {
    g->graph_type = UNRECOGNIZED;
    g->hasGraph = false;
    g->initial_seed = CONVENTIONAL_SEED;
    g->nodes = 0;
}






void printGraphT(GraphT *g, FILE *fp) {
    FILE *out = fp?fp:stdout;
    
    fprintf(out, "Graph type: %s\n", g->gname);
    fprintf(out, "Graph nodes: %d\n", g->nodes);
    fprintf(out, "Graph edges: %d\n", (int)g->edges);
    fprintf(out, "Graph network diameter: %d, from %d to %d\n", (int)g->diameter, (int)g->pfrom, (int)g->pto);
    fprintf(out, "Graph min degree: %.3f\n", (double)g->min_degree);
    fprintf(out, "Graph max degree: %.3f\n", (double)g->max_degree);
    fprintf(out,"--------------------------------\n");   
}




void saveGraphToFile(GraphT *g, char prefix[], int flen) {
    
    char filename[flen+16];
    snprintf(filename, flen+15, "Dots/%s.dot", prefix);
    std::cout << "Saving "<< g->gname << " graph to Dot file "<< filename << std::endl;
    
    FILE *fp = fopen(filename, "w");
    if (fp==NULL){
        std::cout << "Error opening "<< filename << std::endl;
        return;
    }

    if (igraph_is_directed(&(g->graph))) {
        fprintf(fp, "/*directed\n");
    } else {
       fprintf(fp, "/*undirected\n");
    }
    
    igraph_write_graph_edgelist(&(g->graph), fp);
    fprintf(fp, "Min degree %.6f, max degree %.6f, diameter %d\n", (double)g->min_degree, (double)g->max_degree, (int)g->diameter);
    fprintf(fp, "-----------------*/\n");
    
    igraph_write_graph_dot(&(g->graph), fp);
    //igraph_write_graph_graphml(&(g->graph), fp, false); //to use graphML format

    fclose(fp);
}




void getGraphProperties(GraphT *g) {

    igraph_vector_t result;
    
    igraph_vector_init(&result, 0);

    igraph_degree(&(g->graph), &result, igraph_vss_all(), IGRAPH_ALL, IGRAPH_NO_LOOPS);
    
    igraph_vector_minmax(&result, &(g->min_degree), &(g->max_degree));
    
    
    igraph_vector_t path;           
    igraph_vector_init(&path, 0);   
    

    igraph_bool_t unconn = true;
   
    
    igraph_diameter(&(g->graph), &(g->diameter), &(g->pfrom), &(g->pto), &path, IGRAPH_UNDIRECTED, unconn);  
    
    g->edges = igraph_ecount(&(g->graph)); 
}




igraph_t createGeometricConnectedGraph(igraph_integer_t nodes, igraph_real_t radius, int *trials) {

    igraph_t geoGraph;
    igraph_bool_t isConnected = false;

    igraph_grg_game(&geoGraph, nodes, radius, 0, 0, 0);
    igraph_is_connected(&geoGraph, &isConnected, IGRAPH_WEAK);

    int fails = 0;
    while(!isConnected && fails < MAX_TRIALS_CONNECTED) {
        ++fails;

        igraph_destroy(&geoGraph);
        igraph_grg_game(&geoGraph, nodes, radius, 0, 0, 0);
        igraph_is_connected(&geoGraph, &isConnected, IGRAPH_WEAK);
    }//wend
    if (!isConnected){
        std::cout<<"Error in creation of a connected graph\n";
        exit(EGRAPH);
    }

    if (trials) {
        *trials = fails;
    }

    #ifndef NDEBUG
        igraph_integer_t edges = igraph_ecount(&geoGraph);
        std::cout<<"Connected Geometric Random Graph with "<< nodes << " nodes and number of edges = "<<(int)edges << ", in "<<fails<<" trials"<<std::endl;
    #endif

    return geoGraph;
}





igraph_t createBarabasiAlbertGraph(igraph_integer_t nodes, igraph_real_t power, igraph_integer_t m, igraph_real_t A, int *trials) {

    igraph_t BAGraph;
    igraph_bool_t isConnected = false;
    
    igraph_barabasi_game(&BAGraph, nodes, power, m, 0, 0, A, IGRAPH_UNDIRECTED, IGRAPH_BARABASI_PSUMTREE, 0);
    igraph_is_connected(&BAGraph, &isConnected, IGRAPH_WEAK);

    int fails = 0;
    while(!isConnected && fails < MAX_TRIALS_CONNECTED){
        ++fails;
        igraph_destroy(&BAGraph);
        igraph_barabasi_game(&BAGraph, nodes, power, m, 0, 0, A, IGRAPH_UNDIRECTED, IGRAPH_BARABASI_PSUMTREE, 0);
        igraph_is_connected(&BAGraph, &isConnected, IGRAPH_WEAK);
    }//wend
    if (!isConnected){
        std::cout<<"Error in creation of a connected graph\n";
        exit(EGRAPH);
    }

    if (trials) {
        *trials = fails;
    }

    #ifndef NDEBUG
        igraph_integer_t edges = igraph_ecount(&BAGraph);
        std::cout<<"Connected Barabasi-Albert Random Graph with "<< nodes<< " nodes and number of edges = "<<(int)edges << ", in "<<fails<<" trials"<<std::endl;
    #endif

    return BAGraph;
}




igraph_t createErdosRenyiGraph(igraph_integer_t nodes, igraph_erdos_renyi_t type, igraph_real_t param, int *trials) {
    
    igraph_t ERGraph;
    igraph_bool_t isConnected = false;

    igraph_erdos_renyi_game(&ERGraph, type, nodes, param, IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);
    igraph_is_connected(&ERGraph, &isConnected, IGRAPH_WEAK);

    int fails = 0;
    while (!isConnected && fails < MAX_TRIALS_CONNECTED) {
        ++fails;
        igraph_destroy(&ERGraph);
        igraph_erdos_renyi_game(&ERGraph, type, nodes, param, IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);
        igraph_is_connected(&ERGraph, &isConnected, IGRAPH_WEAK);
    }//wend
    if (!isConnected){
        std::cout<<"Error in creation of a connected graph\n";
        exit(EGRAPH);
    }

    if (trials){
        *trials = fails;
    }

    #ifndef NDEBUG
        igraph_integer_t edges = igraph_ecount(&ERGraph);
        std::cout<<"Connected Erdos-Renyi Random Graph with "<< nodes << " nodes and number of edges = "<<(int)edges << ", in "<<fails<<" trials"<<std::endl;
    #endif

    return ERGraph;
}





igraph_t createRegularGraph(igraph_integer_t nodes, igraph_integer_t degree, int *trials) {

    igraph_t regularGraph;
    igraph_bool_t isConnected = false;

    igraph_k_regular_game(&regularGraph, nodes, degree, IGRAPH_UNDIRECTED, 0);
    igraph_is_connected(&regularGraph, &isConnected, IGRAPH_WEAK);

    int fails = 0;
    while (!isConnected && fails < MAX_TRIALS_CONNECTED) {
        ++fails;
        igraph_destroy(&regularGraph);
        igraph_k_regular_game(&regularGraph, nodes, degree, IGRAPH_UNDIRECTED, 0);
        igraph_is_connected(&regularGraph, &isConnected, IGRAPH_WEAK);
    }//wend
    if (!isConnected){
        std::cout<<"Error in creation of a connected graph\n";
        exit(EGRAPH);
    }

    if (trials){
        *trials = fails;
    }

    #ifndef NDEBUG
        igraph_integer_t edges = igraph_ecount(&regularGraph);
        std::cout<<"k-Regular Random Graph with "<< nodes << " nodes and number of edges = "<<(int)edges << ", in "<<fails<<" trials"<<std::endl;
    #endif

    return regularGraph;
}




long getPeerNeighbours(int nodeI, GraphT *g, igraph_vector_t *peerNeighbours, int fan_out, bool *FailedPeers, int npeers, bool isChurnOn) {

    assert(g->nodes == npeers); 

    igraph_vector_init(peerNeighbours, 0);
    long neighboursCount = 0;

    if (isChurnOn) {

        igraph_vector_t tmp;
        igraph_vector_init(&tmp, 0);
        
        igraph_neighbors(&(g->graph), &tmp, nodeI, IGRAPH_ALL);
        long tmpCount = igraph_vector_size(&tmp);

        for (long int l = 0; l < tmpCount; ++l) {

            int nodeJ = (int)igraph_vector_e(&tmp, l);  
            
            if (!FailedPeers[nodeJ]) {
                igraph_vector_push_back(peerNeighbours, nodeJ); 
            }//fi not failed

        }//for l
        igraph_vector_destroy(&tmp);
    
    } else {

        igraph_neighbors(&(g->graph), peerNeighbours, nodeI, IGRAPH_ALL);
    }//fi churn

    neighboursCount = igraph_vector_size(peerNeighbours); 
    
    if ( neighboursCount > fan_out ) {
        
        igraph_vector_shuffle(peerNeighbours); 

        igraph_vector_remove_section(peerNeighbours, fan_out-1, neighboursCount-1);

        neighboursCount = igraph_vector_size(peerNeighbours);
    }

    return neighboursCount;
}

