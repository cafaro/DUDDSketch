/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/

#include "GraphNet.h"     


void freePeersT(PeersT *p){ }

void initPeersT(PeersT *p) {

    p->num_peers = 0;
    p->fan_out = 0;
    
    p->churntype = NOCHURN;
    
    p->peer_failure_p = 0.0;

    p->rounds = 0;
}




void printPeersT(PeersT *p, FILE *fp) {
    FILE *out = fp?fp:stdout;
    
    fprintf(out, "Peers: %d\n",p->num_peers);
    fprintf(out, "Fan-out: %d\n", p->fan_out);
    
    fprintf(out, "Rounds to execute: %d\n", p->rounds);
    
    fprintf(out, "Churning model: %s\n", p->churnname);

    fprintf(out, "Failure prob: %.3f\n", p->peer_failure_p);
    fprintf(out,"--------------------------------\n");
}




double createP2PGraph(GraphT *g, int kdegree, double radius, double ERparamP, double ERparamM, double BApower, double BAa, int BAm, int *trials) {
    
    igraph_i_set_attribute_table(&igraph_cattribute_table);
 
    unsigned long int seed = g->initial_seed; 
    
    igraph_rng_seed(igraph_rng_default(), seed);

    Timer gtime;

    startTimer(&gtime);
    switch (g->graph_type) {

            case BARABASI:
                g->graph = createBarabasiAlbertGraph(g->nodes, BApower, BAm, BAa, trials); 
                break;

            case ERDOS:
                g->graph = createErdosRenyiGraph(g->nodes, IGRAPH_ERDOS_RENYI_GNP, ERparamP, trials);
                break;
                
            case GEOMETRIC:
                g->graph = createGeometricConnectedGraph(g->nodes, radius, trials);
                break;

            case REGULAR:
                g->graph = createRegularGraph(g->nodes, kdegree, trials);
                break;
            
            default:
                g->graph = createRegularGraph(g->nodes, kdegree, trials);
                break;

        }//switch
    stopTimer(&gtime);
    
    g->hasGraph = true; 
    
    double elapsedSeconds = getElapsedSeconds(&gtime);

return elapsedSeconds;
}


