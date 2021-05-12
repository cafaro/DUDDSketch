/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/


#ifndef __GRAPH_H__
#define __GRAPH_H__


#include "igraph.h"         
#include "Base.h"           


//************************************* Constants and Types

const int MAX_TRIALS_CONNECTED = 1000;          
const int EGRAPH = 700;                         
const unsigned long CONVENTIONAL_SEED = 42;     

extern char IGRAPHVERSION[NLEN];



enum GRAPH_TYPE{ UNRECOGNIZED=-1,    
                 BARABASI=1,         
                 ERDOS=2,            
                 GEOMETRIC=3,        
                 REGULAR=4           
                };



typedef struct GraphT {
    
    GRAPH_TYPE graph_type;          
    char gname[NLEN];              
    bool hasGraph;                  
    
    unsigned long initial_seed;     
    int nodes;                      
    
    igraph_t graph;                 
    igraph_real_t min_degree;       
    igraph_real_t max_degree;       
    igraph_integer_t diameter;      
    igraph_integer_t pfrom;         
    igraph_integer_t pto;           
    igraph_integer_t edges;         
    
} GraphT;



void get_iGraphVersion();

void getGraphName(GRAPH_TYPE gtype, char *gname, int len);


void freeGraphT(GraphT *g);


void initGraphT(GraphT *g);

void printGraphT(GraphT *g, FILE *fp);

void saveGraphToFile(GraphT *g, char prefix[], int flen);



void getGraphProperties(GraphT *g);



igraph_t createGeometricConnectedGraph(igraph_integer_t nodes, igraph_real_t radius, int *trials);


igraph_t createBarabasiAlbertGraph(igraph_integer_t nodes, igraph_real_t power, igraph_integer_t m, igraph_real_t A, int *trials);


igraph_t createErdosRenyiGraph(igraph_integer_t nodes, igraph_erdos_renyi_t type, igraph_real_t param, int *trials);


igraph_t createRegularGraph(igraph_integer_t nodes, igraph_integer_t degree, int *trials);


long getPeerNeighbours(int nodeI, GraphT *g, igraph_vector_t *peerNeighbours, int fan_out, bool *FailedPeers, int npeers, bool isChurnOn);


#endif // __GRAPH_H__


