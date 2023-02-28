#include <chrono>
#include <cstdio>
#include <igraph/igraph.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <map>
#include "uddsketch/uddsketch.h"
#include "rnd.hpp"
#include "cmdline_parser.hpp"

using namespace std;
using namespace cmdline_options;

using sketch_type = UDDSketch;
using myclock=std::chrono::high_resolution_clock;
using unifint=std::uniform_int_distribution<unsigned>;
using unifreal=std::uniform_real_distribution<double>;

struct opts {
    // uddsketch
    double  a;              // value of parameter alpha for the algorithm's sketch
    int     m;              // value of parameter m (max number of buckets in the sketch)
    string  dist;           // random distribution of generated items
    double  p;              // first parameter of input stream random distribution
    double  q;              // second parameter of input stream random distribution
    uint    s;              // seed
    bool    csv;            // csv output
    // p2p
    long    n;                   // total number of items
    int     peers;               // number of peers
    int     fanOut;              // number of neighbors to talk with at each round
    int     graphType;           // type of graph to generate
    int     roundsToExecute;     // number of rounds to be executed
    string  churnType;           // model of churning to be used
    double  peerFailureProb;     // probability of a peer to fail in failstop churning
    string  inputFileName;       // file containing the input stream
    string  outputFilename;      // output filename for statistics
    bool    tracePeerLifetime;   // whether peer lifetime is to be tracked
};

struct PeerStats {
    double sketchInitialAlpha;
    int sketchMaxBuckets;
    double sketchAlpha;
    int sketchBuckets;
    int sketchBytes;
    std::vector<double> estimatedQuantiles;
    double p_estimate;
    double n_estimate;
};

igraph_t generateGeometricGraph(igraph_integer_t n, igraph_real_t radius);
igraph_t generateBarabasiAlbertGraph(igraph_integer_t n, igraph_real_t power, igraph_integer_t m, igraph_real_t A);
igraph_t generateErdosRenyiGraph(igraph_integer_t n, igraph_erdos_renyi_t type, igraph_real_t param);
igraph_t generateRegularGraph(igraph_integer_t n, igraph_integer_t k);
igraph_t generateRandomGraph(int type, int n);
void printGraphType(int type);
double compute_shiftedPareto_variate(double beta, double alpha, double mu, double u);
double compute_exponential_variate(double lambda, double u);

int main(int argc, char **argv)
{
    // parse command-line parameters
    opts opts{};
    options_parser parser(argc, argv);
    parser.add_option<double>({opts.a, "a", "", "parameter alpha for sketch accuracy", false, 0.001, {}});
    parser.add_option<int>({opts.m, "m", "", "parameter m for sketch max size", false, 200, {}});
    parser.add_option<std::string>({opts.dist, "d", "", "input distribution: unif|norm|exp|adv", false, "unif", {"unif", "norm", "exp", "adv"}});
    parser.add_option<double>({opts.p, "p", "", "random distribution first parameter", false, 1.0, {}});
    parser.add_option<double>({opts.q, "q", "", "random distribution second parameter", false, 1000.0, {}});
    parser.add_option<unsigned>({opts.s, "s", "", "random distribution seed", false, 0, {}});
    parser.add_option<long>({opts.n, "n", "", "number of items", false, 1000000, {}});
    parser.add_option<int>({opts.peers, "np", "", "number of peers", false, 1000, {}});
    parser.add_option<int>({opts.fanOut, "f", "", "fanout", false, 1, {}});
    parser.add_option<int>({opts.graphType, "g", "", "graph type: 1:geometric|2:barabasi|3:erdos|4:regular", false, 2, {1, 2, 3, 4}});
    parser.add_option<int>({opts.roundsToExecute, "r", "", "rounds to be executed", false, 10, {}});
    parser.add_option<std::string>({opts.churnType, "", "churn", "churning model: none|failstop|yao|yaoexp", false, "none", {"none", "failstop", "yao", "yaoexp"}});
    parser.add_option<std::string>({opts.inputFileName, "if", "", "input filename", false, "", {}});
    parser.add_option<std::string>({opts.outputFilename, "of", "", "output filename", false, "", {}});
    parser.add_option<bool>({opts.tracePeerLifetime, "tp", "", "trace peer lifetime", false, false, {}});
    parser.add_option<double>({opts.peerFailureProb, "fp", "", "probability of a peer to fail in failstop churning", false, 0.3, {}});

    try {
        parser.parse_input();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        parser.print_usage();
        return 1;
    }

    vector<double> input_stream;

    if (opts.inputFileName.size() > 0) {
        FILE *inputFile = fopen(opts.inputFileName.c_str(), "r");
        if (inputFile == NULL) {
            std::cerr << "Error opening input file.\n";
            return -1;
        }
        double item;
        long i = 0;
        while(fscanf(inputFile, "%lf", &item)!=EOF) {
            input_stream.push_back(item);
            i++;
        }
        fclose(inputFile);
        opts.n = i;
    } 
 
    vector<long> peerLastItem(opts.peers, 0);

    for(int i = 0; i < (opts.peers - 1); i++){
        peerLastItem[i] = floor((float) (i+1) * ((float)opts.n/(float)opts.peers) - 1);
    }

    peerLastItem[opts.peers - 1] = opts.n - 1;

    long sum = peerLastItem[0] + 1;
    long substream_max_length = sum;
    for(int i = 1; i < opts.peers; i++) {
        long len = peerLastItem[i] - peerLastItem[i-1];
        sum += len;
        substream_max_length = ( substream_max_length < len)? len : substream_max_length;
    }

    if(sum != opts.n) {
        fprintf(stdout, "ERROR: ni = %ld != sum = %ld\n", opts.n, sum);
    	exit(EXIT_FAILURE);
    }

    auto u = rnd_gen<unifint>(unifint::param_type{0, std::numeric_limits<unsigned>::max()}, opts.s);
    unsigned local_seed = 0;
    
    if (opts.inputFileName.size() == 0) {
        local_seed = u.get_value();

        std::shared_ptr<std::vector<double>> in_stream;
        if (opts.dist == "unif" || opts.dist == "norm" || opts.dist == "exp") {
            if (opts.dist == "unif") {
                auto rg = rnd_gen<unifreal>(unifreal::param_type{opts.p, opts.q}, local_seed);
                in_stream = rg.get_stream(opts.n);
            } else if (opts.dist == "norm") {
                auto rg = rnd_gen<std::normal_distribution<>>(std::normal_distribution<>::param_type{opts.p, opts.q}, local_seed);
                in_stream = rg.get_stream(opts.n);
            } else if (opts.dist == "exp") {
                auto rg = rnd_gen<std::exponential_distribution<>>(std::exponential_distribution<>::param_type{opts.p}, local_seed);
                in_stream = rg.get_stream(opts.n);
            }
            for (auto i : *in_stream) {
                input_stream.push_back(i);
            }
        }

        if (opts.dist == "adv") {
            double gamma = (1+opts.a)/(1-opts.a);
            int k = 1;
            if (opts.peers > 10) k = 1;
            if (opts.peers > 100) k = 10;
            if (opts.peers > 1000) k = 100;
            
            int groups = opts.peers / k;
            int rem = opts.peers % k;
            int maxid = std::ceil(std::log(opts.q)/std::log(gamma));
            int minid = std::ceil(std::log(opts.p)/std::log(gamma));
            int idinc = (maxid-minid+1)/groups;
            //fprintf(stderr, "Peers with same bucket ids: %d, minid: %d, maxid: %d, id increment: %d\n", groups, minid, idinc*groups, idinc);
            {
                auto rg0 = rnd_gen<unifreal>(unifreal::param_type{pow(gamma, minid), pow(gamma, minid+idinc-1)}, local_seed);
                in_stream = rg0.get_stream(peerLastItem[k+rem-1]+1);
                for (auto i : *in_stream) {
                    input_stream.push_back(i);
                }
            }
            int c = minid + idinc;
            for (int i = k+rem-1; i < opts.peers-k; i=i+k) {
                auto rg = rnd_gen<unifreal>(unifreal::param_type{pow(gamma, c), pow(gamma, c+idinc-1)}, local_seed);
                c += idinc;
                in_stream = rg.get_stream(peerLastItem[i+k] - peerLastItem[i]);
                for (auto i : *in_stream) {
                    input_stream.push_back(i);
                }
            }
        }
    }

    double qs_rank[99];
    for (int i = 1; i <= 99; i++){
        qs_rank[i-1] = i/100.0;
    }

    bool outputOnFile = (opts.outputFilename.size() > 0);

    // seed igraph PRNG
    igraph_rng_seed(igraph_rng_default(), u.get_value());

    auto start_time = myclock::now();

    // generate a connected random graph
    igraph_t graph = generateRandomGraph(opts.graphType, opts.peers);

    // determine minimum and maximum vertex degree for the graph
    if (!outputOnFile) {
        igraph_vector_t result;
        igraph_real_t mindeg;
        igraph_real_t maxdeg;
        igraph_integer_t diameter;

        igraph_vector_init(&result, 0);
        igraph_degree(&graph, &result, igraph_vss_all(), IGRAPH_ALL, IGRAPH_NO_LOOPS);
        igraph_vector_minmax(&result, &mindeg, &maxdeg);
        igraph_diameter(&graph, &diameter, 0, 0, 0, 0, IGRAPH_UNDIRECTED);
    
        printf("Minimum degree is %d, Maximum degree is  %d, Diameter is %d\n", (int) mindeg, (int) maxdeg, (int)diameter);
    }

    // Apply Space Saving to each peer' substream
    if (!outputOnFile) {
        printf("\nRunning UDSketch algorithm on each peer's local stream...\n");
    }

    vector<UDDSketch> sketches(opts.peers, UDDSketch(opts.a, opts.m));
    
    // processing local streams
    long start = 0;
    for(int peerID = 0; peerID < opts.peers; peerID++){
    
        for (long i = start; i <= peerLastItem[peerID]; i++) {
            sketches[peerID].add(input_stream.at(i));
        }

        start = peerLastItem[peerID] + 1;
    }

    if (!outputOnFile) {
        printf("Sequential UDDSketch done!\n");
    } 
    
    // Uncomment and give an output file to see peers' sketches after sequential step.
    //
    // if (outputOnFile) {
    //     ofstream output_file(opts.outputFilename, std::ios::out);
    //     for(int peerID = 0; peerID < opts.peers; peerID++){
    //        auto store = sketches[peerID].get_store_snap();
    //        output_file << peerID << ", ";
    //        for (auto it : store) {
    //          output_file << it.first << ":" << it.second << ", "; 
    //        }
    //        output_file << "\n";
    //     }
    //     exit(EXIT_SUCCESS);
    // }

    // this is used to estimate the number of peers
    double inverseP_estimate[opts.peers];
    memset(inverseP_estimate, 0, sizeof(double)*opts.peers);
    inverseP_estimate[0] = 1.0;
    
    double prev_inverseP_estimate[opts.peers];
    
    bool peer_off[opts.peers];
    for (int i = 0; i < opts.peers; i++)
        peer_off[i] = (opts.churnType == "none" || opts.churnType == "failstop")? false : true;
    
    int num_peers_off = (opts.churnType == "none" || opts.churnType == "failstop")? 0 : opts.peers;

    double peer_avg_on_duration[opts.peers];
    double peer_avg_off_duration[opts.peers];
    
    std::random_device rd1;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd1()); //Standard mersenne_twister_engine seeded with rd1()
    std::uniform_real_distribution<> unif(0.0, 1.0);
    
    double alpha;
    double beta;
    double mu;
    
    alpha = 3.0;
    beta = 1.0;
    mu = 1.01;
    for (int i = 0; i < opts.peers; i++) {
        peer_avg_on_duration[i] = compute_shiftedPareto_variate(beta,alpha,mu,unif(gen)); //generate a shifted pareto (type II) 
    }
    
    beta = 2.0;
    for (int i = 0; i < opts.peers; i++) {
        peer_avg_off_duration[i] = compute_shiftedPareto_variate(beta,alpha,mu,unif(gen)); //generate a shifted pareto (type II) 
    }

    int peer_onoff_period[opts.peers];
    memset(peer_onoff_period, 0, opts.peers*sizeof(int));
        
    if (!outputOnFile) {
        printf("\nStarting distributed algorithm...\n");
    }
    
    ofstream trace_file;
    if (opts.tracePeerLifetime) {
        trace_file.open("trace.csv", std::ios::out);
        trace_file << "round,peer,state\n";
    }

    int rounds = 0;
    while(rounds < opts.roundsToExecute){
        for(int peerID = 0; peerID < opts.peers; peerID++) {
            if (opts.churnType == "failstop") {
                if (!peer_off[peerID]) {
                    if (unif(gen) < opts.peerFailureProb) {
                        peer_off[peerID] = true;
                        num_peers_off++;
                    }
                }
            } else if (opts.churnType == "yao" || opts.churnType == "yaoexp") {
                if (peer_onoff_period[peerID] > 0){
                    peer_onoff_period[peerID]--;
                } else if (!peer_off[peerID]) {
                    beta = 2.0 * peer_avg_off_duration[peerID];
                    alpha = 3.0;
                    mu = 0.0;
                    peer_onoff_period[peerID] = (int) round(compute_shiftedPareto_variate(beta,alpha,mu,unif(gen)));
                    if (peer_onoff_period[peerID] > 0) {
                        peer_off[peerID] = true;
                        num_peers_off++;
                    }
                } else {
                    if (opts.churnType == "yaoexp") {
                        double exp_lambda = 1.0/peer_avg_on_duration[peerID];
                        peer_onoff_period[peerID] = (int) round(compute_exponential_variate(exp_lambda,unif(gen)));
                    } else {
                        beta = 2.0 * peer_avg_on_duration[peerID];
                        alpha = 3.0;
                        mu = 0.0;
                        peer_onoff_period[peerID] = (int) round(compute_shiftedPareto_variate(beta,alpha,mu,unif(gen)));
                    }
                    if (peer_onoff_period[peerID]>0) {
                        peer_off[peerID] = false;
                        num_peers_off--;
                    }
                }
                if (opts.tracePeerLifetime)
                    trace_file << rounds << "," << peerID << "," << peer_off[peerID] << "\n";
            }
        }

        for(int peerID = 0; peerID < opts.peers; peerID++){
            
            if (peer_off[peerID])
                continue;
            
            // determine peer neighbors
            igraph_vector_t neighbors;
            igraph_vector_init(&neighbors, 0);
            if (opts.churnType != "none") {
                igraph_vector_t neighbors_tmp;
                igraph_vector_init(&neighbors_tmp, 0);
                igraph_neighbors(&graph, &neighbors_tmp, peerID, IGRAPH_ALL);
                long neighbors_tmp_size = igraph_vector_size(&neighbors_tmp);
                for (int i = 0; i < neighbors_tmp_size; i++) {
                    int id = (int) VECTOR(neighbors_tmp)[i];
                    if (!peer_off[id])
                        igraph_vector_push_back (&neighbors, id);
                }
                igraph_vector_destroy(&neighbors_tmp);
            } else {
                igraph_neighbors(&graph, &neighbors, peerID, IGRAPH_ALL);
            }
            
            long neighborsSize = igraph_vector_size(&neighbors);
            if(opts.fanOut < neighborsSize){
                // randomly sample fanOut neighbors
                igraph_vector_shuffle(&neighbors);
                igraph_vector_remove_section(&neighbors, opts.fanOut, neighborsSize-1);
            }

            neighborsSize = igraph_vector_size(&neighbors);
            for(int i = 0; i < neighborsSize; i++){
                int neighborID = (int) VECTOR(neighbors)[i];

                sketches[peerID].mergeAndMean(sketches[neighborID]);
                sketches[neighborID] = sketches[peerID];
                
                double mean = (inverseP_estimate[peerID] + inverseP_estimate[neighborID]) / 2.0;
                inverseP_estimate[peerID] = mean;
                inverseP_estimate[neighborID] = mean;
            }
            igraph_vector_destroy(&neighbors);
        }

        rounds++;
        cerr << "\rRounds processed: " << rounds << " Peers failed: " << num_peers_off << "      ";
    }

    if (!outputOnFile) {
        printf("\n\nThe distributed algorithm is terminated\n");
    }

    // output results
    double p_estimate[opts.peers];
    for(int i = 0; i<opts.peers; i++){
        if (inverseP_estimate[i] > 0)
            p_estimate[i] = 1.0/inverseP_estimate[i];
        else
            p_estimate[i] = 0;
    }

    if (!outputOnFile) {
        double p_estimate_value = 0.0;
        for(int i = 0; i < opts.peers; i++) {
            p_estimate_value += p_estimate[i];
        }
        printf("Estimate of number of peers (average value): %6.16f\n", p_estimate_value/opts.peers);
    }

    if (outputOnFile) {
        // collect peers stats
        PeerStats peerStats[opts.peers];
        for (int i = 0; i < opts.peers; i++) {
            if (opts.roundsToExecute > 0) 
                sketches[i].sketch_restore_from_p2p_average(p_estimate[i]);
            peerStats[i].sketchInitialAlpha = sketches[i].get_initial_alpha();
            peerStats[i].sketchMaxBuckets = sketches[i].get_max_number_buckets();
            peerStats[i].sketchAlpha = sketches[i].get_alpha();
            peerStats[i].sketchBuckets = sketches[i].get_number_buckets();
            peerStats[i].sketchBytes = sketches[i].get_size_bytes();
            peerStats[i].p_estimate = p_estimate[i];
        
            for (int c = 1; c <= 99; c++){
                peerStats[i].estimatedQuantiles.push_back(sketches[i].get_quantile_int(qs_rank[c-1]));
            }
        }

        // sequential algorithm sketch
        sketch_type seq_sketch(opts.a, opts.m);
        
        for (int i = 0; i < opts.n; i++) {
            seq_sketch.add(input_stream.at(i));
        }

        PeerStats groundStats;
        groundStats.sketchInitialAlpha = seq_sketch.get_initial_alpha();
        groundStats.sketchMaxBuckets = seq_sketch.get_max_number_buckets();
        groundStats.sketchAlpha = seq_sketch.get_alpha();
        groundStats.sketchBuckets = seq_sketch.get_number_buckets();
        groundStats.sketchBytes = seq_sketch.get_size_bytes();
        groundStats.p_estimate = opts.peers;

        for (int i = 1; i <= 99; i++){
            groundStats.estimatedQuantiles.push_back(seq_sketch.get_quantile_int(qs_rank[i-1]));
        }

        auto seq_store = seq_sketch.get_store_snap();

    
        ofstream output_file(opts.outputFilename, std::ios::out);
        output_file.precision(10);
        output_file << "peerid,initial_alpha,max_number_of_buckets,final_alpha,num_of_buckets,sketch_size,p_estimate";
        for (int i = 1; i <= 99; i++){
            output_file << ",q" << i;
        } 
        // for (int i = 1; i <= groundStats.sketchBuckets; i++){
        //     output_file << ",b" << i;
        // } 
        output_file << "\n";
        output_file << "-1,"
                    << groundStats.sketchInitialAlpha << "," 
                    << groundStats.sketchMaxBuckets << "," 
                    << groundStats.sketchAlpha << "," 
                    << groundStats.sketchBuckets << "," 
                    << groundStats.sketchBytes << "," 
                    << groundStats.p_estimate;
        for (int i = 1; i <= 99; i++){
            output_file << "," << groundStats.estimatedQuantiles[i-1];
        }
        // for (auto it : seq_store) {
        //     output_file << "," << it.first << ":" << it.second;
        // }
        output_file << "\n";
        
        for (int i = 0; i < opts.peers; i++) {
            output_file << i << "," 
                        << peerStats[i].sketchInitialAlpha << "," 
                        << peerStats[i].sketchMaxBuckets << "," 
                        << peerStats[i].sketchAlpha << "," 
                        << peerStats[i].sketchBuckets << "," 
                        << peerStats[i].sketchBytes << "," 
                        << peerStats[i].p_estimate ;
            for (int c = 1; c <= 99; c++){
                //output_file << "," <<  fabs(peerStats[i].estimatedQuantiles[c-1] - groundStats.estimatedQuantiles[c-1])/groundStats.estimatedQuantiles[c-1];
                output_file << "," << peerStats[i].estimatedQuantiles[c-1];
            }
            // auto peerStore = sketches[i].get_store_snap();
            // for (auto it : peerStore) {
            //     output_file << "," << it.first << ":" << it.second;
            // }
            output_file << "\n";
        }
    }

    return 0;
}

igraph_t generateGeometricGraph(igraph_integer_t n, igraph_real_t radius)
{
    igraph_t G_graph;
    igraph_bool_t connected;

    // generate a connected random graph using the geometric model
    igraph_grg_game(&G_graph, n, radius, 0, 0, 0);

    igraph_is_connected(&G_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&G_graph);
        igraph_grg_game(&G_graph, n, radius, 0, 0, 0);

        igraph_is_connected(&G_graph, &connected, IGRAPH_WEAK);
    }

    return G_graph;
}

igraph_t generateBarabasiAlbertGraph(igraph_integer_t n, igraph_real_t power, igraph_integer_t m, igraph_real_t A)
{

    // n = The number of vertices in the graph
    // power = Power of the preferential attachment. The probability that a vertex is cited is proportional to d^power+A, 
    // where d is its degree, power and A are given by arguments. In the classic preferential attachment model power=1
    // m = number of outgoing edges generated for each vertex
    // A = The probability that a vertex is cited is proportional to d^power+A, where d is its degree, 
    // power and A are given by arguments

    igraph_t BA_graph;
    igraph_bool_t connected;

    // generate a connected random graph using the Barabasi-Albert model
    igraph_barabasi_game(/* graph=    */ &BA_graph,
                         /* n=        */ n,
                         /* power=    */ power,
                         /* m=        */ m,
                         /* outseq=   */ 0,
                         /* outpref=  */ 0,
                         /* A=        */ A,
                         /* directed= */ IGRAPH_UNDIRECTED,
                         /* algo=     */ IGRAPH_BARABASI_PSUMTREE,
                         /* start_from= */ 0);


    igraph_is_connected(&BA_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&BA_graph);
        igraph_barabasi_game(/* graph=    */ &BA_graph,
                             /* n=        */ n,
                             /* power=    */ power,
                             /* m=        */ m,
                             /* outseq=   */ 0,
                             /* outpref=  */ 0,
                             /* A=        */ A,
                             /* directed= */ IGRAPH_UNDIRECTED,
                             /* algo=     */ IGRAPH_BARABASI_PSUMTREE,
                             /* start_from= */ 0);

        igraph_is_connected(&BA_graph, &connected, IGRAPH_WEAK);
    }

    return BA_graph;
}



igraph_t generateErdosRenyiGraph(igraph_integer_t n, igraph_erdos_renyi_t type, igraph_real_t param)
{
    // n = The number of vertices in the graph
    // type = IGRAPH_ERDOS_RENYI_GNM G(n,m) graph, m edges are selected uniformly randomly in a graph with n vertices.
    //      = IGRAPH_ERDOS_RENYI_GNP G(n,p) graph, every possible edge is included in the graph with probability p

    igraph_t ER_graph;
    igraph_bool_t connected;

    // generate a connected random graph using the Erdos-Renyi model
    igraph_erdos_renyi_game(&ER_graph, type, n, param, IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);

    igraph_is_connected(&ER_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&ER_graph);
        igraph_erdos_renyi_game(&ER_graph, type, n, param, IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);

        igraph_is_connected(&ER_graph, &connected, IGRAPH_WEAK);
    }

    return ER_graph;
}

igraph_t generateRegularGraph(igraph_integer_t n, igraph_integer_t k)
{
    // n = The number of vertices in the graph
    // k = The degree of each vertex in an undirected graph. For undirected graphs, at least one of k and the number of vertices must be even.


    igraph_t R_graph;
    igraph_bool_t connected;

    // generate a connected regular random graph
    igraph_k_regular_game(&R_graph, n, k, IGRAPH_UNDIRECTED, 0);

    igraph_is_connected(&R_graph, &connected, IGRAPH_WEAK);
    while(!connected){
        igraph_destroy(&R_graph);
        igraph_k_regular_game(&R_graph, n, k, IGRAPH_UNDIRECTED, 0);

        igraph_is_connected(&R_graph, &connected, IGRAPH_WEAK);
    }

    return R_graph;
}

igraph_t generateRandomGraph(int type, int n)
{
    igraph_t random_graph;

    switch (type) {
        case 1:
            random_graph = generateGeometricGraph(n, sqrt(100.0/(float)n));
            break;
        case 2:
            random_graph = generateBarabasiAlbertGraph(n, 1.0, 5, 1.0);
            break;
        case 3:
            random_graph = generateErdosRenyiGraph(n, IGRAPH_ERDOS_RENYI_GNP, 10.0/(float)n);
            // random_graph = generateErdosRenyiGraph(n, IGRAPH_ERDOS_RENYI_GNM, ceil(n^2/3));
            break;
        case 4:
            random_graph = generateRegularGraph(n, n-1);
            break;

        default:
            break;
    }

    return random_graph;

}

void printGraphType(int type)
{

    switch (type) {
        case 1:
            printf("Geometric random graph\n");
            break;
        case 2:
            printf("Barabasi-Albert random graph\n");
            break;
        case 3:
            printf("Erdos-Renyi random graph\n");
            break;
        case 4:
            printf("Regular random graph\n");
            break;

        default:
            break;
    }
}

double compute_shiftedPareto_variate(double beta, double alpha, double mu, double u) {
    return beta/pow(1.0-u,1.0/alpha) - beta + mu;
}

double compute_exponential_variate(double lambda, double u) {
    return -log(u)/lambda;
}
