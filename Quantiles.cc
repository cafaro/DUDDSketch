/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/


#include "Quantiles.h"

double quickselect(double *data, int len, int pos) {
  
  int i, ir, j, l, mid;
  double a, temp;

  l=0;
  ir=len-1;

  for(;;) {
    if (ir <= l+1) { 
      if (ir == l+1 && data[ir] < data[l]) {
	    SWAP(data[l],data[ir]);
      }
    return data[pos];
    }
    else 
    {
      mid=(l+ir) >> 1; 
      SWAP(data[mid],data[l+1]);
      if (data[l] > data[ir]) {
	    SWAP(data[l],data[ir]);
      }
      if (data[l+1] > data[ir]) {
	    SWAP(data[l+1],data[ir]);
      }
      if (data[l] > data[l+1]) {
	    SWAP(data[l],data[l+1]);
      }
      i=l+1; 
      j=ir;
      a=data[l+1];

      for (;;) { 
	    do i++; while (data[i] < a); 
	    do j--; while (data[j] > a); 
	    if (j < i) break; 
	    SWAP(data[i],data[j]);
      } 
      data[l+1]=data[j]; 
      data[j]=a;
      if (j >= pos) ir=j-1; 
      if (j <= pos) l=i;
    }
  }
}


double getQuantileFor(double q, SketchT *S, KEY_T *qIndex, COUNT_T *qCount, int *rank) {
    
    assert((q <= 1.0) && (q >= 0.0));

    double quantileEstimate = 0.0;
        
        COUNT_T n = S->posipop + S->negapop;                

        int threshold = std::ceil(q*(n-1));                 
        
        double gamma = S->gamma;                            
        
        COUNT_T count = 0;                                  
        KEY_T bkey = 0;                                     
        COUNT_T current_bcount = 0;                         

        int sign = 0;                                       
        std::map<KEY_T,COUNT_T>::iterator it;               
        std::map<KEY_T,COUNT_T>::reverse_iterator nit;      
        
        if ((double)S->negapop > 0.0) {
            sign = -1;

            nit = S->negaSketch.rbegin();

            bkey = nit->first;

            count = nit->second;
            
            current_bcount = count;
            ++nit;
            while (count <= threshold && nit != S->negaSketch.rend()) {

                bkey = nit->first;

                current_bcount = nit->second;
                
                count += current_bcount;
                ++nit;
            }//wend on negative sketch
        }//fi negasketch

        if (count <= threshold) {
            
            sign = 1;
            it = S->posiSketch.begin();
            while(count <= threshold && it != S->posiSketch.end()){

                bkey = it->first;
                current_bcount = it->second;
                
                count += current_bcount;
                ++it;
            }//wend positive sketch
        }//fi posi sketch     
        
        quantileEstimate = (sign * 2 * pow(gamma, bkey))/(gamma + 1);

        if (qIndex){
            *qIndex = bkey;             
        }
        if (qCount){
            *qCount = current_bcount;   
        }
        if (rank){
            *rank = threshold;           
        }
        
    return quantileEstimate; 
}


double querySketchForQuantile(double q, SketchF *S, int *qIndex, unsigned long *qCount, unsigned long *rank) {
    
    assert((q <= 1.0) && (q >= 0.0));

    double quantileEstimate = 0.0;
        
        unsigned long n = S->posipop + S->negapop;     

        unsigned long threshold = std::ceil(q*(n-1));  
        
        double gamma = S->gamma;                       
        
        unsigned long count = 0;                       
        int bkey = 0;                                  
        unsigned long current_bcount = 0;              

        int sign = 0;                                       
        std::map<int,unsigned long>::iterator it;           
        std::map<int,unsigned long>::reverse_iterator nit;  
        
        if (S->negapop > 0) {
            sign = -1;

            nit = S->negaSketch.rbegin();

            bkey = nit->first;

            count = nit->second;
            
            current_bcount = count;
            ++nit;
            while (count <= threshold && nit != S->negaSketch.rend()) {

                bkey = nit->first;

                current_bcount = nit->second;
                
                count += current_bcount;
                ++nit;
            }//wend on negative sketch
        }//fi negasketch

        if (count <= threshold) {
            
            sign = 1;
            it = S->posiSketch.begin();
            while(count <= threshold && it != S->posiSketch.end()){

                bkey = it->first;
                current_bcount = it->second;
                
                count += current_bcount;
                ++it;
            }//wend positive sketch
        }//fi posi sketch     
        
        quantileEstimate = (sign * 2 * pow(gamma, bkey))/(gamma + 1);

        if (qIndex){
            *qIndex = bkey;             
        }
        if (qCount){
            *qCount = current_bcount;   
        }
        if (rank){
            *rank = threshold;            
        }
        
    return quantileEstimate; 
}


void getFullQuantiles(double *Qs, int Ns, SketchF *S, FILE *fp) {
    
    double approxQ[Ns];
    int bKey[Ns];
    unsigned long bCount[Ns];
    
    if (fp) {
        fprintf(fp, "Percentile,Rank,Estimate,Key,Count (%lu)\n", S->posipop + S->negapop);
    } else {
        std::cout << "Full Sketch Quantiles estimation: "<<std::endl;
    }//fi fp

    double absErr = 0.0;
    for (int c = 0; c<Ns; ++c) {
        unsigned long rank = 0;
        
        approxQ[c] = querySketchForQuantile(Qs[c], S, &bKey[c], &bCount[c], &rank);
        
        if (fp) {
            fprintf(fp, "%.3f,%ld,%.16f,%d,%ld\n", Qs[c], rank, approxQ[c], bKey[c], bCount[c]);
        }//fi fp

    }//for c


}



void logAvgQuantiles(double *Qs, int Ns, SketchT *S, FILE *fp) {
    
    double approxQ[Ns];
    KEY_T bKey[Ns];
    COUNT_T bCount[Ns];
    
    for (int c = 0; c<Ns; ++c) {
        
        int rank = 0;
    
        approxQ[c] = getQuantileFor(Qs[c], S, &bKey[c], &bCount[c], &rank);
        if (fp) {
            fprintf(fp, "%.2f,%.16f,", Qs[c], approxQ[c]);
        } 
    
    }//for c

}


void logHeader(FILE *fp, double *Qs, int Ns) {

    for (int c = 0; c<Ns; ++c) {
        fprintf(fp, "%.2f,EstQ,", Qs[c]);
    }

}


