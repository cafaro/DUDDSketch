/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/

#include "Sketch.h"     


void initSketchType(SketchT *s, int bound, double alpha, double gamma, double NULLBOUND) {

    s->bound = bound;
    s->alpha = alpha;

    s->gamma = gamma;
    if (!s->gamma) {
        s->gamma = (1+alpha)/(1-alpha);
    }//fi gamma
    
    s->base = std::log10(s->gamma);

    s->NULLBOUND = NULLBOUND; 
    if (s->NULLBOUND <0 ) {
        s->NULLBOUND = pow(s->gamma, -MIN_KEY);
    }//fi NULLBOUND
    
    s->posibins = 0;
    s->posipop = 0.0;

    s->negabins = 0;
    s->negapop = 0.0;

    s->collapses = 0;
}



void initFullSketch(SketchF *s, int bound, double alpha, double gamma, double NULLBOUND) {

    s->bound = bound;
    s->alpha = alpha;

    s->gamma = gamma;
    if (!s->gamma) {
        s->gamma = (1+alpha)/(1-alpha);
    }//fi gamma
    
    s->base = std::log10(s->gamma);

    s->NULLBOUND = NULLBOUND; 
    if (s->NULLBOUND <0 ) {
        s->NULLBOUND = pow(s->gamma, -MIN_KEY);
    }//fi NULLBOUND
    
    s->posibins = 0;
    s->posipop = 0;

    s->negabins = 0;
    s->negapop = 0;

    s->collapses = 0;
}



void debugSketchInstance(SketchT *s){
    std::cout << "Sketch with " << s->bound << " memory bound" << std::endl;
    std::cout << "Current alpha: " << s->alpha << ", gamma: " << s->gamma << std::endl;
    std::cout << "Collapses executed: " << s->collapses << std::endl;
    std::cout << "Positive: in " << s->posibins << " buckets there are " << (double)s->posipop<< " items" << std::endl;
    std::cout << "Negative: in " << s->negabins << " buckets there are " << (double)s->negapop<< " items" << std::endl;
}



void debugSketch(std::map<KEY_T, COUNT_T>& mySketch, FILE *log) {    
    
    for(auto it = mySketch.begin(); it != mySketch.end(); ++it){
        fprintf(log, "key: %d \t count: %.3f\n", (int)it->first , (double)it->second);
    }
    fprintf(log, "\n");
}



void collapseUniformly(std::map<KEY_T, COUNT_T>& mySketch) {
    
    std::map<KEY_T, COUNT_T>newSketch;         
    std::map<KEY_T, COUNT_T>::iterator it;

    it = mySketch.begin();
    if (it->first == -MIN_KEY) { 
        newSketch[-MIN_KEY] += it->second;
        ++it;
    }
    while(it != mySketch.end() ){
        double k = (double)it->first;
        KEY_T k_new = std::ceil(k/2);
        newSketch[k_new] += it->second;
        ++it;
    }//wend

    mySketch.swap(newSketch);
}




void collapseUniformlyFull(std::map<int, unsigned long>& mySketch) {
    
    std::map<int, unsigned long>newSketch;         
    std::map<int, unsigned long>::iterator it;

    it = mySketch.begin();
    if (it->first == -MIN_KEY) { 
        newSketch[-MIN_KEY] += it->second;
        ++it;
    }
    while(it != mySketch.end() ){
        double k = (double)it->first;
        int k_new = std::ceil(k/2);
        newSketch[k_new] += it->second;
        ++it;
    }//wend

    mySketch.swap(newSketch);
}



int addKeyToSketch(std::map<KEY_T, COUNT_T>&sketch, KEY_T key) {
    int res = 0;

    std::map<KEY_T, COUNT_T>::iterator it = sketch.find(key);
    if ( it == sketch.end() ) {   
        
        sketch[key] = 1;
        res = 1;
    } else {
        
        sketch[key] += 1;
        res = 0;
    }//fi 

    return res;
}




int addKeyToFullSketch(std::map<int, unsigned long>&sketch, int key) {
    int res = 0;

    std::map<int, unsigned long>::iterator it = sketch.find(key);
    if ( it == sketch.end() ) {   
        sketch[key] = 1;
        res = 1;
    } else {
        sketch[key] += 1;
        res = 0;
    }//fi 

    return res;
}




double fillSketches(SketchT *S, double *buffer, unsigned long len) {
    
    KEY_T key = 0;      

    Timer sketchTime;   
    
    startTimer(&sketchTime);
    for(unsigned long i = 0; i < len; ++i) {

        #ifdef POSI
            
            key = (KEY_T)std::ceil(std::log10(buffer[i])/S->base);
            S->posibins += addKeyToSketch(S->posiSketch, key);
            S->posipop += 1;
            
        #else
        
           
            
            if (buffer[i] > S->NULLBOUND) {
            
                key = (KEY_T)std::ceil(std::log10(buffer[i])/S->base);
                S->posibins += addKeyToSketch(S->posiSketch, key);
                S->posipop += 1;

            }//fi (1)
            else if ( -S->NULLBOUND <= buffer[i] && buffer[i] <= S->NULLBOUND) {  
                
                key = -MIN_KEY; 
                S->posibins += addKeyToSketch(S->posiSketch, key);
                S->posipop += 1;
            
            } else { 
                
                
                key = (KEY_T)std::ceil(std::log10(-buffer[i])/S->base);
                S->negabins += addKeyToSketch(S->negaSketch, key);
                S->negapop += 1;

            }//posi vs nega
        #endif 

        while ((S->posibins + S->negabins) > S->bound) {
            
            collapseUniformly(S->posiSketch);
            S->posibins = S->posiSketch.size();
            
            collapseUniformly(S->negaSketch);
            S->negabins = S->negaSketch.size();
            
           
            S->alpha = (2*S->alpha)/(1 + pow(S->alpha,2));
                
            S->gamma = (1+S->alpha)/(1-S->alpha);
            S->base = std::log10(S->gamma);

            S->collapses += 1;
            
        }//wend collapses

    }//for 
    stopTimer(&sketchTime);

    return getElapsedSeconds(&sketchTime);
}




double fillFullSketches(SketchF *S, double *buffer,  unsigned long len) {
    
    int key = 0;      
    Timer sketchTime;   
    startTimer(&sketchTime);

    for(unsigned long i = 0; i < len; ++i) {

        #ifdef POSI
            
            key = std::ceil(std::log10(buffer[i])/S->base);
            S->posibins += addKeyToFullSketch(S->posiSketch, key);
            S->posipop += 1;
        
        #else
                    
            if (buffer[i] > S->NULLBOUND) {
            
                key = std::ceil(std::log10(buffer[i])/S->base);
                S->posibins += addKeyToFullSketch(S->posiSketch, key);
                S->posipop += 1;

            }//fi (1)
            else if ( -S->NULLBOUND <= buffer[i] && buffer[i] <= S->NULLBOUND) {  
                
                key = -MIN_KEY; 
                S->posibins += addKeyToFullSketch(S->posiSketch, key);
                S->posipop += 1;
            
            } else { 
                
                key = std::ceil(std::log10(-buffer[i])/S->base);
                S->negabins += addKeyToFullSketch(S->negaSketch, key);
                S->negapop += 1;

            }//posi vs nega
        #endif 

        while ((S->posibins + S->negabins) > S->bound) {
            
            collapseUniformlyFull(S->posiSketch);
            S->posibins = S->posiSketch.size();
            
            collapseUniformlyFull(S->negaSketch);
            S->negabins = S->negaSketch.size();
            
            S->alpha = (2*S->alpha)/(1 + pow(S->alpha,2));
                
            S->gamma = (1+S->alpha)/(1-S->alpha);
            S->base = std::log10(S->gamma);

            S->collapses += 1;
            
        }//wend collapses

    }//for 
    stopTimer(&sketchTime);

    return getElapsedSeconds(&sketchTime);
}


void averagingBins(SketchT *S) {

    std::map<KEY_T, COUNT_T>::iterator it;
    COUNT_T pop = 0.0;

    it = S->posiSketch.begin();
    while(it != S->posiSketch.end()){
        it->second = it->second/2.0;
        pop += it->second;
        ++it;
    }//wend posi
    S->posipop = pop;

    pop = 0;
    it = S->negaSketch.begin();
    while(it != S->negaSketch.end()){
        it->second = it->second/2.0;
        pop += it->second;
        ++it;
    }//wend nega
    S->negapop = pop;
}



void updateRemotePeerState(SketchT *L, SketchT *R) {
    
    R->posiSketch = L->posiSketch; 
    R->posibins = L->posibins;
    R->posipop = L->posipop;

    R->negaSketch = L->negaSketch;
    R->negabins = L->negabins;
    R->negapop = L->negapop;

    R->collapses = L->collapses;
    R->alpha = L->alpha;
    R->gamma = L->gamma;
    R->base = L->base;
}





int mergeSketches(int localId, SketchT *L, int remoteId, SketchT *R, double *elapsedSeconds, bool remotePeerConverged){
    
    if (!L || !R) {
        std::cout<< "Merging sketches: no correct inputs to function"<< std::endl;
        return ESKETCH;
    }

    
    #ifdef DEBUG
        std::cout<<"\n\tMERGE: Local peer " << localId << ", Remote peer " << remoteId << std::endl;
        
        std::cout << "Local sketch has "<<L->posiSketch.size() << " + " << L->negaSketch.size() << " bins and pop = "<< L->posipop+L->negapop<<std::endl;

        debugSketch(L->posiSketch, stdout);
        debugSketch(L->negaSketch, stdout);

        std::cout << "Remote sketch has "<<R->posiSketch.size() << " + " << R->negaSketch.size() << " bins and pop = "<< R->posipop+R->negapop<<std::endl;
        
        debugSketch(R->posiSketch, stdout);
        debugSketch(R->negaSketch, stdout);
    #endif

    int res = 0;
    std::map<KEY_T, COUNT_T>::iterator it;
    Timer t;
    
    int collapsesA=0, collapsesB =0;
    
    startTimer(&t);
    if (L->collapses != R->collapses) {

            while (L->collapses < R->collapses) {
                
                collapseUniformly(L->posiSketch);
                L->posibins = L->posiSketch.size();
            
                collapseUniformly(L->negaSketch);
                L->negabins = L->negaSketch.size();
                
                L->alpha = (2*L->alpha)/(1 + pow(L->alpha,2));
                    
                L->gamma = (1+L->alpha)/(1-L->alpha);
                L->base = std::log10(L->gamma);

                L->collapses += 1;
                ++collapsesA;
            }//wend

            while(R->collapses < L->collapses){
                collapseUniformly(R->posiSketch);
                R->posibins = R->posiSketch.size();
            
                collapseUniformly(R->negaSketch);
                R->negabins = R->negaSketch.size();
                
                R->alpha = (2*R->alpha)/(1 + pow(R->alpha,2));
                    
                R->gamma = (1+R->alpha)/(1-R->alpha);
                R->base = std::log10(R->gamma);

                R->collapses += 1;
                ++collapsesA;
            }//wend

    assert(L->collapses == R->collapses);
    }//fi 

    #ifdef DEBUG
        if (collapsesA){
            std::cout<<"Additional collapses before merge: "<< collapsesA << std::endl;
            std::cout<<"L->collapses == R->collapses="<<R->collapses<<std::endl;
            std::cout<<"L->alpha == R->alpha="<<L->alpha<<std::endl;
        }
    #endif

    it = R->posiSketch.begin();
    while(it!= R->posiSketch.end()){
        L->posiSketch[it->first] += it->second;
        ++it;
    }//wend posi
    L->posibins = L->posiSketch.size();
    L->posipop += R->posipop;
    
    
    it = R->negaSketch.begin();
    while(it!= R->negaSketch.end()){
        L->negaSketch[it->first] += it->second;
        ++it;
    }//wend nega
    L->negabins = L->negaSketch.size();
    L->negapop += R->negapop;
    
    #ifdef DEBUG
        std::cout << "Local posi sketch now has "<<L->posibins<< " bins and  " << L->posipop << " Ppop"<<std::endl;
        std::cout << "Local nega sketch now has "<<L->negabins<< " bins and  " << L->negapop << " Ppop"<<std::endl;
    #endif

    
    while ((L->posibins + L->negabins) > L->bound) {

        collapseUniformly(L->posiSketch);
        L->posibins = L->posiSketch.size();
        
        collapseUniformly(L->negaSketch);
        L->negabins = L->negaSketch.size();
        
        L->alpha = (2*L->alpha)/(1 + pow(L->alpha,2));
            
        L->gamma = (1+L->alpha)/(1-L->alpha);
        L->base = std::log10(L->gamma);

        L->collapses +=1;
        ++collapsesB;
    }//wend collapses
    

    averagingBins(L);
    stopTimer(&t);

    updateRemotePeerState(L,R);
    
    if (elapsedSeconds){
        *elapsedSeconds = getElapsedSeconds(&t);
    }

return res;
}


void addItemToSketch(SketchT *S, double item) {
    
    KEY_T key = 0;      

    #ifdef POSI
        
        
        key = (KEY_T)std::ceil(std::log10(item)/S->base);
        S->posibins += addKeyToSketch(S->posiSketch, key);
        S->posipop += 1;
        
    #else
    
      
        
        if (item > S->NULLBOUND) {
        
            key = (KEY_T)std::ceil(std::log10(item)/S->base);
            S->posibins += addKeyToSketch(S->posiSketch, key);
            S->posipop += 1;

        }//fi (1)
        else if ( -S->NULLBOUND <= item && item <= S->NULLBOUND) {  
            
            key = -MIN_KEY; 
            S->posibins += addKeyToSketch(S->posiSketch, key);
            S->posipop += 1;
        
        } else { 
            
            key = (KEY_T)std::ceil(std::log10(-item)/S->base);
            S->negabins += addKeyToSketch(S->negaSketch, key);
            S->negapop += 1;

        }//posi vs nega
    #endif 

    while ((S->posibins + S->negabins) > S->bound) {
        
        collapseUniformly(S->posiSketch);
        S->posibins = S->posiSketch.size();
        
        collapseUniformly(S->negaSketch);
        S->negabins = S->negaSketch.size();
        
        S->alpha = (2*S->alpha)/(1 + pow(S->alpha,2));
            
        S->gamma = (1+S->alpha)/(1-S->alpha);
        S->base = std::log10(S->gamma);

        S->collapses += 1;
        
    }//wend collapses
}


void addItemToFullSketch(SketchF *S, double item) {
    
    int key = 0;      

    #ifdef POSI
        
        key = std::ceil(std::log10(item)/S->base);
        S->posibins += addKeyToFullSketch(S->posiSketch, key);
        S->posipop += 1;
    
    #else
                
        if (item > S->NULLBOUND) {
        
            key = std::ceil(std::log10(item)/S->base);
            S->posibins += addKeyToFullSketch(S->posiSketch, key);
            S->posipop += 1;

        }//fi (1)
        else if ( -S->NULLBOUND <= item && item <= S->NULLBOUND) {  
            
            key = -MIN_KEY; 
            S->posibins += addKeyToFullSketch(S->posiSketch, key);
            S->posipop += 1;
        
        } else { 
            
            key = std::ceil(std::log10(-item)/S->base);
            S->negabins += addKeyToFullSketch(S->negaSketch, key);
            S->negapop += 1;

        }//posi vs nega
    #endif 

        while ((S->posibins + S->negabins) > S->bound) {
            
            collapseUniformlyFull(S->posiSketch);
            S->posibins = S->posiSketch.size();
            
            collapseUniformlyFull(S->negaSketch);
            S->negabins = S->negaSketch.size();
            
            S->alpha = (2*S->alpha)/(1 + pow(S->alpha,2));
                
            S->gamma = (1+S->alpha)/(1-S->alpha);
            S->base = std::log10(S->gamma);

            S->collapses += 1;
            
        }//wend collapses
}

