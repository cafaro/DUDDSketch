/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/

#ifndef __SKETCH_H__
#define __SKETCH_H__


#include "InputSet.h"     

#include <map>
#include <algorithm>
#include <iterator>
#include <vector>
#include <utility>


const int ESKETCH = 33;                     


typedef struct SketchT {
    
    std::map<KEY_T, COUNT_T> posiSketch;    
    COUNT_T posipop;                        
    int posibins;                           
    
    std::map<KEY_T, COUNT_T> negaSketch;    
    COUNT_T negapop;                        
    int negabins;                           
    
    int collapses;                          
    int bound;                              

    double alpha;                            
    double gamma;                           
    double base;                            
    double NULLBOUND;                       

} SketchT;




typedef struct SketchF {
    
    std::map<int, unsigned long> posiSketch;    
    unsigned long posipop;                       
    int posibins;                           
    
    std::map<int, unsigned long> negaSketch;    
    unsigned long negapop;                        
    int negabins;                           

    int collapses;                          
    int bound;                              

    double alpha;                          
    double gamma;                           
    double base;                           
    double NULLBOUND;                       
    
} SketchF;



void initSketchType(SketchT *s, int bound, double alpha, double gamma, double NULLBOUND);



void initFullSketch(SketchF *s, int bound, double alpha, double gamma, double NULLBOUND);

void debugSketchInstance(SketchT *s);


void debugSketch(std::map<KEY_T, COUNT_T>& mySketch, FILE *log);


void collapseUniformly(std::map<KEY_T, COUNT_T>& mySketch);

void collapseUniformlyFull(std::map<int, unsigned long>& mySketch);

int addKeyToSketch(std::map<KEY_T, COUNT_T>&sketch, KEY_T key);


int addKeyToFullSketch(std::map<int, unsigned long>&sketch, int key);

double fillSketches(SketchT *S, double *buffer, unsigned long len);


double fillFullSketches(SketchF *S, double *buffer,  unsigned long len);



void averagingBins(SketchT *S);

void updateRemotePeerState(SketchT *L, SketchT *R);


int mergeSketches(int localId, SketchT *L, int remoteId, SketchT *R, double *elapsedSeconds, bool remotePeerConverged);



void addItemToSketch(SketchT *S, double item);

void addItemToFullSketch(SketchF *S, double item);


#endif // __SKETCH_H__

