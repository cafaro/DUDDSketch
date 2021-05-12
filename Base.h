/********************************************************/
/* Distributed UDDSketch                                */
/*                                                      */
/* Coded by Catiuscia Melle                             */
/*                                                      */
/* April 8, 2021                                        */
/*                                                      */
/*                                                      */
/********************************************************/


#ifndef __COMMON_BASE_H__
#define __COMMON_BASE_H__


//************************************* Includes

#include <iostream>             
#include <iomanip>              
#include <cmath>   
#include <chrono>               
#include <random>               

#include <utility>
#include <set>
#include <algorithm>
#include <unordered_set>

#include <unistd.h>             
#include <string.h>             
#include <sys/time.h>           

#include <errno.h>


//************************************* Constants

const int MIN_PARAMS = 9;       
const int FLEN = 256;           
const int NLEN = 64;            
const int EPARAMS = 3;          

enum CHURN_MODEL{NOCHURN=0, FAILSTOP=1, YAO=2, YAOEXP=3};




//************************************* Types definition

typedef int KEY_T;              
typedef double COUNT_T;         




typedef struct Timer {
    struct timeval start;
    struct timeval end;
} Timer;




//************************************* Utility functions

void usage(char *msg);


//************************************* File logs

unsigned long composeFileName(char *fname, int flen, char *gname, int peers, int rounds, char *dname, int fo, int churntype, double failp);

FILE *openLogFile(char prefix[], int prefixlen);



//************************************* Time evaluation functions

/** @brief Initializes the _start_ component of a Timer */
void startTimer(Timer *t);

/** @brief Initializes the _end_ component of a Timer   */
void stopTimer(Timer *t);

/** @brief Computes elapsed time in milliseconds        */
double getElapsedMilliSecs(Timer *t);

/** @brief Computes elapsed time in seconds             */
double getElapsedSeconds(Timer *t);


#endif // __COMMON_BASE_H__
