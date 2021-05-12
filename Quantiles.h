/************************************************************/
/* Distributed UDDSketch                                    */
/*                                                          */
/* Coded by Catiuscia Melle                                 */
/*                                                          */
/* April 8, 2021                                            */
/*                                                          */
/*                                                          */
/* (C code for the quickselect algorithm, taken from        */
/*  Numerical Recipes in C):                                */
/* http://www.stat.cmu.edu/~ryantibs/median/quickselect.c   */
/*                                                          */
/************************************************************/


#ifndef __QUANTILES_H__
#define __QUANTILES_H__

#include "Sketch.h"

#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;

double quickselect(double *data, int len, int pos);


double getQuantileFor(double q, SketchT *S, KEY_T *qIndex, COUNT_T *qCount, int *rank);

double querySketchForQuantile(double q, SketchF *S, int *qIndex, unsigned long *qCount, unsigned long *rank);

void getFullQuantiles(double *Qs, int Ns, SketchF *S, FILE *fp);

void logAvgQuantiles(double *Qs, int Ns, SketchT *S, FILE *fp);

void logHeader(FILE *fp, double *Qs, int Ns);

#endif // __QUANTILES_H__

