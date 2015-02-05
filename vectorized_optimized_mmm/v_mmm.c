#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "execute.h"
#include <emmintrin.h>

#define PROCESSOR 2.67/1E9
#define BLOCK 16
#define N PROBLEM_SIZE
double A[N][N] __attribute__ ((aligned (16)));
double B[N][N] __attribute__ ((aligned (16)));
double C[N][N] __attribute__ ((aligned (16)));
double F[N][N] __attribute__ ((aligned (16)));

int min(int a, int b){
 if (a<b) return a;
 return b;
}
/*vectorized & cache optimized implementation*/
void v_MMM11(int x0,int y0,int x1,int y1){
if (y0!=x1) return;//simple error check
int i,j,k,ii,jj,kk,tmp_jj,tmp_kk;
jj=0;kk=0;
register double x,y,z;
__m128d reg_a,reg_b,reg_c;
int SQUARE=min(x0>>4,y1>>4)*BLOCK;

//do the largest square matrix first
for(ii=0; ii<SQUARE; ii+=BLOCK){//i dimension window
 for(i=0; i<BLOCK; i++){
  for(kk=0; kk<SQUARE; kk+=BLOCK){//k dimension window
    for(k=0; k<BLOCK; k++){
      reg_a=_mm_load_sd(&A[i+ii][k+kk]);
      reg_a=_mm_unpacklo_pd(reg_a,reg_a);
      for(jj=0; jj<SQUARE;jj+=BLOCK){//j dimension window
       for(j=0; j<BLOCK; j+=2){
         reg_b=_mm_load_pd(&B[k+kk][j+jj]);
         reg_c=_mm_load_pd(&C[i+ii][j+jj]);
         _mm_store_pd(&C[i+ii][j+jj],
           _mm_add_pd(reg_c,_mm_mul_pd(reg_a,reg_b)));;
       }//improve further by unrolling!? no time..
      }
     } 
  }
}
}
//save the window variables needed for later
tmp_jj=jj;
tmp_kk=kk;
//clean up the bottom portion of the matrix
int b_i;
for (b_i=0;b_i+ii<x0; b_i++){
 for(kk=0;kk<SQUARE;kk+=BLOCK){
  for(k=0; k<BLOCK; k++){
  y=A[b_i+ii][k+kk];
  for(jj=0; jj<SQUARE; jj+=BLOCK){
    for(j=0; j<BLOCK; j++){
     C[b_i+ii][j+jj]+=y*B[k+kk][j+jj];
    }
  }
  }
 }
}
int b_j;
//clean up the right strip of the matrix
if(y1%BLOCK!=0){
for (i=0; i<x0; i++){
 for(k=0;k<x1;k++){
  z=A[i][k];
  if (k>=tmp_kk){
  for(j=0;j<y1;j++){
    C[i][j]+=z*B[k][j];
  }
  }else{
  for(b_j=0;b_j+tmp_jj<y1;b_j++){
    C[i][b_j+tmp_jj]+=z*B[k][b_j+tmp_jj];
  }
  }
 }
}
}

}

//this one is intentionally wrong, using for sanity check in check()
void MMM0(){
    
int i,j,k;
for(i=0; i<N; i++){
  for(j=0; j<N; j++){
    for(k=0; k<N; k++){
      C[j][i]+=A[i][k]*B[k][j];//wrong!
    }
  } 
}
}

/* simple, naive, but correct ijk matrix multiply implementation */
void MMM1(){
    
int i,j,k;
for(i=0; i<N; i++){
  for(j=0; j<N; j++){
    for(k=0; k<N; k++){
      C[i][j]+=A[i][k]*B[k][j];
    }
  } 
}
}


/* initialize arrays with positive random values */
void initialize(){

/* seed for intializing the intial random values in array */
srand(time(NULL));
int i,j;
for(i=0; i<N; i++){
  for(j=0; j<N; j++){
    A[i][j]=(rand()%(i+j+1))*1.0f;
    B[i][j]=(rand()%(i+j+1))*1.0f;
    C[i][j]=0.0f;
  }
 }
}

/* zero out FLUSH_SIZE bytes in memory*/
int flush(){
char* mem=(char*)malloc(FLUSH_SIZE);
  assert(mem!=NULL);
  int i;
  for(i=0; i<(FLUSH_SIZE); i++){
   *(mem+i)=0;
  }
return 0;
}
	

void check(){
int i,j;
/*do one implementation*/
initialize();
v_MMM11(N,N,N,N);
/*copy the result to F*/
for(i=0; i<N; i++){
 for(j=0; j<N; j++){
  F[i][j]=C[i][j];
 }
}
/*clear C*/
for(i=0; i<N; i++)
 for(j=0;j<N; j++)
  C[i][j]=0.0f;

/*do another, overwriting C*/
MMM1();

/*check equality*/
for(i=0; i<N; i++){
 for(j=0; j<N; j++){
  if (C[i][j]!=F[i][j]){
   printf("%s, C:%f != F:%f, at indices: (%d,%d)\n","not equal:",C[i][j],F[i][j],i,j);
   return;
  }
 }
}
}


int main(){
unsigned long long a,b;

printf("\nMMM12:");
/* initialize the arrays A and B*/
initialize();
/*flush the cache ~16Mb just to be safe*/
flush();
/* matrix multiply */
a = rdtsc();
v_MMM11(N,N,N,N);
b = rdtsc();
printf("\n\tcycles: %llu  elapsed sec : %f\n",(b-a), ((b-a)/PROCESSOR));
check();
 return 0;
}

