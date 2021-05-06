/*
 * -----------------------------------------------------------------
 * Programmer(s): F. Durastante @ IAC-CNR
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2019, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------
 * This is the testing routine to check the SUNLinSol PSBLAS module
 * implementation.
 * -----------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>

#include <sundials/sundials_types.h>
#include <sundials/sundials_iterative.h>
#include <sunmatrix/sunmatrix_psblas.h>
#include <nvector/nvector_psblas.h>
#include <sundials/sundials_math.h>
#include <sunlinsol/sunlinsol_psblas.h>

#include "test_sunlinsol.h"

#include "mpi.h"

#define NBMAX       20

/* prototypes for PSBLAS matrix generation */
double  a1(double x, double y, double  z);
double  a2(double x, double y, double  z);
double  a3(double x, double y, double  z);
double  b1(double x, double y, double  z);
double  b2(double x, double y, double  z);
double  b3(double x, double y, double  z);
double 	g(double x, double y, double z);
psb_i_t matgen(psb_c_ctxt cctxt, psb_i_t nl, psb_i_t idim, psb_l_t vl[],SUNMatrix A);
/*-------------------------------------------------
 * Routine to read input from file
 *------------------------------------------------*/

 #define LINEBUFSIZE 1024

 static char buffer[LINEBUFSIZE+1];

int get_buffer(FILE *fp)
{
  char *info;
  while(!feof(fp)) {
    info = fgets(buffer,LINEBUFSIZE,fp);
    if (buffer[0]!='%') break;
  }
}

void get_iparm(FILE *fp, int *val)
{
  get_buffer(fp);
  sscanf(buffer,"%d ",val);
}

void get_dparm(FILE *fp, double *val)
{
  get_buffer(fp);
  sscanf(buffer,"%lf ",val);
}

void get_hparm(FILE *fp, char *val)
{
  get_buffer(fp);
  sscanf(buffer,"%s ",val);
}

void sync_device();

int main(int argc, char *argv[])
{
  int             fails=0;                 /* counter for test failures */
  int             passfail=0;              /* overall pass/fail flag    */
  SUNLinearSolver LS;                      /* linear solver object      */
  SUNMatrix       A;                       /* left-hand side            */
  N_Vector        x, xhat, b;              /* test vectors              */
  double          tol;                     /* tolerance for solution    */
  psb_c_SolverOptions options;             /* Solver options            */
  psb_c_ctxt  *cctxt;                      /* PSBLAS Context            */
  psb_i_t      ictxt;                      /* PSBLAS Integer Context    */
  psb_i_t      nprocs, myid;               /* Number of procs, proc id  */
  psb_c_descriptor *cdh;                   /* PSBLAS Descriptor         */
  /* Auxiliary variabales */
  psb_i_t      info;              				 /* FLAG value for PSBLAS     */
  char        *fout;
  MPI_Comm     comm;											 /* MPI Comminicator          */
  psb_i_t nb,nlr,nl;									     /* Block Data distribution   */
  psb_l_t i,ng, *vl, k;
  /* Input parameters */
  char methd[20],ptype[20];                /* Solve method and type     */
  char afmt[8];
  psb_i_t nparms;
  psb_i_t idim,istop,itmax,itrace,irst;

  /* Get processor number and total number of processes */
  cctxt = psb_c_new_ctxt();
  psb_c_init(cctxt);
  psb_c_info(*cctxt,&myid,&nprocs);
  psb_c_get_i_ctxt(*(cctxt),&ictxt,&info);
  comm = MPI_Comm_f2c(ictxt);

  /* Read input and decide test-problem */
  psb_c_barrier(*cctxt);
  if (myid == 0) {
    fout = fgets(buffer,LINEBUFSIZE,stdin);
    sscanf(buffer,"%d ",&nparms);
    fout = fgets(buffer,LINEBUFSIZE,stdin);
    sscanf(buffer,"%s",methd);
    fout = fgets(buffer,LINEBUFSIZE,stdin);
    sscanf(buffer,"%s",ptype);
    fout = fgets(buffer,LINEBUFSIZE,stdin);
    sscanf(buffer,"%s",afmt);
    fout = fgets(buffer,LINEBUFSIZE,stdin);
    sscanf(buffer,"%d",&idim);
    fout = fgets(buffer,LINEBUFSIZE,stdin);
    sscanf(buffer,"%d",&istop);
    fout = fgets(buffer,LINEBUFSIZE,stdin);
    sscanf(buffer,"%d",&itmax);
    fout = fgets(buffer,LINEBUFSIZE,stdin);
    sscanf(buffer,"%d",&itrace);
    fout = fgets(buffer,LINEBUFSIZE,stdin);
    sscanf(buffer,"%d",&irst);
    fout = fgets(buffer,LINEBUFSIZE,stdin);
    sscanf(buffer,"%le",&tol);
  }
  /* Now broadcast the values, and check they're OK */
  psb_c_ibcast(*cctxt,1,&nparms,0);
  psb_c_hbcast(*cctxt,methd,0);
  psb_c_hbcast(*cctxt,ptype,0);
  psb_c_hbcast(*cctxt,afmt,0);
  psb_c_ibcast(*cctxt,1,&idim,0);
  psb_c_ibcast(*cctxt,1,&istop,0);
  psb_c_ibcast(*cctxt,1,&itmax,0);
  psb_c_ibcast(*cctxt,1,&itrace,0);
  psb_c_ibcast(*cctxt,1,&irst,0);
  psb_c_dbcast(*cctxt,1,&tol,0);
  psb_c_barrier(*cctxt);
  if(myid == 0){
    printf("**************************************************************\n");
    printf("Poisson test\n");
    printf("**************************************************************\n");
    printf(" Read %d parameters.\n",nparms);
    printf(" Solving with method %s preconditioned by %s\n",methd,ptype);
    printf(" A is stored in format %s of size %d^3 x %d^3\n",afmt,idim,idim);
    printf(" Stopping criterion is %d over %d maxit (trace %d)\n",istop,itmax,itrace);
    printf(" irst = %d, tolerance = %e\n",irst,tol);
    printf("**************************************************************\n");
  }
  psb_c_barrier(*cctxt);

  /* Construction of the matrix         */
  cdh = psb_c_new_descriptor();
  psb_c_set_index_base(0);
  /* Simple minded BLOCK data distribution */
  ng = ((psb_l_t) idim)*idim*idim;
  nb = (ng+nprocs-1)/nprocs;
  nl = nb;
  if ( (ng -myid*nb) < nl) nl = ng -myid*nb;
  fprintf(stdout,"%d: Input data %d %ld %d %d\n",myid,idim,ng,nb, nl);
  vl = malloc(nb*sizeof(psb_l_t));
  if (vl == NULL) {
    fprintf(stderr,"On %d: malloc failure\n",myid);
    psb_c_abort(*cctxt);
  }
  i = ((psb_l_t)myid) * nb;
  for (k=0; k<nl; k++)
    vl[k] = i+k;

  info=psb_c_cdall_vl(nl,vl,*cctxt,cdh);
  if (info != 0) {
    fprintf(stderr,"From cdall: %d\nBailing out\n",info);
    psb_c_abort(*cctxt);
  }

  if(myid == 0) printf("Descriptor for square problem allocated.\n");

  /* Allocate the space for matrices and vectors */
  A = SUNPSBLASMatrix(cctxt, cdh);
  if (A == NULL) {
    if (myid == 0) printf("FAIL: Unable to create a new matrix \n\n");
      psb_c_abort(*cctxt);
      return(1);
  }
  x = N_VNew_PSBLAS(cctxt, cdh);
  if (x == NULL) {
    SUNMatDestroy_PSBLAS(A);
    if (myid == 0) printf("FAIL: Unable to create a new vector \n\n");
      psb_c_abort(*cctxt);
      return(1);
  }
  b = N_VNew_PSBLAS(cctxt, cdh);
  if (b == NULL) {
    SUNMatDestroy(A);
    N_VDestroy(x);
    if (myid == 0) printf("FAIL: Unable to create a new vector \n\n");
      psb_c_abort(*cctxt);
      return(1);
  }
  xhat = N_VNew_PSBLAS(cctxt,cdh);

  /* Populate the A matrix */
  if (matgen(*cctxt, nl, idim, vl,A)!= 0) {
    fprintf(stderr,"Error during matrix build loop for A\n");
    psb_c_abort(*cctxt);
    return(1);
  }

  /* Assemble the descriptor */
  info=psb_c_cdasb(cdh);
  if (info!=0){
    if (myid == 0) printf("FAIL: Unable to assemble the descriptor \n\n");
      psb_c_abort(*cctxt);
    return(info);
  }

  N_VConst_PSBLAS(1.0,xhat);
  SUNMatAsb_PSBLAS(A);
  N_VAsb_PSBLAS(b);
  N_VAsb_PSBLAS(x);

  /* Set up the solver options */
  psb_c_DefaultSolverOptions(&options);
  options.eps    = tol;
  options.itmax  = itmax;
  options.irst   = irst;
  options.itrace = itrace;
  options.istop  = istop;
  /* Create PSBLAS/MLD2P4 linear solver */
  LS = SUNLinSol_PSBLAS(options, methd, ptype, cctxt);
  SUNLinSolInitialize_PSBLAS(LS);
  SUNLinSolSetc_PSBLAS(LS,"SMOOTHER_TYPE","BJAC");
  SUNLinSolSetc_PSBLAS(LS,"COARSE_SOLVE","BJAC");
  SUNLinSolSeti_PSBLAS(LS,"COARSE_SWEEPS",8);

  /* On the call of the test the vector x should contain the solution */
  SUNMatMatvec(A,xhat,b);


  /* Test Routines */
  fails += Test_SUNLinSolGetType(LS, SUNLINEARSOLVER_MATRIX_ITERATIVE,myid);
  fails += Test_SUNLinSolSetup(LS, A, myid);
  fails += Test_SUNLinSolSolve(LS, A, xhat, b, tol,myid);
  fails += Test_SUNLinSolLastFlag(LS, myid);
  fails += Test_SUNLinSolNumIters(LS, myid);
  fails += Test_SUNLinSolResNorm(LS, myid);

  /* Print result */
  if (fails) {
    printf("FAIL: SUNLinSol_PSBLAS module, failed %i tests\n\n", fails);
    passfail += 1;
  } else if (myid == 0) {
    printf("SUCCESS: SUNLinSol_PSBLAS module, passed all tests\n\n");
  }

  /* check if any other process failed */
  (void) MPI_Allreduce(&passfail, &fails, 1, MPI_INT, MPI_MAX, comm);

  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  N_VDestroy(xhat);
  N_VDestroy(x);
  N_VDestroy(b);

  /* Free solver and vectors */
  if ((info=psb_c_cdfree(cdh))!=0) {
    fprintf(stderr,"From cdfree: %d\nBailing out\n",info);
    psb_c_abort(*cctxt);
  }

  free(cdh);

  if (myid == 0) fprintf(stderr,"program completed successfully, fails is %d\n",passfail);

  psb_c_barrier(*cctxt);
  psb_c_exit(*cctxt);

  return(passfail);
}

int check_vector(N_Vector X, N_Vector Y, realtype tol)
{
  int failure = 0;
  realtype *xdata, *ydata;
  sunindextype xldata, yldata;
  sunindextype i;
  N_Vector Z;

  Z = N_VClone(X);

  N_VLinearSum(-1.0,X,1.0,Y,Z);

  if (psb_c_dgenrm2(NV_PVEC_P(Z),NV_DESCRIPTOR_P(Z))/psb_c_dgenrm2(NV_PVEC_P(X),NV_DESCRIPTOR_P(X)) > tol){
    return(1);
  }else{
    return(0);
  }
}

/*-------------------------------------------------
 * PSBLAS Auxiliary functions for matrix creation
 *------------------------------------------------*/
 double  a1(double x, double y, double  z)
 {
   return(1.0/80.0);
 }
 double a2(double x, double y, double  z)
 {
   return(1.0/80.0);
 }
 double a3(double x, double y, double  z)
 {
   return(1.0/80.0);
 }
 double  c(double x, double y, double  z)
 {
   return(0.0);
 }
 double  b1(double x, double y, double  z)
 {
   return(0.0/sqrt(3.0));
 }
 double b2(double x, double y, double  z)
 {
   return(0.0/sqrt(3.0));
 }
 double b3(double x, double y, double  z)
 {
   return(0.0/sqrt(3.0));
 }

 double g(double x, double y, double z)
 {
   if (x == 1.0) {
     return(1.0);
   } else if (x == 0.0) {
     return( exp(-y*y-z*z));
   } else {
     return(0.0);
   }
 }

 psb_i_t matgen(psb_c_ctxt cctxt, psb_i_t nl, psb_i_t idim, psb_l_t vl[],SUNMatrix A)
 {
   psb_i_t iam, np;
   psb_l_t ix, iy, iz, el,glob_row;
   psb_i_t i, k, info;
   double x, y, z, deltah, sqdeltah, deltah2;
   double val[10*NBMAX], zt[NBMAX];
   psb_l_t irow[10*NBMAX], icol[10*NBMAX];

   info = 0;
   psb_c_info(cctxt,&iam,&np);
   deltah = (double) 1.0/(idim+1);
   sqdeltah = deltah*deltah;
   deltah2  = 2.0* deltah;
   psb_c_set_index_base(0);
   for (i=0; i<nl;  i++) {
     glob_row=vl[i];
     el=0;
     ix = glob_row/(idim*idim);
     iy = (glob_row-ix*idim*idim)/idim;
     iz = glob_row-ix*idim*idim-iy*idim;
     x=(ix+1)*deltah;
     y=(iy+1)*deltah;
     z=(iz+1)*deltah;
     zt[0] = 0.0;
     /*  internal point: build discretization */
     /*  term depending on   (x-1,y,z)        */
     val[el] = -a1(x,y,z)/sqdeltah-b1(x,y,z)/deltah2;
     if (ix==0) {
       zt[0] += g(0.0,y,z)*(-val[el]);
     } else {
       icol[el]=(ix-1)*idim*idim+(iy)*idim+(iz);
       el=el+1;
     }
     /*  term depending on     (x,y-1,z) */
     val[el]  = -a2(x,y,z)/sqdeltah-b2(x,y,z)/deltah2;
     if (iy==0) {
       zt[0] += g(x,0.0,z)*(-val[el]);
     } else {
       icol[el]=(ix)*idim*idim+(iy-1)*idim+(iz);
       el=el+1;
     }
     /* term depending on     (x,y,z-1)*/
     val[el]=-a3(x,y,z)/sqdeltah-b3(x,y,z)/deltah2;
     if (iz==0) {
       zt[0] += g(x,y,0.0)*(-val[el]);
     } else {
       icol[el]=(ix)*idim*idim+(iy)*idim+(iz-1);
       el=el+1;
     }
     /* term depending on     (x,y,z)*/
     val[el]=2.0*(a1(x,y,z)+a2(x,y,z)+a3(x,y,z))/sqdeltah + c(x,y,z);
     icol[el]=(ix)*idim*idim+(iy)*idim+(iz);
     el=el+1;
     /*  term depending on     (x,y,z+1) */
     val[el] = -a3(x,y,z)/sqdeltah+b3(x,y,z)/deltah2;
     if (iz==idim-1) {
       zt[0] += g(x,y,1.0)*(-val[el]);
     } else {
       icol[el]=(ix)*idim*idim+(iy)*idim+(iz+1);
       el=el+1;
     }
     /* term depending on     (x,y+1,z) */
     val[el] = -a2(x,y,z)/sqdeltah+b2(x,y,z)/deltah2;
     if (iy==idim-1) {
       zt[0] += g(x,1.0,z)*(-val[el]);
     } else {
       icol[el]=(ix)*idim*idim+(iy+1)*idim+(iz);
       el=el+1;
     }
     /*  term depending on     (x+1,y,z) */
     val[el] = -a1(x,y,z)/sqdeltah+b1(x,y,z)/deltah2;
     if (ix==idim-1) {
       zt[0] += g(1.0,y,z)*(-val[el]);
     } else {
       icol[el]=(ix+1)*idim*idim+(iy)*idim+(iz);
       el=el+1;
     }
     for (k=0; k<el; k++) irow[k]=glob_row;
     if ((info=psb_c_dspins(el,irow,icol,val,SM_PMAT_P(A),SM_DESCRIPTOR_P(A)))!=0)
       fprintf(stderr,"From psb_c_dspins: %d\n",info);
   }

   return(info);


 }

 void sync_device()
 {
 }
