
/* ----------------------------------------------------------------------
 * Programmer(s): F. Durastante @ IAC-CNR
 * ----------------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2019, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * ----------------------------------------------------------------------
 * Example (richards):
 * ----------------------------------------------------------------------
 *  Run command line: mpirun -np N -machinefile machines GinzburgLandau < \
 * \ kinsol.inp
 *  where N is the number of processors, and kinsol.inp contains the general
 * setting for the example.
 *
 * ----------------------------------------------------------------------
 * The structure of the application is the following:
 * 1. Initialize parallel environment with psb_init
 * 2. Initialize index space with psb_cdall
 * 3. Loop over the topology of the discretization mesh and build the descriptor
 *    with psb_cdins
 * 4. Assemble the descriptor with psb_cdasb
 * 5. Allocate the sparse matrices and dense vectors with psb_spall and psb_geall
 * 6. Loop over the time steps using the KINSOL routines to solve the set of
 *    nonlinear equation at each step.
 * -----------------------------------------------------------------------
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 #include <psb_util_cbind.h>

 #include <kinsol/kinsol.h>             /* access to KINSOL func., consts.    */
 #include <nvector/nvector_psblas.h>    /* access to PSBLAS N_Vector          */
 #include <sunmatrix/sunmatrix_psblas.h>/* access to PSBLAS SUNMATRIX 			  */
 #include <sunlinsol/sunlinsol_psblas.h>/* access to PSBLAS SUNLinearSolver   */
 #include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype    */
 #include <sundials/sundials_math.h>    /* access to SUNMAX, SUNRabs, SUNRsqrt*/
 #include <sundials/sundials_iterative.h>

 #include <mpi.h>

 /* ------------------------------------------------
  * Auxiliary functions for KINSOL
 -------------------------------------------------*/
 // These functions are declared static to avoid conflicts with things that
 // could be somewhere else in the KINSOL library. It works also without the
 // static declaration, but you never know.
 static int funcprpr(N_Vector u, N_Vector fval, void *user_data);
 static int jac(N_Vector y, N_Vector f, SUNMatrix J,
                void *user_data, N_Vector tmp1, N_Vector tmp2);
 static int check_flag(void *flagvalue, const char *funcname, int opt, int id);
 static void PrintFinalStats(void *kmem, int i, void *user_data);
 static void UpdatePerformanceStats(void *kmem, int i, psb_d_t timeofthestep,
      void *user_data, void *perf_info);

 /*--------------------------------------------------
  * Coefficient functions for J and F
  *-------------------------------------------------*/
  static double Sfun(double p, double alpha, double beta, double thetas,
                      double thetar);
  static double Kfun(double p, double a, double gamma, double Ks);
  static double Sfunprime(double p, double alpha, double beta, double thetas,
                      double thetar);
  static double Kfunprime(double p, double a, double gamma, double Ks);
  static double sgn(double x);
  static double source(double x, double y, double z, double t);
  static double boundary(double x, double y, double z, double t, void *user_data);
  static double upstream(double pL, double pU, void *user_data);
  static double upstream10(double pL, double pU, void *user_data);
  static double upstream01(double pL, double pU, void *user_data);
  static double chi(double pL, double pU);

  bool BUILD_AUXILIARY_MATRIX = true;

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
 void get_dparm2(FILE *fp, double *val, int size)
 {
   get_buffer(fp);
   val = (double *) malloc( sizeof(double)*size );
   for(int i=0; i<size; i++)
    sscanf(buffer,"%lf ", &val[i]);
 }
 void get_hparm(FILE *fp, char *val)
 {
   get_buffer(fp);
   sscanf(buffer,"%s ",val);
 }
 /*-----------------------------------------------
  * Auxiliary data structures
  *-----------------------------------------------*/
  struct user_data_for_f {
    psb_l_t *vl;          // Local portion of the global indexs
    psb_i_t idim;         // Number of dofs in direction x
    psb_i_t jdim;         // Number of dofs in direction y
    psb_i_t kdim;         // Number of dofs in direction z
    psb_i_t nl;           // Number of blocks in the distribution
    psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks, rho, phi, pr; // Problem Parameters
    psb_d_t xmax,ymax,L;  // Size of the box
    psb_d_t dt;
    N_Vector oldpressure; // Old Pressure value for Euler Time-Stepping
    SUNLinearSolver *LS;  // Pointer to Linear Solver Object
    SUNMatrix B;          // Matrix on which the preconditioner is built
    psb_i_t timestep;     // Actual time-step
    psb_i_t functioncount;
    psb_i_t jacobiancount;
    psb_i_t verbosity;
    bool buildjacobian;
  };

  struct performanceinfo {
    psb_i_t *numjacobianspertimestep;
    psb_i_t *numfevalspertimestep;
    psb_i_t *numofnonliniterationpertimestep;
    psb_d_t *timepertimestep;
    psb_i_t *liniterperstep;
    psb_i_t numberoftimesteps;
 };

 #define NBMAX       30

 int main(int argc, char *argv[]){

    void *kmem;                              /* Pointer to KINSOL memory    */
    SUNLinearSolver LS;                      /* linear solver object        */
    psb_c_SolverOptions options;             /* Solver options              */
    struct user_data_for_f user_data;        /* User data for computing F,J */
    /* BLOCK data distribution */
    psb_i_t DataDistribution;                /* Data Distribution 2D/3D     */
    psb_l_t ng,nl,*vl;
    psb_i_t ix, iy, iz, el, nb,sizes[3],ijk[3],ijkinsert[3], dims[3];
    psb_l_t glob_row;
    psb_i_t npx, npy, npz, mynx, myny, mynz, iamx, iamy, iamz;
    psb_i_t *bndx, *bndy, *bndz;
    psb_l_t irow[10*NBMAX], icol[10*NBMAX];
    /* Problem data */
    N_Vector     u,constraints,su,sc;
    SUNMatrix    J;
    /* Input from file */
    psb_i_t nparms, idim, jdim, kdim, Nt, newtonmaxit, istop, itmax, itrace, irst, verbositylevel;
    psb_i_t NtToDo, kinsetprintlevel;
    psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks, Tmax, tol, rho, phi, pr;
    psb_d_t xmax, ymax, L;
    double  fnormtol, scsteptol;
    char inglobalstrategy[20], inetachoice[20];
    char methd[20],ptype[20],afmt[8]; /* Solve method, p. type, matrix format */
    /* Preconditioner Parameters */
    char smther[20],restr[20],prol[20],solve[20],variant[20]; //1st smoother
    psb_i_t jsweeps, novr, fill, invfill;
    psb_d_t thr;
    char smther2[20],restr2[20],prol2[20],solve2[20],variant2[20]; //2nd smoother
    psb_i_t jsweeps2, novr2, fill2, invfill2;;
    psb_d_t thr2;
    char par_aggr_alg[20], aggr_prol[20], aggr_type[20], aggr_ord[20]; // AMG aggregation
    char aggr_filter[20], mlcycle[20];
    psb_d_t mncrratio, *athresv, athres;
    psb_i_t thrvsz, csize, bcm_alg, bcm_sweeps, maxlevs;
    char cmat[20], csolve[20], csbsolve[20], cvariant[20], ckryl[20]; // coarsest-level solver
    char checkres[20], printres[20];
    psb_i_t cfill, cinvfill, cjswp, crkiter, crktrace, checkiter, printiter, outer_sweeps;
    psb_i_t aggr_size, crkistopc;
    psb_d_t cthres, crkeps, ctol;
    /* Parallel Environment */
    psb_c_ctxt *cctxt;
    psb_i_t ictxt,iam,np;
    psb_c_descriptor *cdh;
    /* Time Stepping */
    psb_d_t dt;
    /* Auxiliary variable */
    int i,k;
    psb_d_t normu0;
    /* Flags */
    psb_i_t info;
    bool verbose = SUNFALSE;
    /* Set global strategy flag */
    int globalstrategy, etachoice;
    /* Performance variables */
    struct performanceinfo perf_info;
    FILE *outfileforperformance;
    char outfilename[100];
    psb_d_t tic, toc, timecdh, timesolve;

    cctxt = psb_c_new_ctxt();
    psb_c_init(cctxt);
    psb_c_info(*cctxt,&iam,&np);
    psb_c_get_i_ctxt(*(cctxt),&ictxt,&info);

    if(verbose){
      fprintf(stdout,"Initialization of the Richards miniapp I'm %d of %d\n",iam,np);
      fflush(stdout);
    }else{
      if(iam == 0)
      fprintf(stdout,"Initialization of the Richards miniapp on %d processes\n"
        ,np);
      fflush(stdout);
    }

    /* ------------------------------------------------------------------------
     * Read Problem Settings from file
     *------------------------------------------------------------------------*/
    if (iam == 0) {
      //get_hparm(stdin, NULL); // Problem Parameters
      get_iparm(stdin,&idim);
      get_iparm(stdin,&jdim);
      get_iparm(stdin,&kdim);
      get_dparm(stdin,&xmax);
      get_dparm(stdin,&ymax);
      get_dparm(stdin,&L);
      get_dparm(stdin,&thetas);
      get_dparm(stdin,&thetar);
      get_dparm(stdin,&alpha);
      get_dparm(stdin,&beta);
      get_dparm(stdin,&a);
      get_dparm(stdin,&gamma);
      get_dparm(stdin,&Ks);
      get_dparm(stdin,&rho);
      get_dparm(stdin,&phi);
      get_dparm(stdin,&pr);
      get_dparm(stdin,&Tmax);
      get_iparm(stdin,&DataDistribution);
      //get_hparm(stdin, NULL); // Newton Parameters
      get_iparm(stdin,&Nt);
      get_iparm(stdin,&NtToDo);
      get_iparm(stdin,&newtonmaxit);
      get_dparm(stdin,&fnormtol);
      get_dparm(stdin,&scsteptol);
      get_hparm(stdin,inglobalstrategy);
      get_hparm(stdin,inetachoice);
      get_iparm(stdin,&verbositylevel);
      get_iparm(stdin,&kinsetprintlevel);
      //get_hparm(stdin, NULL); // PSBLAS parameters
      get_hparm(stdin,afmt);
      get_hparm(stdin,methd);
      get_iparm(stdin,&istop);
      get_iparm(stdin,&itmax);
      get_iparm(stdin,&itrace);
      get_iparm(stdin,&irst);
      get_dparm(stdin,&tol);
      get_hparm(stdin,ptype);
      //get_hparm(stdin, NULL); // First smoother (for all levels but coarsest)
      get_hparm(stdin,smther);    // Smoother type JACOBI FBGS GS BWGS BJAC AS. For 1-level, repeats previous.
      get_iparm(stdin,&jsweeps);   // Number of sweeps for smoother
      get_iparm(stdin,&novr);      // Number of overlap layers for AS preconditioner
      get_hparm(stdin,restr);     // AS restriction operator: NONE HALO
      get_hparm(stdin,prol);      // AS prolongation operator: NONE SUM AVG
      get_hparm(stdin,solve);     // Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
      get_hparm(stdin,variant);   // AINV variant: LLK, etc
      get_iparm(stdin,&fill);      // Fill level P for ILU(P) and ILU(T,P)
      get_iparm(stdin,&invfill);   // Inverse fill-in for INVK
      get_dparm(stdin,&thr);       // Threshold T for ILU(T,P)
      //get_hparm(stdin, NULL); // Second smoother, always ignored for non-ML
      get_hparm(stdin,smther2);    // Smoother type JACOBI FBGS GS BWGS BJAC AS. For 1-level, repeats previous.
      get_iparm(stdin,&jsweeps2);  // Number of sweeps for smoother
      get_iparm(stdin,&novr2);     // Number of overlap layers for AS preconditioner
      get_hparm(stdin,restr2);     // AS restriction operator: NONE HALO
      get_hparm(stdin,prol2);      // AS prolongation operator: NONE SUM AVG
      get_hparm(stdin,solve2);     // Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
      get_hparm(stdin,variant2);   // AINV variant: LLK, etc
      get_iparm(stdin,&fill2);     // Fill level P for ILU(P) and ILU(T,P)
      get_iparm(stdin,&invfill2);  // Inverse fill-in for INVK
      get_dparm(stdin,&thr2);      // Threshold T for ILU(T,P)
      //get_hparm(stdin, NULL);  //  Multilevel parameters
      get_hparm(stdin,mlcycle);    // Type of multilevel CYCLE: VCYCLE WCYCLE KCYCLE MULT ADD
      get_iparm(stdin,&outer_sweeps);// Number of outer sweeps for ML
      get_iparm(stdin,&maxlevs);     // Max Number of levels in a multilevel preconditioner; if <0, lib default
      get_iparm(stdin,&csize);       // Target coarse matrix size; if <0, lib default
      get_hparm(stdin,par_aggr_alg); // Parallel aggregation: DEC, SYMDEC, COUPLED
      get_hparm(stdin,aggr_prol);    // Aggregation prolongator: SMOOTHED UNSMOOTHED
      get_hparm(stdin,aggr_type);    // Type of aggregation: SOC1 (Vanek&B&M), SOC2(Gratton), MATCHBOXP
      get_iparm(stdin,&aggr_size);   // Requested size of the aggregates for MATCHBOXP
      get_hparm(stdin,aggr_ord);     // Ordering of aggregation NATURAL DEGREE
      get_hparm(stdin,aggr_filter);  // Filtering of matrix:  FILTER NOFILTER
      get_dparm(stdin,&mncrratio);   // Coarsening ratio, if < 0 use library default1
      //get_iparm(stdin,&thrvsz);      // Number of thresholds in vector, next line ignored if <= 0
      //if(thrvsz < 0) thrvsz = 1;
      //get_dparm2(stdin,athresv,thrvsz); // Thresholds
      get_dparm(stdin,&athres);      // Smoothed aggregation threshold, ignored if < 0
      //get_iparm(stdin,&bcm_alg);     // BCM method: 0 PREIS, 1 MC64, 2 SPRAL (auction)
      //get_iparm(stdin,&bcm_sweeps);  // BCM Pairing sweeps
      //get_hparm(stdin, NULL); // Coarse level solver
      get_hparm(stdin,csolve);     // Coarsest-level solver: MUMPS(global) UMF SLU SLUDIST JACOBI GS BJAC RKR(global)
      get_hparm(stdin,csbsolve);   // Coarsest-level subsolver for BJAC: ILU ILUT MILU UMF MUMPS(local) SLU RKR(local)
      get_hparm(stdin,cvariant);   // AINV Variant for the coarse solver
      get_hparm(stdin,ckryl);      // Krylov method for RKR solver/subsolver, ignored otherwise
      get_hparm(stdin,cmat);       // Coarsest-level matrix distribution: DIST  REPL
      get_iparm(stdin,&cfill);     // Coarsest-level fillin P for ILU(P) and ILU(T,P)
      get_iparm(stdin,&cinvfill);  // Coarsest-level inverse Fill level P for INVK
      get_dparm(stdin,&cthres);    // Coarsest-level threshold T for ILU(T,P)
      get_iparm(stdin,&cjswp);     // Number of sweeps for JACOBI/GS/BJAC coarsest-level solver
      get_iparm(stdin,&crkiter);   // maxit for RKR
      get_dparm(stdin,&crkeps);    // eps for RKR
      get_iparm(stdin,&crktrace);  // itrace for RKR
      get_iparm(stdin,&crkistopc); // istopc for RKR
      get_hparm(stdin,checkres);   // Check the BJAC residual: T F
      get_hparm(stdin,printres);   // Print the BJAC residual: T F
      get_iparm(stdin,&checkiter); // ITRACE for residual check
      get_iparm(stdin,&printiter); // ITRACE for residual print
      get_dparm(stdin,&ctol);      // Tolerance for exit from BJAC
      /* Print the Problem infos, information on the preconditioner are printed
       * by the PSBLAS interface.                                             */
      fprintf(stdout, "\nModel Parameters:\n");
      fprintf(stdout, "Saturated moisture contents        : %1.3f\n",thetas);
      fprintf(stdout, "Residual moisture contents         : %1.3f\n",thetar);
      fprintf(stdout, "Saturated hydraulic conductivity   : %1.3e\n",Ks);
      fprintf(stdout, "Water density (ρ)                  : %1.3e\n",rho);
      fprintf(stdout, "Porosity of the medium (ϕ)         : %1.3e\n",phi);
      fprintf(stdout, "                                   : (α        ,β    ,a        ,γ    )\n");
      fprintf(stdout, "van Genuchten empirical parameters : (%1.3e,%1.3f,%1.3e,%1.3f)\n",
        alpha,beta,a,gamma);
      fprintf(stdout, "Initial value of the pressure head is %lf cm\n",pr);
      fprintf(stdout, "Solving in a box [0,%lf]x[0,%lf]x[0,%lf] with %dx%dx%d dofs\n",xmax,ymax,L,idim,jdim,kdim);
      fflush(stdout);
   }
    psb_c_ibcast(*cctxt,1,&idim,0);
    psb_c_ibcast(*cctxt,1,&jdim,0);
    psb_c_ibcast(*cctxt,1,&kdim,0);
    psb_c_dbcast(*cctxt,1,&xmax,0);
    psb_c_dbcast(*cctxt,1,&ymax,0);
    psb_c_dbcast(*cctxt,1,&L,0);
    psb_c_dbcast(*cctxt,1,&thetas,0);
    psb_c_dbcast(*cctxt,1,&thetar,0);
    psb_c_dbcast(*cctxt,1,&alpha,0);
    psb_c_dbcast(*cctxt,1,&beta,0);
    psb_c_dbcast(*cctxt,1,&a,0);
    psb_c_dbcast(*cctxt,1,&gamma,0);
    psb_c_dbcast(*cctxt,1,&Ks,0);
    psb_c_dbcast(*cctxt,1,&rho,0);
    psb_c_dbcast(*cctxt,1,&phi,0);
    psb_c_dbcast(*cctxt,1,&pr,0);
    psb_c_dbcast(*cctxt,1,&Tmax,0);
    psb_c_ibcast(*cctxt,1,&DataDistribution,0);
    // Newton Parameters
    psb_c_ibcast(*cctxt,1,&Nt,0);
    psb_c_ibcast(*cctxt,1,&NtToDo,0);
    psb_c_ibcast(*cctxt,1,&newtonmaxit,0);
    psb_c_dbcast(*cctxt,1,&fnormtol,0);
    psb_c_dbcast(*cctxt,1,&scsteptol,0);
    psb_c_hbcast(*cctxt,inglobalstrategy,0);
    psb_c_hbcast(*cctxt,inetachoice,0);
    psb_c_ibcast(*cctxt,1,&verbositylevel,0);
    psb_c_ibcast(*cctxt,1,&kinsetprintlevel,0);
    // PSBLAS parameters
    psb_c_hbcast(*cctxt,afmt,0);
    psb_c_hbcast(*cctxt,methd,0);
    psb_c_ibcast(*cctxt,1,&istop,0);
    psb_c_ibcast(*cctxt,1,&itmax,0);
    psb_c_ibcast(*cctxt,1,&irst,0);
    psb_c_ibcast(*cctxt,1,&itrace,0);
    psb_c_dbcast(*cctxt,1,&tol,0);
    psb_c_hbcast(*cctxt,ptype,0);
    // First smoother (for all levels but coarsest)
    psb_c_hbcast(*cctxt,smther,0);    // Smoother type JACOBI FBGS GS BWGS BJAC AS. For 1-level, repeats previous.
    psb_c_ibcast(*cctxt,1,&jsweeps,0);// Number of sweeps for smoother
    psb_c_ibcast(*cctxt,1,&novr,0);   // Number of overlap layers for AS preconditioner
    psb_c_hbcast(*cctxt,restr,0);     // AS restriction operator: NONE HALO
    psb_c_hbcast(*cctxt,prol,0);      // AS prolongation operator: NONE SUM AVG
    psb_c_hbcast(*cctxt,solve,0);     // Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
    psb_c_hbcast(*cctxt,variant,0);   // AINV variant: LLK, etc
    psb_c_ibcast(*cctxt,1,&fill,0);   // Fill level P for ILU(P) and ILU(T,P)
    psb_c_ibcast(*cctxt,1,&invfill,0);// Inverse fill-in for INVK
    psb_c_dbcast(*cctxt,1,&thr,0);    // Threshold T for ILU(T,P)
    // Second smoother, always ignored for non-ML
    psb_c_hbcast(*cctxt,smther2,0);   // Smoother type JACOBI FBGS GS BWGS BJAC AS. For 1-level, repeats previous.
    psb_c_ibcast(*cctxt,1,&jsweeps2,0); // Number of sweeps for smoother
    psb_c_ibcast(*cctxt,1,&novr2,0);  // Number of overlap layers for AS preconditioner
    psb_c_hbcast(*cctxt,restr2,0);    // AS restriction operator: NONE HALO
    psb_c_hbcast(*cctxt,prol2,0);     // AS prolongation operator: NONE SUM AVG
    psb_c_hbcast(*cctxt,solve2,0);    // Subdomain solver for BJAC/AS: JACOBI GS BGS ILU ILUT MILU MUMPS SLU UMF
    psb_c_hbcast(*cctxt,variant2,0);  // AINV variant: LLK, etc
    psb_c_ibcast(*cctxt,1,&fill2,0);  // Fill level P for ILU(P) and ILU(T,P)
    psb_c_ibcast(*cctxt,1,&invfill2,0);// Inverse fill-in for INVK
    psb_c_dbcast(*cctxt,1,&thr2,0);   // Threshold T for ILU(T,P)
    //  Multilevel parameters
    psb_c_hbcast(*cctxt,mlcycle,0);   // Type of multilevel CYCLE: VCYCLE WCYCLE KCYCLE MULT ADD
    psb_c_ibcast(*cctxt,1,&outer_sweeps,0);// Number of outer sweeps for ML
    psb_c_ibcast(*cctxt,1,&maxlevs,0);// Max Number of levels in a multilevel preconditioner; if <0, lib default
    psb_c_ibcast(*cctxt,1,&csize,0);  // Target coarse matrix size; if <0, lib default
    psb_c_hbcast(*cctxt,par_aggr_alg,0); // Parallel aggregation: DEC, SYMDEC, COUPLED
    psb_c_hbcast(*cctxt,aggr_prol,0); // Aggregation prolongator: SMOOTHED UNSMOOTHED
    psb_c_hbcast(*cctxt,aggr_type,0); // Type of aggregation: SOC1 (Vanek&B&M), SOC2(Gratton), MATCHBOXP
    psb_c_ibcast(*cctxt,1,&aggr_size,0); // Requested size of the aggregates
    psb_c_hbcast(*cctxt,aggr_ord,0);  // Ordering of aggregation NATURAL DEGREE
    psb_c_hbcast(*cctxt,aggr_filter,0);// Filtering of matrix:  FILTER NOFILTER
    psb_c_dbcast(*cctxt,1,&mncrratio,0);// Coarsening ratio, if < 0 use library default1
    psb_c_ibcast(*cctxt,1,&thrvsz,0); // Number of thresholds in vector, next line ignored if <= 0
    //psb_c_dbcast(*cctxt,thrvsz,athresv,0); // Thresholds
    psb_c_dbcast(*cctxt,1,&athres,0); // Smoothed aggregation threshold, ignored if < 0
    //psb_c_ibcast(*cctxt,1,&bcm_alg,0);// BCM method: 0 PREIS, 1 MC64, 2 SPRAL (auction)
    //psb_c_ibcast(*cctxt,1,&bcm_sweeps,0); // BCM Pairing sweeps
    // Coarse level solver
    psb_c_hbcast(*cctxt,csolve,0);     // Coarsest-level solver: MUMPS(global) UMF SLU SLUDIST JACOBI GS BJAC RKR(global)
    psb_c_hbcast(*cctxt,csbsolve,0);   // Coarsest-level subsolver for BJAC: ILU ILUT MILU UMF MUMPS(local) SLU RKR(local)
    psb_c_hbcast(*cctxt,cvariant,0);   // AINV Variant for the coarse solver
    psb_c_hbcast(*cctxt,ckryl,0);      // Krylov method for RKR solver/subsolver, ignored otherwise
    psb_c_hbcast(*cctxt,cmat,0);       // Coarsest-level matrix distribution: DIST  REPL
    psb_c_ibcast(*cctxt,1,&cfill,0);   // Coarsest-level fillin P for ILU(P) and ILU(T,P)
    psb_c_ibcast(*cctxt,1,&cinvfill,0);// Coarsest-level inverse Fill level P for INVK
    psb_c_dbcast(*cctxt,1,&cthres,0);  // Coarsest-level threshold T for ILU(T,P)
    psb_c_ibcast(*cctxt,1,&cjswp,0);   // Number of sweeps for JACOBI/GS/BJAC coarsest-level solver
    psb_c_ibcast(*cctxt,1,&crkiter,0); // maxit for RKR
    psb_c_dbcast(*cctxt,1,&crkeps,0);  // eps for RKR
    psb_c_ibcast(*cctxt,1,&crktrace,0);// itrace for RKR
    psb_c_ibcast(*cctxt,1,&crkistopc,0);// istopc for RKR
    psb_c_hbcast(*cctxt,checkres,0);   // Check the BJAC residual: T F
    psb_c_hbcast(*cctxt,printres,0);   // Print the BJAC residual: T F
    psb_c_ibcast(*cctxt,1,&checkiter,0); // ITRACE for residual check
    psb_c_ibcast(*cctxt,1,&printiter,0); // ITRACE for residual print
    psb_c_dbcast(*cctxt,1,&ctol,0);      // Tolerance for exit from BJAC

    /*-------------------------------------------------------------------------
     * Perform a 3D BLOCK data distribution
     *------------------------------------------------------------------------*/
    if(iam==0){
      fprintf(stdout, "\nStarting %dD BLOCK data distribution\n", DataDistribution);
      fflush(stdout);
    }
    psb_c_barrier(*cctxt);
    cdh = psb_c_new_descriptor();
    psb_c_set_index_base(0);
    if (DataDistribution == 3) {
       /* 3D Block Distribution : used when we scale proportionally all the
          grid size with the number of processes.                             */
       ng = ((psb_l_t) idim)*jdim*kdim;
       nb = (ng+np-1)/np;
       nl = nb;
       if ( (ng -iam*nb) < nl) nl = ng -iam*nb;
         fprintf(stdout,"%d: Input data %d %ld %d %ld\n",iam,idim,ng,nb, nl);
       if ((vl=malloc((nb+1)*sizeof(psb_l_t)))==NULL) {
         fprintf(stderr,"On %d: malloc failure\n",iam);
         psb_c_abort(*cctxt);
      }
      i = ((psb_l_t)iam) * nb;
      for (k=0; k < nl; k++){
       vl[k] = i+k;
      }
      if ((info=psb_c_cdall_vl(nl,vl,*cctxt,cdh))!=0) {
         fprintf(stderr,"From cdall: %d\nBailing out\n",info);
         psb_c_abort(*cctxt);
      }
   } else if (DataDistribution == 2) {
      /* We build here the communicator for the case in which only Nx and Ny
      scale with the number of processes and Nz remains fixed. We use the MPI
      function to divide the grid on the xy plane, then we impose that the
      number of processes along z is 1: npx × npy × 1 processor grid*/
      ng = ((psb_l_t) idim)*jdim*kdim;
      if( np == 1){
         dims[0] = 0; dims[1] = 0; dims[2] = 0;
      } else {
         dims[0] = 0; dims[1] = 0; dims[2] = 1;
      }
      info = MPI_Dims_create( np , 3, dims);
      npx = dims[0];
      npy = dims[1];
      npz = dims[2];
      sizes[0] = npx; sizes[1] = npy; sizes[2] = npz;

      if(iam == 0 && verbositylevel > 0)
         fprintf(stderr, "Processor grid is [%d,%d,%d]\n",npx,npy,npz);

      // These vectors will contain tnhe bounds on the indexes owned by the
      // different processes
      bndx = (psb_i_t *) malloc( (npx+1)*sizeof(psb_i_t) );
      bndy = (psb_i_t *) malloc( (npy+1)*sizeof(psb_i_t) );
      bndz = (psb_i_t *) malloc( (npz+1)*sizeof(psb_i_t) );

      info = psb_c_i_idx2ijk(ijk,iam,sizes,3,0);
      iamx = ijk[0]; iamy = ijk[1]; iamz = ijk[2];

      // Now let's split the 3D cube in hexahedra
      info = psb_c_dist1didx(idim,npx,0,npx+1,bndx);
      mynx = bndx[iamx+1] - bndx[iamx];
      if( verbositylevel > 0)
         fprintf(stderr, "Process %d mynx = %d - %d = %d\n",
                  iam,bndx[iamx+1],bndx[iamx],mynx);
      info = psb_c_dist1didx(jdim,npy,0,npy+1,bndy);
      myny = bndy[iamy+1] - bndy[iamy];
      if( verbositylevel > 0)
         fprintf(stderr, "Process %d myny = %d - %d = %d\n",
                  iam,bndy[iamy+1],bndy[iamy],myny);
      info = psb_c_dist1didx(kdim,npz,0,npz+1,bndz);
      mynz = bndz[iamz+1] - bndz[iamz];
      if( verbositylevel > 0)
         fprintf(stderr, "Process %d myny = %d - %d = %d\n",
                  iam,bndz[iamz+1],bndz[iamz],mynz);

      // How many indices do I own?
      nl = mynx*myny*mynz;
      if( verbositylevel > 0)
         fprintf(stderr, "Process %d owns %ld indexes (%d,%d,%d) \n",
         iam, nl, mynx, myny, mynz);
      // We populate the vector containing the local indexes we know of
      vl = (psb_l_t *) malloc((nl+1)*sizeof(psb_l_t));
      sizes[0] = idim; sizes[1] = jdim; sizes[2] = kdim;
      nb = 0;
      for (int i = bndx[iamx]; i < bndx[iamx+1]; i++){
         for(int j = bndy[iamy]; j < bndy[iamy+1]; j++){
            for(int k = bndz[iamz]; k < bndz[iamz+1]; k++){
               ijk[0] = i; ijk[1] = j; ijk[2] = k;
               vl[nb] = psb_c_l_ijk2idx(ijk,sizes,3,0);
               nb = nb+1;
            }
         }
      }
      if( nb != nl ){
         fprintf(stderr, "Inizialization error: NR vs NLR %d /= %ld (%d,%d,%d)\n"
            ,nb,nl,mynx,myny,mynz);
         free(vl);
         free(bndx);
         free(bndy);
         free(bndz);
         psb_c_barrier(*cctxt);
         psb_c_abort(*cctxt);
      }
      // Then we allocate the comunicator on the given global/local indexes
      if ((info=psb_c_cdall_vl(nl,vl,*cctxt,cdh))!=0) {
         fprintf(stderr,"From cdall: %d\nBailing out\n",info);
         free(vl);
         free(bndx);
         free(bndy);
         free(bndz);
         psb_c_abort(*cctxt);
      }
      // we no longer need these entries:
      free(bndx);
      free(bndy);
      free(bndz);
   } else {
      fprintf(stderr, "Unknown Data Distribution %d\n",DataDistribution);
      psb_c_barrier(*cctxt);
      psb_c_abort(*cctxt);
   }
   /* We store into the user_data_for_f the owned indices, these are then
   used to compute the Jacobian and the function evaluations */
   user_data.vl     = vl;
   user_data.idim   = idim;
   user_data.jdim   = jdim;
   user_data.kdim   = kdim;
   user_data.xmax   = xmax;
   user_data.ymax   = ymax;
   user_data.L      = L;
   user_data.nl     = nl;
   user_data.thetas = thetas;
   user_data.thetar = thetar;
   user_data.alpha  = alpha;
   user_data.beta   = beta;
   user_data.a      = a;
   user_data.gamma  = gamma;
   user_data.Ks     = Ks;
   user_data.rho    = rho;
   user_data.phi    = phi;
   user_data.pr     = pr;
   user_data.timestep = 1;
   user_data.functioncount = 0;
   user_data.jacobiancount = 0;
   user_data.verbosity = verbositylevel; // 0 Do not print/dump anything,
                            // 1 Print what we are computing,
                            // 2 Print and dump everything (MEMORY! SMALL DEBUG)
   user_data.buildjacobian = true; // At the begin we always need a Jacobian

   /*We need to reuse the same communicator many times, namely every time we
   need to populate a new Jacobian. Therefore we use the psb_c_cdins routine
   to generate the distributed adjacency graph for our problem.              */
   sizes[0] = idim; sizes[1] = jdim; sizes[2] = kdim;
   for (i=0; i < nl;  i++) {
     glob_row=vl[i];
     el = 0;
     psb_c_l_idx2ijk(ijk,glob_row,sizes,3,0);
     ix = ijk[0]; iy = ijk[1]; iz = ijk[2];
     /*  term depending on   (i-1,j,k)        */
     if(ix != 0){
       ijkinsert[0]=ix-1; ijkinsert[1]=iy; ijkinsert[2]=iz;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     /*  term depending on     (i,j-1,k)        */
     if (iy != 0){
       ijkinsert[0]=ix; ijkinsert[1]=iy-1; ijkinsert[2]=iz;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     /* term depending on      (i,j,k-1)        */
     if (iz != 0){
       ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz-1;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     /* term depending on      (i,j,k)          */
     ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz;
     icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
     el=el+1;
     /*  term depending on     (i,j,k+1)        */
     if (iz != kdim-1) {
       ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz+1;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     /*  term depending on     (i,j+1,k)        */
     if (iy != jdim-1){
       ijkinsert[0]=ix; ijkinsert[1]=iy+1; ijkinsert[2]=iz;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     /* term depending on      (i+1,j,k)        */
     if (ix != idim-1){
       ijkinsert[0]=ix+1; ijkinsert[1]=iy; ijkinsert[2]=iz;
       icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
       el=el+1;
     }
     for (k=0; k<el; k++) irow[k]=glob_row;
     if ((info=psb_c_cdins(el,irow,icol,cdh))!=0)
      fprintf(stderr,"From psb_c_cdins: %d\n",info);
   }

   tic = psb_c_wtime();
   if ((info=psb_c_cdasb(cdh))!=0)  return(info);
   toc = psb_c_wtime();
   timecdh = toc-tic;


   if (iam == 0){
     printf("Built communicator on %ld global rows\n",psb_c_cd_get_global_rows(cdh));
     fprintf(stdout,"Communicator Building time: %lf s\n",timecdh);
   }

   /*-------------------------------------------------------
    * Linear Solver Setup and construction
    *-------------------------------------------------------*/
    psb_c_DefaultSolverOptions(&options);
    options.eps    = tol;
    options.itmax  = itmax;
    options.irst   = irst;
    options.itrace = itrace;
    options.istop  = istop;
    if (iam == 0){
      fprintf(stdout,"Solver options:\n");
      fprintf(stdout,"%s solver maxit = %d, irst = %d, itrace = %d, istop = %d, tol = %e\n",
         methd,itmax,irst,itrace,istop,tol);
    }
    /*-------------------------------------------------------------------------
     * Create PSBLAS/AMG4PSBLAS linear solver
     *-------------------------------------------------------------------------*/
    LS = SUNLinSol_PSBLAS(options, methd, ptype, cctxt);
    if(check_flag((void *)LS, "SUNLinSol_PSBLAS", 0, iam)) psb_c_abort(*cctxt);

    SUNLinSolInitialize_PSBLAS(LS);

    if(iam == 0) fprintf(stdout, "Setting the Preconditioner Options.\n Preconditioner is: %s\n",ptype);

    /*-------------------------------------------------------------------------
     * Set AMG4PSBLAS options: anything is preconditionable!
     *-------------------------------------------------------------------------*/
    if( strcmp(ptype,"NONE")==0 || strcmp(ptype,"NOPREC")==0 ){
        // Do nothing, keep defaults
    }else if( strcmp(ptype,"L1-JACOBI")==0 || strcmp(ptype,"JACOBI")==0 || strcmp(ptype,"GS")==0 || strcmp(ptype,"FWGS")==0 || strcmp(ptype,"FBGS")==0  ){
      info = SUNLinSolSeti_PSBLAS(LS,"SMOOTHER_SWEEPS",jsweeps);
      if (check_flag(&info, "SMOOTHER_SWEEPS", 1, iam)) psb_c_abort(*cctxt);
    }else if( strcmp(ptype,"BJAC")==0 || strcmp(ptype,"L1-BJAC")==0 ){
      info = SUNLinSolSeti_PSBLAS(LS,"SMOOTHER_SWEEPS",jsweeps);
      if (check_flag(&info, "SMOOTHER_SWEEPS", 1, iam)) psb_c_abort(*cctxt);
      if (strcmp(solve,"INVK")){
        // TO BE FIXED ON INTERFACE MADE
      }else if (strcmp(solve,"INVT")){
        // TO BE FIXED ON INTERFACE MADE
      }else if (strcmp(solve,"AINV")){
        // TO BE FIXED ON INTERFACE MADE
        info = SUNLinSolSetc_PSBLAS(LS,"AINV_ALG",variant);
      }else{
        info = SUNLinSolSetc_PSBLAS(LS,"SUB_SOLVE",solve);
        if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
      }
      info = SUNLinSolSeti_PSBLAS(LS,"SUB_FILLIN",fill);
      if (check_flag(&info, "SUB_FILLIN", 1, iam)) psb_c_abort(*cctxt);
      info = SUNLinSolSeti_PSBLAS(LS,"INV_FILLIN",invfill);
      if (check_flag(&info, "INV_FILLIN", 1, iam)) psb_c_abort(*cctxt);
      info = SUNLinSolSetr_PSBLAS(LS,"SUB_ILUTHRS",thr);
      if (check_flag(&info, "SUB_ILUTHRS", 1, iam)) psb_c_abort(*cctxt);
    }else if( strcmp(ptype,"AS")==0){
      info = SUNLinSolSeti_PSBLAS(LS,"SMOOTHER_SWEEPS",jsweeps);
      if (check_flag(&info, "SMOOTHER_SWEEPS", 1, iam)) psb_c_abort(*cctxt);
      info = SUNLinSolSeti_PSBLAS(LS,"SUB_OVR",novr);
      if (check_flag(&info, "SUB_OVR", 1, iam)) psb_c_abort(*cctxt);
      info = SUNLinSolSetc_PSBLAS(LS,"SUB_RESTR",restr);
      if (check_flag(&info, "SUB_RESTR", 1, iam)) psb_c_abort(*cctxt);
      info = SUNLinSolSetc_PSBLAS(LS,"SUB_PROL",prol);
      if (strcmp(solve,"INVK")){
        // TO BE FIXED ON INTERFACE MADE
      }else if (strcmp(solve,"INVT")){
        // TO BE FIXED ON INTERFACE MADE
      }else if (strcmp(solve,"AINV")){
        // TO BE FIXED ON INTERFACE MADE
        info = SUNLinSolSetc_PSBLAS(LS,"AINV_ALG",variant);
      }else{
        info = SUNLinSolSetc_PSBLAS(LS,"SUB_SOLVE",solve);
        if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
      }
      info = SUNLinSolSeti_PSBLAS(LS,"SUB_FILLIN",fill);
      if (check_flag(&info, "SUB_FILLIN", 1, iam)) psb_c_abort(*cctxt);
      info = SUNLinSolSeti_PSBLAS(LS,"INV_FILLIN",invfill);
      if (check_flag(&info, "INV_FILLIN", 1, iam)) psb_c_abort(*cctxt);
      info = SUNLinSolSetr_PSBLAS(LS,"SUB_ILUTHRS",thr);
      if (check_flag(&info, "SUB_ILUTHRS", 1, iam)) psb_c_abort(*cctxt);
    }else if( strcmp(ptype,"ML")==0){
      if(iam == 0) fprintf(stdout, "Setting options for the %s preconditioner\n",ptype);
      // Multilevel Preconditioner
      info = SUNLinSolSetc_PSBLAS(LS,"ML_CYCLE",mlcycle);
      info = SUNLinSolSeti_PSBLAS(LS,"OUTER_SWEEPS",outer_sweeps);
      info = SUNLinSolSetc_PSBLAS(LS,"PAR_AGGR_ALG",par_aggr_alg);
      info = SUNLinSolSetc_PSBLAS(LS,"AGGR_PROL",aggr_prol);
      info = SUNLinSolSetc_PSBLAS(LS,"AGGR_TYPE",aggr_type);
      info = SUNLinSolSeti_PSBLAS(LS,"AGGR_SIZE",aggr_size);

      info = SUNLinSolSetc_PSBLAS(LS,"AGGR_ORD",aggr_ord);
      info = SUNLinSolSetc_PSBLAS(LS,"AGGR_FILTER",aggr_filter);
      if(csize > 0){
        info = SUNLinSolSeti_PSBLAS(LS,"MIN_COARSE_SIZE",csize);
      }
      if(mncrratio > 1){
        info = SUNLinSolSeti_PSBLAS(LS,"MIN_CR_RATIO",mncrratio);
      }
      if(maxlevs > 0){
        info = SUNLinSolSeti_PSBLAS(LS,"MAX_LEVS",maxlevs);
      }
      if(athres > 0.0){
        info = SUNLinSolSetr_PSBLAS(LS,"AGGR_THRESH",athres);
      }
      // Missing command on threshold vector: need to fix input
      info = SUNLinSolSetc_PSBLAS(LS,"SMOOTHER_TYPE",smther);
      info = SUNLinSolSeti_PSBLAS(LS,"SMOOTHER_SWEEPS",jsweeps);
      // FIRST SMOOTHER
      if (strcmp(smther,"GS")==0 || strcmp(smther,"BWGS")==0 || strcmp(smther,"FBGS")==0 || strcmp(smther,"JACOBI")==0 || strcmp(smther,"L1-JACOBI")==0 || strcmp(smther,"L1-FBGS")==0 ){
        // do nothing
      }else{
        info = SUNLinSolSeti_PSBLAS(LS,"SUB_OVR",novr);
        info = SUNLinSolSetc_PSBLAS(LS,"SUB_RESTR",restr);
        info = SUNLinSolSetc_PSBLAS(LS,"SUB_PROL",prol);
        if (strcmp(solve,"INVK")==0){
          info = SUNLinSolSetc_PSBLAS(LS,"SUB_SOLVE",solve2);
          if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
        }else if (strcmp(solve,"INVT")==0){
          info = SUNLinSolSetc_PSBLAS(LS,"SUB_SOLVE",solve2);
          if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
        }else if (strcmp(solve,"AINV")==0){
          info = SUNLinSolSetc_PSBLAS(LS,"SUB_SOLVE",solve2);
          if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
          info = SUNLinSolSetc_PSBLAS(LS,"AINV_ALG",variant);
          if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
        }else{
          info = SUNLinSolSetc_PSBLAS(LS,"SUB_SOLVE",solve);
          if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
        }
        info = SUNLinSolSeti_PSBLAS(LS,"SUB_FILLIN",fill);
        if (check_flag(&info, "SUB_FILLIN", 1, iam)) psb_c_abort(*cctxt);
        info = SUNLinSolSeti_PSBLAS(LS,"INV_FILLIN",invfill);
        if (check_flag(&info, "INV_FILLIN", 1, iam)) psb_c_abort(*cctxt);
        info = SUNLinSolSetr_PSBLAS(LS,"SUB_ILUTHRS",thr);
        if (check_flag(&info, "SUB_ILUTHRS", 1, iam)) psb_c_abort(*cctxt);
      }
      // SECOND SMOOTHER
      if ( !strcmp(smther2,"NONE")){
        if (strcmp(smther2,"GS") || strcmp(smther2,"BWGS") || strcmp(smther2,"FBGS") || strcmp(smther2,"JACOBI") || strcmp(smther2,"L1-JACOBI") || strcmp(smther2,"L1-FBGS") ){
          // do nothing
        }else{
          info = SUNLinSolSeti_PSBLAS(LS,"SUB_OVR",novr2);
          info = SUNLinSolSetc_PSBLAS(LS,"SUB_RESTR",restr2);
          info = SUNLinSolSetc_PSBLAS(LS,"SUB_PROL",prol2);
          if (strcmp(solve2,"INVK")==0){
             info = SUNLinSolSetc_PSBLAS(LS,"SUB_SOLVE",solve2);
             if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
          }else if (strcmp(solve2,"INVT")==0){
             info = SUNLinSolSetc_PSBLAS(LS,"SUB_SOLVE",solve2);
             if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
          }else if (strcmp(solve2,"AINV")==0){
            info = SUNLinSolSetc_PSBLAS(LS,"SUB_SOLVE",solve2);
            if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
            info = SUNLinSolSetc_PSBLAS(LS,"AINV_ALG",variant2);
            if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
          }else{
            info = SUNLinSolSetc_PSBLAS(LS,"SUB_SOLVE",solve2);
            if (check_flag(&info, "SUB_SOLVE", 1, iam)) psb_c_abort(*cctxt);
          }
          info = SUNLinSolSeti_PSBLAS(LS,"SUB_FILLIN",fill2);
          if (check_flag(&info, "SUB_FILLIN", 1, iam)) psb_c_abort(*cctxt);
          info = SUNLinSolSeti_PSBLAS(LS,"INV_FILLIN",invfill2);
          if (check_flag(&info, "INV_FILLIN", 1, iam)) psb_c_abort(*cctxt);
          info = SUNLinSolSetr_PSBLAS(LS,"SUB_ILUTHRS",thr2);
          if (check_flag(&info, "SUB_ILUTHRS", 1, iam)) psb_c_abort(*cctxt);
        }
      }
      // COARSE SOLVE
      info = SUNLinSolSetc_PSBLAS(LS,"COARSE_SOLVE",csolve);
      if (check_flag(&info, "COARSE_SOLVE", 1, iam)) psb_c_abort(*cctxt);
      if( strcmp(csolve,"BJAC")==0 || strcmp(csolve,"L1-BJAC")==0 ){
         info = SUNLinSolSetc_PSBLAS(LS,"COARSE_SUBSOLVE",csbsolve);
         if (check_flag(&info, "COARSE_SUBSOLVE", 1, iam)) psb_c_abort(*cctxt);
         if( strcmp(csbsolve,"MUMPS")==0){
            // TODO : Need to implement function to pass the BLR options!
            info = SUNLinSolSetc_PSBLAS(LS,"MUMPS_LOC_GLOB","LOCAL_SOLVER");
            if (check_flag(&info, "MUMPS_LOC_GLOB", 1, iam)) psb_c_abort(*cctxt);
            info = SUNLinSolSeti_PSBLAS(LS,"MUMPS_IPAR_ENTRY",1);
            if (check_flag(&info, "MUMPS_IPAR_ENTRY", 1, iam)) psb_c_abort(*cctxt);
            info = SUNLinSolSetr_PSBLAS(LS,"MUMPS_RPAR_ENTRY",0.4);
            if (check_flag(&info, "MUMPS_RPAR_ENTRY", 1, iam)) psb_c_abort(*cctxt);
         }
         info = SUNLinSolSetc_PSBLAS(LS,"SMOOTHER_STOP",checkres);
         if (check_flag(&info, "SMOOTHER_STOP", 1, iam)) psb_c_abort(*cctxt);
         info = SUNLinSolSetc_PSBLAS(LS,"SMOOTHER_TRACE",printres);
         if (check_flag(&info, "SMOOTHER_TRACE", 1, iam)) psb_c_abort(*cctxt);
         info = SUNLinSolSetr_PSBLAS(LS,"SMOOTHER_STOPTOL",ctol);
         if (check_flag(&info, "SMOOTHER_STOPTOL", 1, iam)) psb_c_abort(*cctxt);
         info = SUNLinSolSeti_PSBLAS(LS,"SMOOTHER_ITRACE",printiter);
         if (check_flag(&info, "SMOOTHER_ITRACE", 1, iam)) psb_c_abort(*cctxt);
         info = SUNLinSolSeti_PSBLAS(LS,"SMOOTHER_RESIDUAL",checkiter);
         if (check_flag(&info, "SMOOTHER_RESIDUAL", 1, iam)) psb_c_abort(*cctxt);
      }
      if( strcmp(csolve,"KRM")==0 ){
         info = SUNLinSolSetc_PSBLAS(LS,"KRM_METHOD",ckryl);
         if (check_flag(&info, "KRM_METHOD", 1, iam)) psb_c_abort(*cctxt);
         info = SUNLinSolSeti_PSBLAS(LS,"KRM_ITMAX",crkiter);
         if (check_flag(&info, "KRM_ITMAX", 1, iam)) psb_c_abort(*cctxt);
         info = SUNLinSolSetr_PSBLAS(LS,"KRM_EPS",crkeps);
         if (check_flag(&info, "KRM_EPS", 1, iam)) psb_c_abort(*cctxt);
         info = SUNLinSolSeti_PSBLAS(LS,"KRM_ITRACE",crktrace);
         if (check_flag(&info, "KRM_ITRACE", 1, iam)) psb_c_abort(*cctxt);
         info = SUNLinSolSeti_PSBLAS(LS,"KRM_ISTOPC",crkistopc);
         if (check_flag(&info, "KRM_ISTOPC", 1, iam)) psb_c_abort(*cctxt);
      }
      info = SUNLinSolSetc_PSBLAS(LS,"COARSE_MAT",cmat);
      if (check_flag(&info, "COARSE_MAT", 1, iam)) psb_c_abort(*cctxt);
      info = SUNLinSolSeti_PSBLAS(LS,"COARSE_FILLIN",cfill);
      if (check_flag(&info, "COARSE_FILLIN", 1, iam)) psb_c_abort(*cctxt);
      info = SUNLinSolSetr_PSBLAS(LS,"COARSE_ILUTHRS",cthres);
      if (check_flag(&info, "COARSE_ILUTHRS", 1, iam)) psb_c_abort(*cctxt);
      info = SUNLinSolSeti_PSBLAS(LS,"COARSE_SWEEPS",cjswp);
      if (check_flag(&info, "COARSE_SWEEPS", 1, iam)) psb_c_abort(*cctxt);
    }else{
      if(iam == 0); fprintf(stderr, "Warning %s is an unknown preconditioner!\n",ptype);
    }

    user_data.LS = &LS;

    /*-----------------------------------------------------*
     * Setting up performance information structure        *
     *-----------------------------------------------------*/

    perf_info.numberoftimesteps = NtToDo;
    perf_info.numjacobianspertimestep = (psb_i_t *) malloc(NtToDo*sizeof(psb_i_t));
    perf_info.numfevalspertimestep = (psb_i_t *) malloc(NtToDo*sizeof(psb_i_t));
    perf_info.timepertimestep = (psb_d_t *) malloc(NtToDo*sizeof(psb_d_t));
    perf_info.liniterperstep = (psb_i_t *) malloc(NtToDo*sizeof(psb_i_t));
    perf_info.numofnonliniterationpertimestep = (psb_i_t *) malloc(NtToDo*sizeof(psb_i_t));

   /*------------------------------------------------------*
    * Solution vector and auxiliary data                   *
    *------------------------------------------------------*/
    constraints = NULL;
    constraints = N_VNew_PSBLAS(cctxt, cdh);
    u = NULL;
    u = N_VNew_PSBLAS(cctxt, cdh);
    sc = NULL;
    sc = N_VNew_PSBLAS(cctxt, cdh);
    su = NULL;
    su = N_VNew_PSBLAS(cctxt, cdh);
    J = NULL;
    J = SUNPSBLASMatrix(cctxt, cdh);
    user_data.B = NULL;
    user_data.B = SUNPSBLASMatrix(cctxt, cdh);
    user_data.oldpressure = N_VNew_PSBLAS(cctxt, cdh);

   /*------------------------------------------------------*
    * We can now initialize the time loop                  *
    *------------------------------------------------------*/
   dt = Tmax/(Nt+1);
   user_data.dt = dt;
   N_VConst(pr,u);  // Initial Condition
   N_VLinearSum(1.0,u,0.0,user_data.oldpressure,user_data.oldpressure);

   N_VConst(0.0,constraints);      // No constraints
   N_VConst(1.0,sc);               // Unweighted norm
   N_VConst(1.0,su);               // Unweighted norm

   /*------------------------------------------------------*
    * Initialization of the nonlinear solver               *
    *------------------------------------------------------*/
    if (strcmp(inglobalstrategy,"LINESEARCH") == 0){
      // Apply the lineasearch as detailed in the Kinsol manual.
      globalstrategy = KIN_LINESEARCH;
    } else {
      // Newton is assumed to be already global convergent, i.e, we start
      // from the basin of attraction of the root, and thus no LINESEARCH is
      // required.
      globalstrategy = KIN_NONE;
    }
    if (strcmp(inetachoice,"ETACHOICE1") == 0){
      etachoice = KIN_ETACHOICE1;
    } else if (strcmp(inetachoice,"ETACHOICE2") == 0) {
      etachoice = KIN_ETACHOICE2;
    } else {
      // In this case the choice of the parameter η is taken from the
      // ϵ inputed for the Krylov method (since it is indeed used to determine
      // the accuracy needed for the solution of the linear systems)
      etachoice = KIN_ETACONSTANT;
    }

   kmem = KINCreate();
   info = KINInit(kmem, funcprpr, u);
   if (check_flag(&info, "KINInit", 1, iam)) psb_c_abort(*cctxt);
   info = KINSetNumMaxIters(kmem, newtonmaxit);
   if (check_flag(&info, "KINSetNumMaxIters", 1, iam)) psb_c_abort(*cctxt);
   info = KINSetPrintLevel(kmem, kinsetprintlevel);
   if (check_flag(&info, "KINSetPrintLevel", 2, iam)) psb_c_abort(*cctxt);
   info = KINSetUserData(kmem, &user_data);
   if (check_flag(&info, "KINSetUserData", 1, iam)) psb_c_abort(*cctxt);
   info = KINSetConstraints(kmem, constraints);
   if (check_flag(&info, "KINSetConstraints", 1, iam)) psb_c_abort(*cctxt);
   info = KINSetFuncNormTol(kmem, fnormtol);
   if (check_flag(&info, "KINSetFuncNormTol", 1, iam)) psb_c_abort(*cctxt);
   info = KINSetScaledStepTol(kmem, scsteptol);
   if (check_flag(&info, "KINSetScaledStepTol", 1, iam)) psb_c_abort(*cctxt);
   /* Attach the linear solver to KINSOL and set its options */
   info = KINSetLinearSolver(kmem, LS, J);
   if (check_flag(&info, "KINSetLinearSolver", 1, iam)) psb_c_abort(*cctxt);
   info = KINSetJacFn(kmem,jac);
   if (check_flag(&info, "KINSetJacFn", 1, iam)) psb_c_abort(*cctxt);
   info = KINSetEtaForm(kmem,etachoice);
   if (check_flag(&info, "KINSetEtaForm", 1, iam)) psb_c_abort(*cctxt);
   if ( etachoice == KIN_ETACONSTANT) {
      info = KINSetEtaConstValue(kmem,options.eps);
      if (check_flag(&info, "KINSetEtaConstValue", 1, iam)) psb_c_abort(*cctxt);
   }

   psb_c_barrier(*cctxt);
   if (iam == 0){
     fprintf(stdout, "\n**********************************************************************\n");
     fprintf(stdout, "************ Time Step %d of %d ****************************************\n", 1,NtToDo );
   }
   psb_c_barrier(*cctxt);
   tic = psb_c_wtime();
   info = KINSol(kmem,           /* KINSol memory block */
                 u,              /* initial guess on input; solution vector */
                 globalstrategy, /* global strategy choice */
                 su,             /* scaling vector for the variable u */
                 sc);            /* scaling vector for function values fval */
   psb_c_barrier(*cctxt);
   toc = psb_c_wtime();
   timesolve = toc-tic;

   if(iam == 0){
     PrintFinalStats(kmem,1,&user_data);
     printf("Step solve time: %1.3e\n", timesolve);
     UpdatePerformanceStats(kmem, 1, timesolve, &user_data, &perf_info);
   }

   if (check_flag(&info, "KINSol", 1, iam)){
      if(iam == 0){
         /* Write the performance on log file */
         sprintf(outfilename,"performancerunon_%dprocess_%dx%dx%d_dofs.csv", np,idim,jdim,kdim);
         outfileforperformance = fopen(outfilename,"w+");
         fprintf(outfileforperformance,"Step,Njac,Nfev,NNLin,NLin,GTime\n");
         fprintf(outfileforperformance, "%d,%d,%d,%d,%d,%lf \n",
            0,
            perf_info.numjacobianspertimestep[0],
            perf_info.numfevalspertimestep[0],
            perf_info.numofnonliniterationpertimestep[0],
            perf_info.liniterperstep[0],
            perf_info.timepertimestep[0]);
         fclose(outfileforperformance);
      }
      free(perf_info.numjacobianspertimestep);
      free(perf_info.numfevalspertimestep);
      free(perf_info.liniterperstep);
      free(perf_info.timepertimestep);
      free(perf_info.numofnonliniterationpertimestep);
      N_VDestroy(u);
      N_VDestroy(constraints);
      N_VDestroy(sc);
      N_VDestroy(su);
      SUNMatDestroy(J);
      SUNMatDestroy(user_data.B);
      SUNLinSolFree(LS);
      free(cdh);
      psb_c_abort(*cctxt);
   } else {
     info = 0;
   }

   KINFree(&kmem);

   if(NtToDo > Nt) NtToDo = Nt;

   for(i=2;i<=NtToDo;i++){  // Main Time Loop
     if (iam == 0){
       fprintf(stdout, "************ Time Step %d of %d ****************************************\n", i,NtToDo);
       fflush(stdout);
     }
     user_data.timestep = i;      // used to compute time depending quantities
     user_data.functioncount = 0; // reset counter for function evals
     user_data.jacobiancount = 0; // reset counter for Jacobian evals
     user_data.buildjacobian = false;
     SUNLinSolSetbuildtype_PSBLAS(*(user_data.LS), -1); // Since we do not change the Jacobian we do not build the preconditioner

     /* For Euler Time-Stepping we take note of the old pressure value */
     N_VLinearSum(1.0,u,0.0,user_data.oldpressure,user_data.oldpressure);

     /* We perform the new incomplete Newton time step using as starting point
     the solution at the previous time step.                                  */
     N_VConst(1.0,sc);               // Unweighted norm
     N_VConst(1.0,su);               // Unweighted norm
     kmem = KINCreate();
     info = KINInit(kmem, funcprpr, u);
     if (check_flag(&info, "KINInit", 1, iam)) psb_c_abort(*cctxt);
     info = KINSetNumMaxIters(kmem, newtonmaxit);
     if (check_flag(&info, "KINSetNumMaxIters", 1, iam)) psb_c_abort(*cctxt);
     info = KINSetPrintLevel(kmem, kinsetprintlevel);
     if (check_flag(&info, "KINSetPrintLevel", 2, iam)) psb_c_abort(*cctxt);
     info = KINSetUserData(kmem, &user_data);
     if (check_flag(&info, "KINSetUserData", 1, iam)) psb_c_abort(*cctxt);
     info = KINSetConstraints(kmem, constraints);
     if (check_flag(&info, "KINSetConstraints", 1, iam)) psb_c_abort(*cctxt);
     info = KINSetFuncNormTol(kmem, fnormtol);
     if (check_flag(&info, "KINSetFuncNormTol", 1, iam)) psb_c_abort(*cctxt);
     info = KINSetScaledStepTol(kmem, scsteptol);
     if (check_flag(&info, "KINSetScaledStepTol", 1, iam)) psb_c_abort(*cctxt);
     /* Attach the linear solver to KINSOL and set its options */
     info = KINSetLinearSolver(kmem, LS, J);
     if (check_flag(&info, "KINSetLinearSolver", 1, iam)) psb_c_abort(*cctxt);
     info = KINSetJacFn(kmem,jac);
     if (check_flag(&info, "KINSetJacFn", 1, iam)) psb_c_abort(*cctxt);
     info = KINSetEtaForm(kmem,etachoice);
     if (check_flag(&info, "KINSetEtaForm", 1, iam)) psb_c_abort(*cctxt);
     if ( etachoice == KIN_ETACONSTANT) {
        info = KINSetEtaConstValue(kmem,options.eps);
        if (check_flag(&info, "KINSetEtaConstValue", 1, iam)) psb_c_abort(*cctxt);
     }

     psb_c_barrier(*cctxt);
     tic = psb_c_wtime();
     info = KINSol(kmem,           /* KINSol memory block */
                   u,              /* initial guess on input; solution vector */
                   globalstrategy, /* global strategy choice */
                   su,             /* scaling vector for the variable u */
                   sc);            /* scaling vector for function values fval */
     psb_c_barrier(*cctxt);
     toc = psb_c_wtime();
     timesolve = toc-tic;

     if(iam == 0){
       PrintFinalStats(kmem,i,&user_data);
       printf("Step solve time: %1.3e\n", timesolve);
       UpdatePerformanceStats(kmem, i, timesolve, &user_data, &perf_info);
     }

     if (check_flag(&info, "KINSol", 1, iam)){
        if(iam == 0){
        /* Write the performance on log file */
        sprintf(outfilename,"performancerunon_%dprocess_%dx%dx%d_dofs.csv", np,idim,jdim,kdim);
        outfileforperformance = fopen(outfilename,"w+");
        fprintf(outfileforperformance,"Step,Njac,Nfev,NNLin,NLin,GTime\n");
        for (int j=0; j < i; j++){
           fprintf(outfileforperformance, "%d,%d,%d,%d,%d,%lf \n",
              j,
              perf_info.numjacobianspertimestep[j],
              perf_info.numfevalspertimestep[j],
              perf_info.numofnonliniterationpertimestep[j],
              perf_info.liniterperstep[j],
              perf_info.timepertimestep[j]);
        }
       }
       fclose(outfileforperformance);
       free(perf_info.numjacobianspertimestep);
       free(perf_info.numfevalspertimestep);
       free(perf_info.liniterperstep);
       free(perf_info.timepertimestep);
       free(perf_info.numofnonliniterationpertimestep);
       N_VDestroy(u);
       N_VDestroy(constraints);
       N_VDestroy(sc);
       N_VDestroy(su);
       SUNMatDestroy(J);
       SUNMatDestroy(user_data.B);
       SUNLinSolFree(LS);
       free(cdh);
       psb_c_abort(*cctxt);
     } else {
       info = 0;
     }

     KINFree(&kmem);
   }
   fprintf(stdout, "\n**********************************************************************\n");

   psb_c_barrier(*cctxt);
   if(iam == 0){
      /* Write the performance on log file */
      sprintf(outfilename,"performancerunon_%dprocess_%dx%dx%d_dofs.csv", np,idim,jdim,kdim);
      outfileforperformance = fopen(outfilename,"w+");
      fprintf(outfileforperformance,"Step,Njac,Nfev,NNLin,NLin,GTime\n");
      for (i=0; i < NtToDo; i++){
         fprintf(outfileforperformance, "%d,%d,%d,%d,%d,%lf \n",
            i,
            perf_info.numjacobianspertimestep[i],
            perf_info.numfevalspertimestep[i],
            perf_info.numofnonliniterationpertimestep[i],
            perf_info.liniterperstep[i],
            perf_info.timepertimestep[i]);
      }
      fclose(outfileforperformance);
   }

   /* Free the Memory */

   free(perf_info.numjacobianspertimestep);
   free(perf_info.numfevalspertimestep);
   free(perf_info.liniterperstep);
   free(perf_info.timepertimestep);
   free(perf_info.numofnonliniterationpertimestep);
   N_VDestroy(u);
   N_VDestroy(constraints);
   N_VDestroy(sc);
   N_VDestroy(su);
   SUNMatDestroy(J);
   SUNMatDestroy(user_data.B);
   SUNLinSolFree(LS);
   free(cdh);
   psb_c_barrier(*cctxt);
   psb_c_exit(*cctxt);

   return(info);
}

/*
*--------------------------------------------------------------------
* FUNCTIONS CALLED BY KINSOL
*--------------------------------------------------------------------
*/
static int funcprpr(N_Vector u, N_Vector fval, void *user_data)
{
/* This function returns the evaluation fval = Φ(u;parameters) to march the
   Newton method.                                                             */
   struct user_data_for_f *input = user_data;
   N_Vector uold;
   psb_i_t iam, np, ictxt, idim, jdim, kdim, nl;
   psb_i_t i, k, info;
   psb_l_t glob_row, irow[2];
   double x, y, z, t, deltahx, sqdeltahx, deltahx2;
   double deltahy, sqdeltahy, deltahy2;
   double deltahz, sqdeltahz, deltahz2;
   double val[2],entries[8];
   psb_i_t ix, iy, iz, ijk[3],sizes[3];
   FILE *outfile1,*outfile2;
   char infilename[20],outfilename[20];

   /* Problem parameters */
   psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks, dt, rho, phi, pr;
   psb_d_t xmax, ymax, L;

   /* Load problem parameters */
   thetas = input->thetas;
   thetar = input->thetar;
   alpha  = input->alpha;
   beta   = input->beta;
   gamma  = input->gamma;
   a      = input->a;
   Ks     = input->Ks;
   rho    = input->rho;
   phi    = input->phi;
   uold   = input->oldpressure;
   dt     = input->dt;
   pr     = input->pr;
   xmax   = input->xmax;
   ymax   = input->ymax;
   L      = input->L;

   info = 0;

   idim = input->idim;
   jdim = input->jdim;
   kdim = input->kdim;
   nl = input->nl;

   // Who am I?
   psb_c_info(*(NV_CCTXT_P(u)),&iam,&np);
   psb_c_set_index_base(0);
   info = psb_c_dhalo(NV_PVEC_P(u),NV_DESCRIPTOR_P(u));
   if( info != 0)
      fprintf(stderr, "Halo Error (u) on process %d info: %d\n",iam,info);
   info = psb_c_dhalo(NV_PVEC_P(uold),NV_DESCRIPTOR_P(uold));
   if( info != 0)
      fprintf(stderr, "Halo Error (uold) on process %d info: %d\n",iam,info);

   if ( input->verbosity > 1 ){
      sprintf(infilename,"fin%d_np%d.dat",input->functioncount,iam);
      outfile1 = fopen(infilename,"w+");
      N_VPrintFile_PSBLAS(u,outfile1);
      fclose(outfile1);
   }

   deltahx = (double) xmax/(idim+1.0);
   deltahy = (double) ymax/(jdim+1.0);
   deltahz = (double) L/(kdim+1.0);
   sqdeltahx = deltahx*deltahx;
   sqdeltahy = deltahy*deltahy;
   sqdeltahz = deltahz*deltahz;
   deltahx2  = 2.0* deltahx;
   deltahy2  = 2.0* deltahy;
   deltahz2  = 2.0* deltahz;
   sizes[0] = idim; sizes[1] = jdim; sizes[2] = kdim;

   if (iam == 0 && input->verbosity > 0){
     fprintf(stdout, "----------------------------------------------------------------------\n");
     fprintf(stdout, "Function Evaluation on the parameters:\n");
     fprintf(stdout, "----------------------------------------------------------------------\n");
     fprintf(stdout, "Saturated moisture contents        : %1.3f\n",thetas);
     fprintf(stdout, "Residual moisture contents         : %1.3f\n",thetar);
     fprintf(stdout, "Saturated hydraulic conductivity   : %1.3e\n",Ks);
     fprintf(stdout, "Water density (ρ)                  : %1.3e\n",rho);
     fprintf(stdout, "Porosity of the medium (ϕ)         : %1.3e\n",phi);
     fprintf(stdout, "                                   : (α        ,β    ,a        ,γ    )\n");
     fprintf(stdout, "van Genuchten empirical parameters : (%1.3e,%1.3f,%1.3e,%1.3f)\n",
     alpha,beta,a,gamma);
     fprintf(stdout, "Initial value of the pressure head is %lf cm\n",pr);
     fprintf(stdout, "Solving in a box [0,%lf]x[0,%lf]x[0,%lf]\n",xmax,ymax,L);
     fprintf(stdout, "With %d x %d x %d dofs\n",idim,jdim,kdim);
     fprintf(stdout, "dx = %1.4f dy = %1.4f dz = %1.4f dt = %1.4f\n",
                        deltahx,deltahy,deltahz,dt);
     fprintf(stdout, "----------------------------------------------------------------------\n");
     fflush(stdout);
   }



   for (i=0; i<nl;  i++) {
     glob_row=input->vl[i];                 // Get the index of the global row
     // We compute the local indexes of the elements on the stencil
     psb_c_l_idx2ijk(ijk,glob_row,sizes,3,0);
     ix = ijk[0]; iy = ijk[1]; iz = ijk[2];
     x = (ix+1)*deltahx; y = (iy+1)*deltahy; z = (iz+1)*deltahz; t = (input->timestep)*dt;
     // We compute the result of Φ(p) by first going back to the (i,j,k)
     // indexing and substiting the value of p[i,j,k] on the boundary with the
     // correct values, otherwise we use the entries stored in u, together with
     // the halo values to produce the NVector fval = Φ(p). Another way of doing
     // this would be assembling every time a bunch of temporary matrices
     // with the values of the nonlinear evaluations and doing some
     // matrix-vector products. This way should be faster, and less taxing on
     // the memory.
     entries[0] = psb_c_dgetelem(NV_PVEC_P(uold),glob_row,
                                 NV_DESCRIPTOR_P(uold)); // u^(l-1)_{i,j,k}
     entries[1] = psb_c_dgetelem(NV_PVEC_P(u),glob_row,
                                  NV_DESCRIPTOR_P(u)); // u^(l)_{i,j,k}
     if (ix == 0) {        // Cannot do i-1
       entries[2] = boundary(0.0,y,z,t,user_data); // u^(l)_{i-1,j,k}
     }else{
       ijk[0] = ix - 1; ijk[1] = iy; ijk[2] = iz;
       entries[2] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u)); // u^(l)_{i-1,j,k}
     }
     if (ix == idim -1){
       entries[3] = boundary(L,y,z,t,user_data);
     }else{
       ijk[0] = ix+1; ijk[1] = iy; ijk[2] = iz;
       entries[3] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u));  // u^(l)_{i+1,j,k}
     }
     if (iy == 0){       // Cannot do j-1
       entries[4] = boundary(x,0.0,z,t,user_data); // u^(l)_{i+1,j,k}
     }else{
       ijk[0] = ix; ijk[1] = iy-1; ijk[2] = iz;
       entries[4] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u));  // u^(l)_{i,j-1,k}
     }
     if (iy == jdim -1){
       entries[5] = boundary(x,L,z,t,user_data);
     }else{
       ijk[0] = ix; ijk[1] = iy+1; ijk[2] = iz;
       entries[5] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u));  // u^(l)_{i,j+1,k}
     }
     if (iz == 0){       // Cannot do k-1
       entries[6] = boundary(x,y,0.0,t,user_data);
     }else{
       ijk[0] = ix; ijk[1] = iy; ijk[2] = iz-1;
       entries[6] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u));  // u^(l)_{i,j,k-1}
     }
     if (iz == kdim -1){ // Cannot do k+1
       entries[7] = boundary(x,y,L,t,user_data);
     }else{
       ijk[0] = ix; ijk[1] = iy; ijk[2] = iz+1;
       entries[7] = psb_c_dgetelem(NV_PVEC_P(u),
                                   psb_c_l_ijk2idx(ijk,sizes,3,0),
                                   NV_DESCRIPTOR_P(u));  // u^(l)_{i,j,k+1}
     }
     // We have now recovered all the entries, and we can compute the glob_rowth
     // entry of the funciton
     val[0] = (rho*phi/dt)*(Sfun(entries[1],alpha,beta,thetas,thetar)
       -Sfun(entries[0],alpha,beta,thetas,thetar))
       - upstream(entries[3],entries[1],user_data)*(entries[3]-entries[1])/sqdeltahx
       + upstream(entries[1],entries[2],user_data)*(entries[1]-entries[2])/sqdeltahx
       - upstream(entries[5],entries[1],user_data)*(entries[5]-entries[1])/sqdeltahy
       + upstream(entries[1],entries[4],user_data)*(entries[1]-entries[4])/sqdeltahy
       - upstream(entries[7],entries[1],user_data)*(entries[7]-entries[1])/sqdeltahz
       + upstream(entries[1],entries[6],user_data)*(entries[1]-entries[6])/sqdeltahz
       + ( Kfun(entries[6],a,gamma,Ks) - Kfun(entries[7],a,gamma,Ks) )/deltahz2;


     irow[0] = glob_row;
     if(psb_c_dgeins(1,irow,val,NV_PVEC_P(fval),NV_DESCRIPTOR_P(fval))!=0)
      fprintf(stderr,"Error from geins on process %d info is %d\n",iam,info);
   }

   psb_c_check_error(*(NV_CCTXT_P(u)));
   // We assemble the vector at the end
   if(psb_c_dgeasb(NV_PVEC_P(fval),NV_DESCRIPTOR_P(fval))!=0)
      fprintf(stderr,"Process %d Completed Vector Assembly with info = %d!\n",iam,info);

   if ( input->verbosity > 1 ){
      printf("I am %d and I'm writing out the vector!\n",iam);
      sprintf(outfilename,"fout%d_np%d.dat",input->functioncount,iam);
      outfile2 = fopen(outfilename,"w+");
      N_VPrintFile_PSBLAS(fval,outfile2);
      fclose(outfile2);
   }

   input->functioncount += 1;

  return(info);
}

static int jac(N_Vector yvec, N_Vector fvec, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2)
{
  /* This function returns the evaluation of the Jacobian of the system upon
  request of the Newton method.                                               */
  struct user_data_for_f *input = user_data;
  psb_i_t iam, np, ictxt, idim, jdim, kdim, nl;
  psb_i_t i, k, info, el;
  psb_l_t glob_row, irow[10*NBMAX], icol[10*NBMAX];
  double x, y, z, t, deltahx, sqdeltahx, deltahx2;
  double deltahy, sqdeltahy, deltahy2;
  double deltahz, sqdeltahz, deltahz2;

  double val[10*NBMAX],entries[8];
  psb_i_t ix, iy, iz, ijk[3],ijkinsert[3],sizes[3];

  /* Problem parameters */
  psb_d_t thetas, thetar, alpha, beta, a, gamma, Ks, dt, rho, phi, pr;
  psb_d_t xmax, ymax, L;
  /* Timings */
  psb_d_t tic, toc, timecdh;
  char filename[100];

  /* Load problem parameters */
  thetas = input->thetas;
  thetar = input->thetar;
  alpha  = input->alpha;
  beta   = input->beta;
  gamma  = input->gamma;
  a      = input->a;
  Ks     = input->Ks;
  dt     = input->dt;
  rho    = input->rho;
  phi    = input->phi;
  pr     = input->pr;
  xmax   = input->xmax;
  ymax   = input->ymax;
  L      = input->L;

  info = 0;

  idim = input->idim;
  jdim = input->jdim;
  kdim = input->kdim;
  nl = input->nl;

  // Who am I?
  psb_c_info(*(NV_CCTXT_P(yvec)),&iam,&np);
  psb_c_set_index_base(0);
  info = psb_c_dhalo(NV_PVEC_P(yvec),NV_DESCRIPTOR_P(yvec));
  if( info != 0)
     fprintf(stderr, "Halo Error (yvec) on process %d info: %d\n",iam,info);

  // Check if I have to build a Jacobian
  if( !input->buildjacobian ){
     input->buildjacobian = true;
     return(info);
  }

  if (iam == 0 && input->verbosity > 0){
    fprintf(stdout, "----------------------------------------------------------------------\n");
    fprintf(stdout, "Jacobian Evaluation on the parameters:\n");
    fprintf(stdout, "----------------------------------------------------------------------\n");
    fprintf(stdout, "Saturated moisture contents        : %1.3f\n",thetas);
    fprintf(stdout, "Residual moisture contents         : %1.3f\n",thetar);
    fprintf(stdout, "Saturated hydraulic conductivity   : %1.3e\n",Ks);
    fprintf(stdout, "Water density (ρ)                  : %1.3e\n",rho);
    fprintf(stdout, "Porosity of the medium (ϕ)         : %1.3e\n",phi);
    fprintf(stdout, "                                   : (α        ,β    ,a        ,γ    )\n");
    fprintf(stdout, "van Genuchten empirical parameters : (%1.3e,%1.3f,%1.3e,%1.3f)\n",
    alpha,beta,a,gamma);
    fprintf(stdout, "Initial value of the pressure head is %lf cm\n",pr);
    fprintf(stdout, "Solving in a box [0,%lf]x[0,%lf]x[0,%lf]\n",xmax,ymax,L);
    fprintf(stdout, "----------------------------------------------------------------------\n");
    fflush(stdout);
  }

  /*---------------------------------------------------------------------------*
   * First of all we make sure that the Jacobian matrix is in assembly state   *
   * and that the prescribred entries can be overwritten.                      *
   ----------------------------------------------------------------------------*/
  SUNMatZero_PSBLAS(J);

  deltahx = (double) xmax/(idim+1.0);
  deltahy = (double) ymax/(jdim+1.0);
  deltahz = (double) L/(kdim+1.0);
  sqdeltahx = deltahx*deltahx;
  sqdeltahy = deltahy*deltahy;
  sqdeltahz = deltahz*deltahz;
  deltahx2  = 2.0* deltahx;
  deltahy2  = 2.0* deltahy;
  deltahz2  = 2.0* deltahz;
  sizes[0] = idim; sizes[1] = jdim; sizes[2] = kdim;
  t = (input->timestep)*dt;

  for (i=0; i<nl;  i++) {
    glob_row=input->vl[i];                 // Get the index of the global row
    // We compute the local indexes of the elements on the stencil
    psb_c_l_idx2ijk(ijk,glob_row,sizes,3,0);
    ix = ijk[0]; iy = ijk[1]; iz = ijk[2];
    x = (ix+1)*deltahx; y = (iy+1)*deltahy; z = (iz+1)*deltahz;
    el = 0;

    /* To compute the expression of the Jacobian for Φ we first need to access
    the entries for the current iterate, these are contained in the N_Vector
    yvec. Another way of doing this would be assembling every time a bunch of
    temporary matrices with the values of the nonlinear evaluations and doing
    some matrix-matrix operationss. This way should be faster, and less taxing
    on the memory.                                                            */
    entries[1] = psb_c_dgetelem(NV_PVEC_P(yvec),glob_row,
                                 NV_DESCRIPTOR_P(yvec)); // u^(l)_{i,j,k}
    if (ix == 0) {        // Cannot do i-1
      entries[2] = boundary(0.0,y,z,t,user_data); // u^(l)_{i-1,j,k}
    }else{
      ijk[0] = ix - 1; ijk[1] = iy; ijk[2] = iz;
      entries[2] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec)); // u^(l)_{i-1,j,k}
    }
    if (ix == idim -1){
      entries[3] = boundary(L,y,z,t,user_data);
    }else{
      ijk[0] = ix+1; ijk[1] = iy; ijk[2] = iz;
      entries[3] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec));  // u^(l)_{i+1,j,k}
    }
    if (iy == 0){       // Cannot do j-1
      entries[4] = boundary(x,0.0,z,t,user_data); // u^(l)_{i+1,j,k}
    }else{
      ijk[0] = ix; ijk[1] = iy-1; ijk[2] = iz;
      entries[4] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j-1,k}
    }
    if (iy == jdim -1){
      entries[5] = boundary(x,L,z,t,user_data);
    }else{
      ijk[0] = ix; ijk[1] = iy+1; ijk[2] = iz;
      entries[5] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j+1,k}
    }
    if (iz == 0){       // Cannot do k-1
      entries[6] = boundary(x,y,0.0,t,user_data);
    }else{
      ijk[0] = ix; ijk[1] = iy; ijk[2] = iz-1;
      entries[6] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j,k-1}
    }
    if (iz == kdim -1){ // Cannot do k+1
      entries[7] = boundary(x,y,L,t,user_data);
    }else{
      ijk[0] = ix; ijk[1] = iy; ijk[2] = iz+1;
      entries[7] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                  psb_c_l_ijk2idx(ijk,sizes,3,0),
                                  NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j,k+1}
    }

    /* Now that we have all the entries of yvec needed to compute the entries
    of the current row of the Jacobian we can loop through the different nonzero
    elements and compute the relative values.                                 */
    /*  term depending on   (i-1,j,k)        */
    val[el] = -(upstream(entries[1],entries[2],user_data)/sqdeltahx) +
   ((entries[1] - entries[2])*upstream01(entries[1],entries[2],user_data))/sqdeltahx;
    if(ix != 0){
      ijkinsert[0]=ix-1; ijkinsert[1]=iy; ijkinsert[2]=iz;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    /*  term depending on     (i,j-1,k)        */
    val[el] = -(upstream(entries[1],entries[4],user_data)/sqdeltahy) +
   ((entries[1] - entries[4])*upstream01(entries[1],entries[4],user_data))/sqdeltahy;
    if (iy != 0){
      ijkinsert[0]=ix; ijkinsert[1]=iy-1; ijkinsert[2]=iz;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    /* term depending on      (i,j,k-1)        */
    val[el] = -(upstream(entries[1],entries[6],user_data)/sqdeltahz) +
   ((entries[1] - entries[6])*upstream01(entries[1],entries[6],user_data))/sqdeltahz +
   Kfunprime(entries[6],alpha,gamma,Ks)/deltahz2;
       if (iz != 0){
      ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz-1;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    /* term depending on      (i,j,k)          */
    val[el] = upstream(entries[1],entries[2],user_data)/sqdeltahx +
      upstream(entries[1],entries[4],user_data)/sqdeltahy +
      upstream(entries[1],entries[6],user_data)/sqdeltahz +
      upstream(entries[3],entries[1],user_data)/sqdeltahx +
      upstream(entries[5],entries[1],user_data)/sqdeltahy +
      upstream(entries[7],entries[1],user_data)/sqdeltahz -
      ((-entries[1] + entries[3])*upstream01(entries[3],entries[1],user_data))/sqdeltahx -
      ((-entries[1] + entries[5])*upstream01(entries[5],entries[1],user_data))/sqdeltahy -
      ((-entries[1] + entries[7])*upstream01(entries[7],entries[1],user_data))/sqdeltahz +
      ((entries[1] - entries[2])*upstream10(entries[1],entries[2],user_data))/sqdeltahx +
      ((entries[1] - entries[4])*upstream10(entries[1],entries[4],user_data))/sqdeltahy +
      ((entries[1] - entries[6])*upstream10(entries[1],entries[6],user_data))/sqdeltahz +
      (phi*rho*Sfunprime(entries[1],alpha,beta,thetas,thetar))/dt;
    ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz;
    icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
    el=el+1;
    /*  term depending on     (i+1,j,k)        */
    val[el] = -(upstream(entries[3],entries[1],user_data)/sqdeltahx) -
   ((-entries[1] + entries[3])*upstream10(entries[3],entries[1],user_data))/sqdeltahx;
    if (ix != idim-1) {
      ijkinsert[0]=ix+1; ijkinsert[1]=iy; ijkinsert[2]=iz;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    /*  term depending on     (i,j+1,k)        */
    val[el] = -(upstream(entries[5],entries[1],user_data)/sqdeltahy) -
      ((-entries[1] + entries[5])*upstream10(entries[5],entries[1],user_data))/sqdeltahy;
       if (iy != idim-1){
      ijkinsert[0]=ix; ijkinsert[1]=iy+1; ijkinsert[2]=iz;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    /* term depending on      (i,j,k+1)        */
    val[el] = -(upstream(entries[7],entries[1],user_data)/sqdeltahz) -
   ((-entries[1] + entries[7])*upstream10(entries[7],entries[1],user_data))/sqdeltahz -
   Kfunprime(entries[7],a,gamma,Ks)/deltahz2;
    if (iz != idim-1){
      ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz+1;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
    }
    for (k=0; k<el; k++) irow[k]=glob_row;
    /* Insert the local portion into the Jacobian */
    if ((info=psb_c_dspins(el,irow,icol,val,SM_PMAT_P(J),SM_DESCRIPTOR_P(J)))!=0)
     fprintf(stderr,"From psb_c_dspins: %d\n",info);
  }

  // Assemble and return
  tic = psb_c_wtime();
  if ((info=psb_c_dspasb(SM_PMAT_P(J),SM_DESCRIPTOR_P(J)))!=0)  return(info);
  toc = psb_c_wtime();
  if (iam == 0) fprintf(stdout, "Buit new Jacobian in %lf s\n",toc-tic);

  if (input->verbosity > 1){
    sprintf(filename,"Jacobian_t%d_it%d_np%d.mtx",input->timestep,input->jacobiancount,iam);
    SUNPSBLASMatrix_Print(J,"Jacobian",filename);
  }
  input->jacobiancount += 1;

  if (input->jacobiancount == 2){
   // We deactivate the build of the preconditioner
   SUNLinSolSetbuildtype_PSBLAS(*(input->LS), 1);
  }

  // We build the symmetric approximation of the Jacobian on which we build the
  // preconditioner
  if (BUILD_AUXILIARY_MATRIX){

     printf("Start building auxiliary matrix\n");

     SUNMatZero_PSBLAS(input->B);
     for (i=0; i<nl;  i++) {
      glob_row=input->vl[i];                 // Get the index of the global row
      // We compute the local indexes of the elements on the stencil
      psb_c_l_idx2ijk(ijk,glob_row,sizes,3,0);
      ix = ijk[0]; iy = ijk[1]; iz = ijk[2];
      x = (ix+1)*deltahx; y = (iy+1)*deltahy; z = (iz+1)*deltahz;
      el = 0;

      /* To compute the expression of the Jacobian for Φ we first need to access
      the entries for the current iterate, these are contained in the N_Vector
      yvec. Another way of doing this would be assembling every time a bunch of
      temporary matrices with the values of the nonlinear evaluations and doing
      some matrix-matrix operationss. This way should be faster, and less taxing
      on the memory.                                                            */
      entries[1] = psb_c_dgetelem(NV_PVEC_P(yvec),glob_row,
                                    NV_DESCRIPTOR_P(yvec)); // u^(l)_{i,j,k}
      if (ix == 0) {        // Cannot do i-1
         entries[2] = boundary(0.0,y,z,t,user_data); // u^(l)_{i-1,j,k}
      }else{
         ijk[0] = ix - 1; ijk[1] = iy; ijk[2] = iz;
         entries[2] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                     psb_c_l_ijk2idx(ijk,sizes,3,0),
                                     NV_DESCRIPTOR_P(yvec)); // u^(l)_{i-1,j,k}
      }
      if (ix == idim -1){
         entries[3] = boundary(L,y,z,t,user_data);
      }else{
         ijk[0] = ix+1; ijk[1] = iy; ijk[2] = iz;
         entries[3] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                     psb_c_l_ijk2idx(ijk,sizes,3,0),
                                     NV_DESCRIPTOR_P(yvec));  // u^(l)_{i+1,j,k}
      }
      if (iy == 0){       // Cannot do j-1
         entries[4] = boundary(x,0.0,z,t,user_data); // u^(l)_{i+1,j,k}
      }else{
         ijk[0] = ix; ijk[1] = iy-1; ijk[2] = iz;
         entries[4] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                     psb_c_l_ijk2idx(ijk,sizes,3,0),
                                     NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j-1,k}
      }
      if (iy == jdim -1){
         entries[5] = boundary(x,L,z,t,user_data);
      }else{
         ijk[0] = ix; ijk[1] = iy+1; ijk[2] = iz;
         entries[5] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                     psb_c_l_ijk2idx(ijk,sizes,3,0),
                                     NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j+1,k}
      }
      if (iz == 0){       // Cannot do k-1
         entries[6] = boundary(x,y,0.0,t,user_data);
      }else{
         ijk[0] = ix; ijk[1] = iy; ijk[2] = iz-1;
         entries[6] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                     psb_c_l_ijk2idx(ijk,sizes,3,0),
                                     NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j,k-1}
      }
      if (iz == kdim -1){ // Cannot do k+1
         entries[7] = boundary(x,y,L,t,user_data);
      }else{
         ijk[0] = ix; ijk[1] = iy; ijk[2] = iz+1;
         entries[7] = psb_c_dgetelem(NV_PVEC_P(yvec),
                                     psb_c_l_ijk2idx(ijk,sizes,3,0),
                                     NV_DESCRIPTOR_P(yvec));  // u^(l)_{i,j,k+1}
      }

      /* Now that we have all the entries of yvec needed to compute the entries
      of the current row of the Jacobian we can loop through the different nonzero
      elements and compute the relative values.                                 */
      /*  term depending on   (i-1,j,k)        */
      val[el] = -(upstream(entries[1],entries[2],user_data)/sqdeltahx) +
     ((entries[1] - entries[2])*upstream01(entries[1],entries[2],user_data))/sqdeltahx;
      if(ix != 0){
         ijkinsert[0]=ix-1; ijkinsert[1]=iy; ijkinsert[2]=iz;
         icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
         el=el+1;
      }
      /*  term depending on     (i,j-1,k)        */
      val[el] = -(upstream(entries[1],entries[4],user_data)/sqdeltahy) +
     ((entries[1] - entries[4])*upstream01(entries[1],entries[4],user_data))/sqdeltahy;
      if (iy != 0){
         ijkinsert[0]=ix; ijkinsert[1]=iy-1; ijkinsert[2]=iz;
         icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
         el=el+1;
      }
      /* term depending on      (i,j,k-1)        */
      val[el] = -(upstream(entries[1],entries[6],user_data)/sqdeltahz) +
     ((entries[1] - entries[6])*upstream01(entries[1],entries[6],user_data))/sqdeltahz;
          if (iz != 0){
         ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz-1;
         icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
         el=el+1;
      }
      /* term depending on      (i,j,k)          */
      val[el] = upstream(entries[1],entries[2],user_data)/sqdeltahx +
         upstream(entries[1],entries[4],user_data)/sqdeltahy +
         upstream(entries[1],entries[6],user_data)/sqdeltahz +
         upstream(entries[3],entries[1],user_data)/sqdeltahx +
         upstream(entries[5],entries[1],user_data)/sqdeltahy +
         upstream(entries[7],entries[1],user_data)/sqdeltahz -
         ((-entries[1] + entries[3])*upstream01(entries[3],entries[1],user_data))/sqdeltahx -
         ((-entries[1] + entries[5])*upstream01(entries[5],entries[1],user_data))/sqdeltahy -
         ((-entries[1] + entries[7])*upstream01(entries[7],entries[1],user_data))/sqdeltahz +
         ((entries[1] - entries[2])*upstream10(entries[1],entries[2],user_data))/sqdeltahx +
         ((entries[1] - entries[4])*upstream10(entries[1],entries[4],user_data))/sqdeltahy +
         ((entries[1] - entries[6])*upstream10(entries[1],entries[6],user_data))/sqdeltahz +
         (phi*rho*Sfunprime(entries[1],alpha,beta,thetas,thetar))/dt;
      ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz;
      icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
      el=el+1;
      /*  term depending on     (i+1,j,k)        */
      val[el] = -(upstream(entries[3],entries[1],user_data)/sqdeltahx) -
     ((-entries[1] + entries[3])*upstream10(entries[3],entries[1],user_data))/sqdeltahx;
      if (ix != idim-1) {
         ijkinsert[0]=ix+1; ijkinsert[1]=iy; ijkinsert[2]=iz;
         icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
         el=el+1;
      }
      /*  term depending on     (i,j+1,k)        */
      val[el] = -(upstream(entries[5],entries[1],user_data)/sqdeltahy) -
         ((-entries[1] + entries[5])*upstream10(entries[5],entries[1],user_data))/sqdeltahy;
          if (iy != idim-1){
         ijkinsert[0]=ix; ijkinsert[1]=iy+1; ijkinsert[2]=iz;
         icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
         el=el+1;
      }
      /* term depending on      (i,j,k+1)        */
      val[el] = -(upstream(entries[7],entries[1],user_data)/sqdeltahz) -
     ((-entries[1] + entries[7])*upstream10(entries[7],entries[1],user_data))/sqdeltahz;
      if (iz != idim-1){
         ijkinsert[0]=ix; ijkinsert[1]=iy; ijkinsert[2]=iz+1;
         icol[el] = psb_c_l_ijk2idx(ijkinsert,sizes,3,0);
         el=el+1;
      }
      for (k=0; k<el; k++) irow[k]=glob_row;
      /* Insert the local portion into the Jacobian */
      if ((info=psb_c_dspins(el,irow,icol,val,SM_PMAT_P(input->B),SM_DESCRIPTOR_P(input->B)))!=0)
        fprintf(stderr,"From psb_c_dspins: %d\n",info);
     }

     tic = psb_c_wtime();
     if ((info=psb_c_dspasb(SM_PMAT_P(input->B),SM_DESCRIPTOR_P(input->B)))!=0)  return(info);
     toc = psb_c_wtime();
     if (iam == 0) fprintf(stdout, "Buit auxiliary matrix in %lf s\n",toc-tic);
     // We say to the linear solver on what matrix he has to compute the
     // preconditioner
     SUNLinSolSetPreconditioner_PSBLAS(*(input->LS),input->B);
  } else {
    SUNLinSolSetPreconditioner_PSBLAS(*(input->LS),J);
  }

  return(info);
}

static double Sfun(double p, double alpha, double beta, double thetas,
                    double thetar){

  double s = 0.0;

  s = alpha*(thetas-thetar)/(alpha + pow(SUNRabs(p),beta)) + thetar;

  return(s);
}
static double Kfun(double p, double a, double gamma, double Ks){

  double K = 0.0;

  K = Ks*a/(a + pow(SUNRabs(p),gamma));

  return(K);

}
static double Sfunprime(double p, double alpha, double beta, double thetas,
                    double thetar){

  double s = 0.0;

  s = -alpha*beta*pow(SUNRabs(p),beta-1)*sgn(p)*(thetas-thetar)
            /pow(alpha + pow(SUNRabs(p),beta),2);

  return(s);

}
static double Kfunprime(double p, double a, double gamma, double Ks){

  double K = 0.0;

  K = -Ks*a*gamma*pow(SUNRabs(p),gamma-1)*sgn(p)
          /pow(a + pow(SUNRabs(p),gamma),2);

  return(K);
}

static double sgn(double x){
  // Can this be done in a better way?
  return((x > 0) ? 1 : ((x < 0) ? -1 : 0));
}

static double source(double x, double y, double z, double t){
  // Source term function
  return(0.0);
}

static double boundary(double x, double y, double z, double t, void *user_data){
  /* Dirichlet boundary, (possibly) time dependent */
  /* Water is applied at z=L such that the pressure head becomes zero
     in the region, and p = pr on the all the remaining boundaries.*/
  struct user_data_for_f *input = user_data;
  psb_d_t pr, xmax, ymax, L;
  psb_d_t res;

  pr     = input->pr;
  xmax   = input->xmax;
  ymax   = input->ymax;
  L      = input->L;

  if( (x >= xmax/4.0) && (x <= 3.0*xmax/4.0)
   && (y >= ymax/4.0) && (y <= 3.0*ymax/4.0) && z == L){
      res = 0.0;
   }
   else{
      res = pr;
   }

  return(res);
}

static double upstream(double pL, double pU, void *user_data){
  /* Upstream mean for the Ks function */
  struct user_data_for_f *input = user_data;
  double a, gamma, Ks;
  double res;

  a      = input->a;
  gamma  = input->gamma;
  Ks     = input->Ks;

  if(pU-pL >= 0.0){
    res = Kfun(pU,a,gamma,Ks);
  }else{
    res = Kfun(pL,a,gamma,Ks);
  }

  return(res);
}

static double upstream10(double pL, double pU, void *user_data){
   /* Derivative (1,0) of upstream mean for the Ks function */
   struct user_data_for_f *input = user_data;
   double a, gamma, Ks;
   double res;

   a      = input->a;
   gamma  = input->gamma;
   Ks     = input->Ks;

   res = 0.0;

   if ((pL-pU) > 0){
      res = Kfunprime(pL, a,gamma,Ks);
   }

   return(res);

}

static double upstream01(double pL, double pU, void *user_data){
   /* Derivative (0,1) of upstream mean for the Ks function */
   struct user_data_for_f *input = user_data;
   double a, gamma, Ks;
   double res;

   a      = input->a;
   gamma  = input->gamma;
   Ks     = input->Ks;

   res = 0.0;

   if ((pL-pU) <= 0){
      res = Kfunprime(pU, a,gamma,Ks);
   }

   return(res);
}

static double chi(double pL, double pU){
  if(pU - pL >= 0.0){
    return(1.0);
  }else{
    return(0.0);
  }
}

/*--------------------------------------------------------
 * KINSOL FLAG AND OUTPUT ROUTINES
 *-------------------------------------------------------*/

static int check_flag(void *flagvalue, const char *funcname, int opt, int id)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr,
            "\nSUNDIALS_ERROR(%d): %s() failed - returned NULL pointer\n\n",
	    id, funcname);
    return(1);
  }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr,
              "\nSUNDIALS_ERROR(%d): %s() failed with flag = %d\n\n",
	      id, funcname, *errflag);
      return(1);
    }
  }

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr,
            "\nMEMORY_ERROR(%d): %s() failed - returned NULL pointer\n\n",
	    id, funcname);
    return(1);
  }

  return(0);
}

static void PrintFinalStats(void *kmem, int i, void *user_data)
{
  long int nni, nfe, nli, npe, nps, ncfl, nfeSG;
  struct user_data_for_f *input = user_data;
  psb_d_t funcnorm;
  int flag;

  // Number of nonlinear solver iteration
  flag = KINGetNumNonlinSolvIters(kmem, &nni);
  check_flag(&flag, "KINGetNumNonlinSolvIters", 1, 0);
  // Number of function evaluations
  flag = KINGetNumFuncEvals(kmem, &nfe);
  check_flag(&flag, "KINGetNumFuncEvals", 1, 0);
  // Number of linear iterations
  flag = KINGetNumLinIters(kmem, &nli);
  check_flag(&flag, "KINGetNumLinIters", 1, 0);
  // Number of linear convergence failures
  flag = KINGetNumLinConvFails(kmem, &ncfl);
  check_flag(&flag, "KINGetNumLinConvFails", 1, 0);
  // Number of linear-function evaluation (in our case this should always be zero)
  flag = KINGetNumLinFuncEvals(kmem, &nfeSG);
  check_flag(&flag, "KINGetNumLinFuncEvals", 1, 0);
  // Last function norm
  flag = KINGetFuncNorm(kmem, &funcnorm);
  check_flag(&flag, "KINGetFuncNorm", 1, 0);

  printf("\n\nFinal Statistics for Time Step %d\n",i);
  printf("nni    = %5ld    nli   = %5ld\n", nni, nli);
  printf("nfe    = %5ld    nfeSG = %5ld\n", nfe, nfeSG);
  printf("njac   = %5d    ncfl  = %5ld\n", input->jacobiancount, ncfl);
  printf("fnorm  = %1.5e\n",funcnorm);
}

static void UpdatePerformanceStats(void *kmem, int i, psb_d_t timeofthestep,
      void *user_data, void *perf_info){
   long int nni, nfe, nli, npe, nps, ncfl, nfeSG;
   struct user_data_for_f *input = user_data;
   struct performanceinfo *pinfo = perf_info;
   psb_d_t funcnorm;
   int flag;

   // Number of nonlinear solver iteration
   flag = KINGetNumNonlinSolvIters(kmem, &nni);
   check_flag(&flag, "KINGetNumNonlinSolvIters", 1, 0);
   // Number of function evaluations
   flag = KINGetNumFuncEvals(kmem, &nfe);
   check_flag(&flag, "KINGetNumFuncEvals", 1, 0);
   // Number of linear iterations
   flag = KINGetNumLinIters(kmem, &nli);
   check_flag(&flag, "KINGetNumLinIters", 1, 0);
   // Number of linear convergence failures
   flag = KINGetNumLinConvFails(kmem, &ncfl);
   check_flag(&flag, "KINGetNumLinConvFails", 1, 0);
   // Number of linear-function evaluation (in our case this should always be zero)
   flag = KINGetNumLinFuncEvals(kmem, &nfeSG);
   check_flag(&flag, "KINGetNumLinFuncEvals", 1, 0);
   // Last function norm
   flag = KINGetFuncNorm(kmem, &funcnorm);
   check_flag(&flag, "KINGetFuncNorm", 1, 0);

   pinfo->numjacobianspertimestep[i-1] = input->jacobiancount;
   pinfo->numfevalspertimestep[i-1] = nfe;
   pinfo->timepertimestep[i-1] = timeofthestep;
   pinfo->liniterperstep[i-1] = nli;
   pinfo->numofnonliniterationpertimestep[i-1] = nni;

}
