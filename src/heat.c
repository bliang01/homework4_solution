#include <stdlib.h>
#include <mpi.h>

/*
  array_copy

  Copies the contents of one array to another. Note that this is "stronger" than
  just having ```toarray` point to the same location as `fromarray`.

  Parameters
  ----------
  toarray : double*
      Array to copy into.
  fromarray : double*
      Array to copy the contents from into `toarray`.
  length : size_t

  Returns
  -------
  None
      `toarray` is modified in-place.
*/
void array_copy(double* toarray, double* fromarray, size_t length)
{
  for (size_t i=0; i<length; ++i)
    toarray[i] = fromarray[i];
}

/*
  heat_serial

  See include/heat.h for documentation.
*/
void heat_serial(double* u, double dx, size_t Nx, double dt, size_t Nt)
{
  // allocate temporary space for the time stepping routine
  double* ut = (double*) malloc(Nx * sizeof(double));
  double* utp1 = (double*) malloc(Nx * sizeof(double));
  double* temp;

  // copy contents of `u` to `ut`
  array_copy(ut, u, Nx);

  // determine the numerical diffusion coefficient
  double nu = dt/(dx*dx);

  // time step `Nt` times with step size `dt`
  for (size_t step=0; step<Nt; ++step)
    {
      // update using Forward Euler
      for (size_t i=1; i<(Nx-1); ++i)
        utp1[i] = ut[i] + nu*(ut[i-1] - 2*ut[i] + ut[i+1]);

      // handle periodic boundary conditions: that is, assume a "wrap-around"
      // from the left edge of the boundary to the right edge
      utp1[0] = ut[0] + nu*(ut[Nx-1] - 2*ut[0] + ut[1]);
      utp1[Nx-1] = ut[Nx-1] + nu*(ut[Nx-2] - 2*ut[Nx-1] + ut[0]);

      // iterate: here we use some pointer arithmetic trickery to make `ut`
      // point to the newly-generated data and make `utp1` point to the old data
      // (which we will write over in the next iteration)
      //
      // again: `ut`, `utp1`, and `temp` are just pointers to location in
      // memory. this is a really cheap way to "swap" array data. though,
      // depending on what you're doing you need to be careful.
      temp = ut;
      ut = utp1;
      utp1 = temp;
    }

  // copy the results (located at `ut`) to `u`. that is, the location in memory
  // pointed to by `u` now contains the solution data.
  array_copy(u, ut, Nx);

  // free allocations (make sure the contents of `ut` are copied to `u` before
  // freeing these allocations!)
  free(ut);
  free(utp1);
}

/*
  heat_parallel
 */
void heat_parallel(double* uk, double dx, size_t Nx, double dt, size_t Nt,
                   MPI_Comm comm)
{
  // get information about the MPI environment and create spaces for Status and
  // Request information (if needed)
  MPI_Request req;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // allocate temporary space for this process's chunk of the heat equation
  // calculation.
  double* ukt = (double*) malloc(Nx * sizeof(double));
  double* uktp1 = (double*) malloc(Nx * sizeof(double));
  double* temp;

  // copy contents of the input initial chunk data to ut for iteration
  array_copy(ukt, uk, Nx);

  // set the numerical diffusion coefficient
  double nu = dt/(dx*dx);

  /*
    REMOVE THIS BEFORE POSING HOMEWORK
  */
  double left_ghost[2];
  double right_ghost[2];
  int left_proc = (rank-1+size) % size;
  int right_proc = (rank+1) % size;
  for (size_t step=0; step<Nt; ++step)
    {
      /*
        Your tasks:

        1) Communicate necessary boundary data

        2) Solve the heat equation using the Forward Euler method in this part
        of the domain.

       */
      // set and communicate boundary data
      left_ghost[0] = ukt[0];
      right_ghost[0] = ukt[Nx-1];

      //printf("[%d->%d] left_ghost->right_ghost\n", rank, left_proc);
      MPI_Send(&left_ghost[0], 1, MPI_DOUBLE, left_proc, 0, comm);
      MPI_Recv(&right_ghost[1], 1, MPI_DOUBLE, right_proc, 0, comm,
               MPI_STATUS_IGNORE);

      //printf("[%d->%d] right_ghost->left_ghost\n", rank, right_proc);
      MPI_Isend(&right_ghost[0], 1, MPI_DOUBLE, right_proc, 0, comm, &req);
      MPI_Recv(&left_ghost[1], 1, MPI_DOUBLE, left_proc, 0, comm,
               MPI_STATUS_IGNORE);

      // update local data using Forward Euler
      //printf("[%d] forward euler\n", rank);
      for (size_t i=1; i<(Nx-1); ++i)
        uktp1[i] = ukt[i] + nu*(ukt[i-1] - 2*ukt[i] + ukt[i+1]);
      uktp1[0] = ukt[0] + nu*(left_ghost[1] - 2*ukt[0] + ukt[1]);
      uktp1[Nx-1] = ukt[Nx-1] + nu*(ukt[Nx-2] - 2*ukt[Nx-1] + right_ghost[1]);

      // local iterate
      temp = ukt;
      ukt = uktp1;
      uktp1 = temp;
      MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
  /*
    END REMOVAL ZONE
   */

  // copy contents of solution, stored in `ut` to the input chunk `uk`. that is,
  // the location pointed to by `uk` now contains the data after iteration.
  array_copy(uk, ukt, Nx);

  // free temporary allocations
  free(ukt);
  free(uktp1);
}
