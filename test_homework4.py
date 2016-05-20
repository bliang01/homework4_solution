import unittest
import numpy
import matplotlib.pyplot as plt

from numpy import array, linspace, pi, sin, exp, zeros, ones, double
from numpy.linalg import norm
from scipy.integrate import trapz, simps

from mpi4py import MPI

from homework4 import (
    heat_serial,
    heat_parallel,
)


class TestHeat(unittest.TestCase):
    r""" The test is written for you. Read closely to see what it going on. Put on
    your parallel hats, beacuse in fact this test is executed by every process
    you spawn!

    (However, only Process 0 is doing any actual comparison of results.)

    """

    def test_heat_parallel(self):
        # This test compares the parallel solution to the serial solution.
        #
        # Each process executes this test. However, Process 0 will takes the
        # results of every Process, which is a chunk of the heat equation
        # solution, and construct the full parallel solution.
        #
        # Process 0 will also compute the serial solution using the
        # `heat_serial` function. The two solutions are compared by computing
        # the norms of the restults.
        #
        # The initial condition u0 of the test is:
        #
        # u0 = 1,  for the first N/comm.size elements
        # u0 = 0,  elsewhere
        #

        comm = MPI.COMM_WORLD
        print 'executed by Process %d'%(comm.rank)

        # problem parameters
        N = 96               # number x,u-points in entire domain
        Nx = N // comm.size  # number x,u-points in each process's chunk
        dx = 1.0/(N+1)       # step size in x-domain
        dt = 0.4*dx**2       # must be <= 0.5*dx**2 or else F.E. is unstable
        Nt = 100             # number of dt-sized time steps to take

        # compute the full parallel solution. u_parallel = None except for rank
        # 0's version, which is the full solution
        u_parallel = example_parallel_solution(dx, N, dt, Nt, comm)

        # create initial condition
        if (comm.rank == 0):
            u_serial = numpy.zeros(N, dtype=double)
            u_serial[:Nx] = 1

            # solve in serial
            u_serial = heat_serial(u_serial, dx, N, dt, Nt)

            # compare solutions
            error = norm(u_serial - u_parallel)
            self.assertLess(error, 1e-8,
                            msg='The parallel heat equation solution is not '
                                'equal to the serial solution')
        else:
            self.assertTrue(True)


def example_parallel_solution(dx, N, dt, Nt, comm):
    r"""
    Given some input parameters, return the full parallel solution.

    The function `heat_parallel()` is meant to be executed by each individual
    process. This function takes in some problem parameters and an MPI
    Communicator and has Process 0 return the full-domain solution by piecing
    together the components computed in parallel.

    The initial condition, u0, of the example is:

        u0 = 1,  for the first N/comm.size elements
        u0 = 0,  elsewhere

    Parameters
    ----------
    dx : double
        The space step size.
    N : int
        The number of space points to use in the full domain.
    dt : double
        The time step size. This should be less than `0.5*dx**2` or else the
        numerical results will be unstable.
    Nt : int
        The number of time steps to take.

    Returns
    -------
    = Process 0 =
    u : double array
        The parallel solution.
    = Process >0 =
    None

    """
    # compute the number of space points used by each chunk
    Nx = N // comm.size

    # create space for entire parallel solution as well as each process's chunk
    u = None
    uk = numpy.empty(Nx, dtype=double)

    # rank 0 process creates space for its u and populates it with an initial
    # condition. (see the docstring above) u is then resized for use in
    # MPI_Scatter so that we can distribute each chunk to each process
    if (comm.rank == 0):
        u = numpy.zeros(N, dtype=double)
        u[:Nx] = 1
        u.reshape(comm.size, Nx)

    # each process receives a chunk of the initial data (see MPI_Scatter). the
    # chunk is represented by each row of u (it was reshaped above) and stored
    # in the current process's version of uk
    comm.Scatter(u, uk, root=0)

    # each process performs the heat equation iteration
    uk = heat_parallel(uk, dx, Nx, dt, Nt, comm)

    # gather results into proccess 0's u vector: that is, take each process's
    # individual uk result and concatenate them into Process 0's u array
    #
    # that is, u is equal to the full parallel solution
    comm.Gather(uk, u, root=0)
    if (comm.rank == 0):
        u.reshape(N)

    # have only the rank 0 process return the full parallel solution (everyone
    # else returns None)
    if (comm.rank == 0):
        return u
    else:
        return None


def plot_example_serial_and_parallel(Nt=100):
    r"""
    Compute the example solutions using the given `Nt` and plot the results.

    This is also run in parallel but only Process 0 computes the serial
    solution and plots the results.

    Parameters
    ----------
    Nt : int
        (Default: 100) the number of time steps to take in computing
        the example solution.

    Returns
    -------
    None
    """
    print 'plot_example_serial_and_parallel()'
    comm = MPI.COMM_WORLD

    # example problem parameters
    N = 96               # number x,u-points in entire domain
    Nx = N // comm.size  # number x,u-points in each process's chunk
    dx = 1.0/(N+1)       # step size in x-domain
    dt = 0.4*dx**2       # must be <= 0.5*dx**2 or else F.E. is unstable

    u_parallel = example_parallel_solution(dx, N, dt, Nt, comm)

    # have only process 0 compute the serial solution and plot
    if (comm.rank == 0):
        u_initial = numpy.zeros(N, dtype=double)
        u_initial[:Nx] = 1

        # solve in serial
        u_serial = heat_serial(u_initial, dx, N, dt, Nt)

        # plot results
        plt.clf()
        plt.plot(u_initial,'k--', linewidth=3, label='Initial Condition')
        plt.plot(u_serial, 'r--', linewidth=3, label='Serial Solution')
        plt.plot(u_parallel, 'r-', label='Parallel Solution')
        plt.legend()
        plt.title('Parallel Heat Equation Solution $dx=%.2e \; dt=%.2e \; N_t=%d$'%(dx,dt,Nt),
                  fontsize=14)
        plt.xlabel('$x$', fontsize=20)
        plt.ylabel('$u$', fontsize=20)

        print 'plot_example_serial_and_parallel() --- saving to parallel_heat.png ...'
        plt.savefig('./parallel_heat.png')

def plot_example_serial(chunks=3, Nt=100):
    r"""
    Compute an example solution using `heat_serial()` and plot the result.

    Try plotting with different

    """
    # in case this is accidentally run in parallel, have the non rank=0
    # processes return immediately
    comm = MPI.COMM_WORLD
    if (comm.rank != 0):
        return

    N = 96            # number x,u-points in entire domain
    Nx = N // chunks  # number x,u-points in each process's chunk
    dx = 1.0/(N+1)    # step size in x-domain
    dt = 0.4*dx**2    # must be <= 0.5*dx**2 or else F.E. is unstable

    u_initial = zeros(N, dtype=double)
    u_initial[:Nx] = 1

    u_serial = heat_serial(u_initial, dx, N, dt, Nt)

    plt.clf()
    plt.plot(u_initial,'k--', linewidth=3, label='Initial Condition')
    plt.plot(u_serial, 'r--', linewidth=3, label='Serial Solution')
    plt.legend()
    plt.title('Serial Heat Equation Solution $dx=%.2e \; dt=%.2e \; N_t=%d$'%(dx,dt,Nt),
              fontsize=14)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$u$', fontsize=20)

    print 'plot_example_serial() --- saving to serial_heat.png ...'
    plt.savefig('./serial_heat.png')


if __name__ == '__main__':
    # plot the serial result to see what kind of initial condition and solution
    # is expected.
    #
    #     $ python test_homework4.py
    #
    # (Comment this out when running parallel tests. See below.)
    plot_example_serial(chunks=3, Nt=100)

    ###################################################
    # RUN THE TESTS AND PARALLEL PLOT BELOW USING MPI #
    #                                                 #
    #     $ mpiexec -n 3 python test_homework4.py     #
    #                                                 #
    # (Comment out the serial plot, above, and remove #
    #  the comments below to run the tests!)          #
    ###################################################

    # plot the serial and parallel example data
    plot_example_serial_and_parallel()

    # run the test
    unittest.main(verbosity=2)
