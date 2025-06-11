
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx.io import gmshio
import gmsh
from dolfinx import fem, io, nls
from dolfinx import plot
import dolfinx.fem.petsc
import dolfinx.nls.petsc
from dolfinx.mesh import create_box, CellType, create_rectangle, locate_entities,meshtags
import pyvista
from ufl import (
    as_matrix,
    as_vector,
    dx,
    dot,
    cos,
    sin,
    SpatialCoordinate,
    Identity,
    grad,
    ln,
    tr,
    det,
    variable,
    derivative,
    TestFunction,
    TrialFunction,
    inner,
    cross,
    inv,
    outer,
)
from petsc4py import PETSc

# %%
# Define the mesh
gmsh.initialize()
gmsh.model.add("beams")

p1 = gmsh.model.geo.addPoint(0, 0, 0)
p2 = gmsh.model.geo.addPoint(30, 0, 0)
p3 = gmsh.model.geo.addPoint(30, 5, 0)
p4 = gmsh.model.geo.addPoint(0, 5, 0)

p5 = gmsh.model.geo.addPoint(5, 2, 0)
p6 = gmsh.model.geo.addPoint(5, 3, 0)
p7 = gmsh.model.geo.addPoint(25, 3, 0)
p8 = gmsh.model.geo.addPoint(25, 2, 0)



l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

l5 = gmsh.model.geo.addLine(p5, p6)
l6 = gmsh.model.geo.addLine(p6, p7)
l7 = gmsh.model.geo.addLine(p7, p8)
l8 = gmsh.model.geo.addLine(p8, p5)

cl2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])




surf1 = gmsh.model.geo.addPlaneSurface([cl1, cl2])

# for i in [surf1, surf2, surf3, surf4, surf5]: #[surf1 ,surf2, surf5]:#
#     gmsh.model.geo.mesh.setTransfiniteSurface(i)
#     gmsh.model.geo.mesh.setRecombine(2, i)
# gmsh.model.geo.mesh.setTransfiniteSurface(surf1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
gmsh.model.addPhysicalGroup(2, [surf1], 1)
# gmsh.model.addPhysicalGroup(2, [surf2], 2)
# gmsh.model.addPhysicalGroup(2, [surf3], 3)
# gmsh.model.addPhysicalGroup(2, [surf4], 4)
# gmsh.model.addPhysicalGroup(2, [surf5], 5)

gmsh.model.setPhysicalName(2, 1, "left_clamp")
# gmsh.model.setPhysicalName(2, 2, "beam_bottom")
# gmsh.model.setPhysicalName(2, 3, "beam_middle")
# gmsh.model.setPhysicalName(2, 4, "beam_top")
# gmsh.model.setPhysicalName(2, 5, "right_clamp")

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(2)
# #
# gmsh.fltk.run()

# gmsh.write("dumbbell_structured.msh")
# %%
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
gmsh.finalize()
dim = msh.topology.dim
print(f"Mesh: {msh.name} with {msh.topology.index_map(dim).size_local} element in {dim}D")

degree = 1
shape = (dim,)

# Define the finite element function space
V = fem.functionspace(msh, ("P", degree, shape))
u = fem.Function(V,name="Displacement")
T_s= fem.functionspace(msh, ("CG", 1)) # Define the scalar function space for the theta
v = TestFunction(V)
du = TrialFunction(V)
dtheta = ufl.TrialFunction(T_s)


# Define the mechanical properties
E = 1.0
nu = 0.3
G = fem.Constant(msh,E / (2.0 * (1.0 + nu)))
K = fem.Constant(msh,E / (3.0 * (1.0 - 2.0 * nu)))
mu0 = fem.Constant(msh,1.00e2)

# Define the design variable
theta = fem.Function(T_s, name="theta")
B_0 = fem.Constant(msh, PETSc.ScalarType(1.0))
B_tilde = B_0* as_vector([cos(theta), sin(theta)])


# Define the load
B_applied = fem.Constant(msh, np.array([0.0, -1.0e-3], dtype=np.float64))

# Define the constitutive model

F = Identity(dim) + grad(u)
J = det(F)

# Compute the right Cauchy-Green deformation tensor
C = F.T * F

# Compute the strain energy density function
I1 = tr(C)
W_tilde = G/2*(J**(-2/3)*I1-2)+K/2*(J-1)**2#-1/mu0*inner(F*B_tilde,B_applied)

P = G*J**(-2/3)*(F-I1/2*inv(F).T) + K*J*(J-1)*inv(F).T - 1/mu0*outer(B_applied,B_tilde)

# Define the weak form
Residual = inner(P, grad(v))*dx


# Define the Jacobian
Jacobian = derivative(Residual, u, du)
dpdtheta = derivative(P, theta, dtheta)

# Define boundary conditions
left = lambda x: np.isclose(x[0], 0.0)
bc_1 = fem.dirichletbc(np.array([0.0, 0.0]), fem.locate_dofs_geometrical(V, left),V)
right = lambda x: np.isclose(x[0], 25.0)
bc_2 = fem.dirichletbc(np.array([-1.0, 0.0]), fem.locate_dofs_geometrical(V, right),V)

# Define the solution function

def solve_problem(u):
    # Create the linear problem
    problem = fem.petsc.NonlinearProblem(Residual, u, [bc_1,bc_2], Jacobian)
    solver = nls.petsc.NewtonSolver(msh.comm, problem)
    # Set Newton solver options
    solver.atol = 1e-4
    solver.rtol = 1e-4
    solver.max_it = 10000
    solver.convergence_criterion = "incremental"
    solver.solve(u) 
    return u

# Define Dirac 
x_target = np.array([5.0, 2.5], dtype=np.float64)
cells = locate_entities(msh, dim, marker=lambda x: np.isclose(x[0], x_target[0]) & np.isclose(x[1], x_target[1]))
cells_tag = meshtags(msh, dim, cells, np.full(len(cells), 1, dtype=np.int32))
dx_sub = ufl.Measure("dx", domain=msh, subdomain_data=cells_tag)

# Initialize the optimization loop
theta.x.array[:] = np.pi / 6  # Reset theta to initial guess
u.x.array[:] = 0.0  # Reset the displacement field
# Record the last step
u_last = u.copy()
# Solve the problem
u = solve_problem(u)

# Compute the objective function
phi = ufl.inner(u[0], fem.Constant(msh, PETSc.ScalarType(1))) * dx_sub(1)

dphi_dtheta = ufl.derivative(phi, theta, dtheta)
dR_dtheta = ufl.derivative(Residual, theta, dtheta)

# Define the adjoint problem
lambda_ = TrialFunction(V)
v_t = TestFunction(V)
dphi_du = derivative(derivative(phi, u, du), u, v_t)
dR_du = -derivative(Residual, u, du)
A = fem.petsc.assemble_matrix(fem.form(dphi_du),diagonal=True)
A.assemble()
b = fem.petsc.assemble_vector(fem.form(dR_du))

# Solve the adjoint problem
solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)  # AMG
fem.petsc.apply_lifting(b, [fem.form(dphi_du)], [[bc_1, bc_2]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, [bc_1, bc_2])


lamda_sol = fem.Function(V)
solver.solve(b, lamda_sol.x.petsc_vec)
