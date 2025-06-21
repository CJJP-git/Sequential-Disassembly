import numpy as np
import ufl

import os

path = "/Users/jipengcui/Documents/0. PhD/2. Research/Sequential-Disassembly/Results/Opt"
os.makedirs(path, exist_ok=True)

from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.io import XDMFFile
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
msh = create_rectangle(    comm=MPI.COMM_WORLD,
    points=[[0.0, 0.0], [10.0, 1.0]],
    n=[100, 10],
    cell_type=CellType.quadrilateral)
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
theta = fem.Function(T_s, name="theta")
theta.x.array[:] = 2 * np.pi * np.random.random(size=theta.x.array.shape)  # Initialize theta to zero
dtheta = ufl.TrialFunction(T_s)


# Define the mechanical properties
E = 1.0e3
nu = 0.3
G = fem.Constant(msh,E / (2.0 * (1.0 + nu)))
K = fem.Constant(msh,E / (3.0 * (1.0 - 2.0 * nu)))
mu0 = fem.Constant(msh,np.pi*4.00e-10)

# Define the design variable

B_0 = fem.Constant(msh, PETSc.ScalarType(1e-3))
B_tilde = B_0* as_vector([cos(theta), sin(theta)])


# Define the load
B_applied = fem.Constant(msh, np.array([0.0, -5e-6], dtype=np.float64))

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
right = lambda x: np.isclose(x[0], 10.0)
bc_2 = fem.dirichletbc(np.array([0.0, 0.0]), fem.locate_dofs_geometrical(V, right),V)
bc = [bc_1, bc_2]

xdmf_theta = XDMFFile(msh.comm, os.path.join(path, "results_theta.xdmf"), "w")
xdmf_u = XDMFFile(msh.comm, os.path.join(path, "results_u.xdmf"), "w")
xdmf_theta.write_mesh(msh)
xdmf_u.write_mesh(msh)

# Optimization loop
n_steps = 100
alpha = 1e4

# Select all y-DoFs within 0.5 units of the midpoint (1.0, 0.5)
center = np.array([5.0, 0.0])
radius = 0.1
dofs_y = fem.locate_dofs_geometrical(
    (V.sub(1), T_s),
    lambda x: np.linalg.norm(x[:2] - center[:, np.newaxis], axis=0) < radius
)

for step in range(n_steps):
    print(f"\n--- Optimization step {step} ---")

    # Reset u
    u.x.array[:] = 0.0

    # Redefine field-dependent expressions
    F = Identity(dim) + grad(u)
    J = det(F)
    C = F.T * F
    I1 = tr(C)
    P = G*J**(-2/3)*(F-I1/2*inv(F).T) + K*J*(J-1)*inv(F).T - 1/mu0*outer(B_applied,B_tilde)

    # Residual and Jacobian
    Residual = inner(P, grad(v))*dx
    Jacobian = derivative(Residual, u, du)

    # Solve the state problem
    problem = fem.petsc.NonlinearProblem(Residual, u, bc, Jacobian)
    solver = nls.petsc.NewtonSolver(msh.comm, problem)
    solver.atol = 1e-4
    solver.rtol = 1e-4
    solver.max_it = 100
    solver.convergence_criterion = "incremental"
    solver.solve(u)

    # Create selection function w
    w = fem.Function(V)
    w.x.array[:] = 0.0
    w.x.array[dofs_y[0]] = 1.0

    # Define objective
    phi = inner(u, w) * dx
    dphi_dtheta = derivative(phi, theta, dtheta)
    dR_dtheta = derivative(Residual, theta, dtheta)
    dR_du = derivative(Residual, u, du)

    # Solve adjoint problem
    A = fem.petsc.assemble_matrix(fem.form(dR_du), bcs=bc)
    A.assemble()
    b_form = fem.form(inner(w, v) * dx)
    b = -fem.petsc.assemble_vector(b_form)
    fem.petsc.apply_lifting(b, [fem.form(dR_du)], [bc])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bc)
    lamda_sol = fem.Function(V)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    ksp.solve(b, lamda_sol.x.petsc_vec)

    # Compute gradient
    adj_term = ufl.replace(dR_dtheta, {v: lamda_sol})
    dL_dtheta = dphi_dtheta + adj_term
    grad_theta = fem.petsc.assemble_vector(fem.form(dL_dtheta))
    grad_theta.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Update theta
    theta.x.array[:] -= alpha * grad_theta.array

    # Write current theta and u to XDMF files
    xdmf_theta.write_function(theta, step)
    xdmf_u.write_function(u, step)

    print("Current displacement at target point:", u.x.array[dofs_y[0]])

xdmf_theta.close()
xdmf_u.close()