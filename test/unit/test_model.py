import pytest
from math import isclose
from numpy.testing import assert_allclose

from mpactpy  import Material, RectangularPinMesh, GeneralCylindricalPinMesh, \
                     Pin, Module, Lattice, Assembly, Core, Model
from test.unit.test_material import material, equal_material, unequal_material
from test.unit.test_pinmesh import general_cylindrical_pinmesh as pinmesh,\
                                   equal_general_cylindrical_pinmesh as equal_pinmesh,\
                                   unequal_general_cylindrical_pinmesh as unequal_pinmesh
from test.unit.test_pin import pin, equal_pin, unequal_pin
from test.unit.test_module import module, equal_module, unequal_module
from test.unit.test_lattice import lattice, equal_lattice, unequal_lattice
from test.unit.test_assembly import assembly, equal_assembly, unequal_assembly
from test.unit.test_core import core, equal_core, unequal_core


@pytest.fixture
def model(core):
    return Model(core          = core,
                 states        = [{"power": "0.0"}],
                 xsec_settings = {"xslib": "ORNL mpact_lib.fmt"},
                 options       = {"bound_cond": "1 1 1 1 1 1"})

@pytest.fixture
def equal_model(equal_core):
    return Model(core          = equal_core,
                 states        = [{"power": "0.0"}],
                 xsec_settings = {"xslib": "ORNL mpact_lib.fmt"},
                 options       = {"bound_cond": "1 1 1 1 1 1"})

@pytest.fixture
def unequal_model(unequal_core):
    return Model(core          = unequal_core,
                 states        = [{"power": "0.0"}],
                 xsec_settings = {"xslib": "ORNL mpact_lib.fmt"},
                 options       = {"bound_cond": "1 1 1 1 1 1"})

def test_core_initialization(model, core):
    assert model.core          == core
    assert model.states        == [{"power": "0.0"}]
    assert model.xsec_settings == {"xslib": "ORNL mpact_lib.fmt"}
    assert model.options       == {"bound_cond": "1 1 1 1 1 1"}

def test_model_equality(model, equal_model, unequal_model):
    assert model == equal_model
    assert model != unequal_model

def test_model_hash(model, equal_model, unequal_model):
    assert hash(model) == hash(equal_model)
    assert hash(model) != hash(unequal_model)

def test_model_write_to_string(model, material):
    output = model.write_to_string(caseid="test", indent=2)
    expected_output = \
f"""CASEID test

MATERIAL
  mat 1 1 {material.density} g/cc 300.0 K \\
    6001 0.002
    92235 0.001

STATE power 0.0

GEOM
  mod_dim 4.0 4.0 3.0

  pinmesh 1 gcyl 0.5 1.0 / -1.0 1.0 -1.0 1.0 / 1.0 2.0 3.0 / 1 2 / 8 8 8 8 / 5 5 5

  pin 1 1 / 1 1 1 1 1 1 1 1 1

  module 1 2 2 1
    1 1
    1 1

  lattice 1 2 2
    1 1
    1 1

  assembly 1
    1 1 1 1

  core
      1
    1 1 1
      1

XSEC
  xslib ORNL mpact_lib.fmt

OPTION
  bound_cond 1 1 1 1 1 1

"""
    assert output == expected_output



def test_harder_model():
    """ This tests a harder model with multiple repeated element definitions that the model
        must successfully NOT repeat when writing to string
    """

    mat = [Material(300.0, {"U235": 1e-3, "H": 2e-3}, Material.MPACTSpecs(replace_isotopes={"H": "H1"})),
           Material(300.0, {"U235": 5e-3, "H": 8e-3}, Material.MPACTSpecs(replace_isotopes={"H": "H1"}))]

    pin_meshes = [RectangularPinMesh([0.5, 1.0], [0.5, 1.0], [1.0], [2, 2], [1, 1], [1]),
                  RectangularPinMesh([0.5, 1.0], [0.5, 1.0], [2.0], [2, 2], [1, 1], [1]),
                  GeneralCylindricalPinMesh([0.25, 0.5, 1.0], -0.5, 0.5, -0.5, 0.5, [1.0], [1, 1, 1], [16, 16, 16, 16], [1]), # Tests having a radius that falls outside the bounds
                  GeneralCylindricalPinMesh([0.25, 0.5],      -0.5, 0.5, -0.5, 0.5, [2.0], [1, 2], [16, 16, 16, 16], [1]),
                  RectangularPinMesh([0.5, 1.0], [0.5, 1.0], [1.0], [2, 2], [1, 1], [1])] # Adding a repeated pin mesh to test filtering of repeated pin meshes

    p = [Pin(pin_meshes[4], [mat[0], mat[0], mat[0], mat[0]]),
         Pin(pin_meshes[1], [mat[0], mat[0], mat[0], mat[0]]),
         Pin(pin_meshes[2], [mat[1], mat[1], mat[0], mat[1]]),  # Tests having a radius that falls outside the bounds.  The last material should not be written
         Pin(pin_meshes[3], [mat[1], mat[1], mat[0]]),
         Pin(pin_meshes[0], [mat[0], mat[0], mat[0], mat[0]])] # Adding a repeated to test filtering of repeated

    m = [Module(1, [[p[0],p[2]],
                    [p[2],p[0]]]),

         Module(1, [[p[2],p[0]],
                    [p[0],p[2]]]),

         Module(1, [[p[3],p[1]],
                    [p[1],p[3]]]),

         Module(1, [[p[0],p[2]],
                    [p[2],p[4]]])] # Adding a repeated to test filtering of repeated

    l = [Lattice([[m[3]]]),
         Lattice([[m[2]]]),
         Lattice([[m[1]]]),
         Lattice([[m[0]]])] # Adding a repeated to test filtering of repeated

    a = [Assembly([l[2], l[0], l[1], l[0]]),
         Assembly([l[0], l[2], l[1], l[2]]),
         Assembly([l[2], l[0], l[1], l[3]])] # Adding a repeated assembly to test filtering of repeated

    output = Model(core          = Core(symmetry_opt = "360", assembly_map = [[a[0],a[1]],
                                                                              [a[1],a[2]]]),
                   states        = [{"power": "0.0"}],
                   xsec_settings = {"xslib": "ORNL mpact_lib.fmt"},
                   options       = {"bound_cond": "1 1 1 1 1 1"}).write_to_string(caseid="test", indent=2)
    expected_output = \
f"""CASEID test

MATERIAL
  mat 1 0 {mat[1].density} g/cc 300.0 K \\
    1001 0.008
    92235 0.005
  mat 2 0 {mat[0].density} g/cc 300.0 K \\
    1001 0.002
    92235 0.001

STATE power 0.0

GEOM
  mod_dim 2.0 2.0 1.0 2.0

  pinmesh 1 gcyl 0.25 0.5 / -0.5 0.5 -0.5 0.5 / 1.0 / 1 1 / 16 16 16 / 1
  pinmesh 2 rec 0.5 1.0 / 0.5 1.0 / 1.0 / 2 2 / 1 1 / 1
  pinmesh 3 gcyl 0.25 0.5 / -0.5 0.5 -0.5 0.5 / 2.0 / 1 2 / 16 16 16 16 / 1
  pinmesh 4 rec 0.5 1.0 / 0.5 1.0 / 2.0 / 2 2 / 1 1 / 1

  pin 1 1 / 1 1 2
  pin 2 2 / 2 2 2 2
  pin 3 3 / 1 1 2
  pin 4 4 / 2 2 2 2

  module 1 2 2 1
    1 2
    2 1
  module 2 2 2 1
    2 1
    1 2
  module 3 2 2 1
    3 4
    4 3

  lattice 1 1 1
    1
  lattice 2 1 1
    2
  lattice 3 1 1
    3

  assembly 1
    1 2 3 2
  assembly 2
    2 1 3 1

  core 360
    1 2
    2 1

XSEC
  xslib ORNL mpact_lib.fmt

OPTION
  bound_cond 1 1 1 1 1 1

"""
    assert output == expected_output
