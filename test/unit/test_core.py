import pytest
from math import isclose
from numpy.testing import assert_allclose

from mpactpy.pin import build_rec_pin
from mpactpy.module import Module
from mpactpy.lattice import Lattice
from mpactpy.assembly import Assembly
from mpactpy.core import Core
from test.unit.test_material import material, equal_material, unequal_material
from test.unit.test_pinmesh import general_cylindrical_pinmesh as pinmesh,\
                                   equal_general_cylindrical_pinmesh as equal_pinmesh,\
                                   unequal_general_cylindrical_pinmesh as unequal_pinmesh
from test.unit.test_pin import pin, equal_pin, unequal_pin
from test.unit.test_module import module, equal_module, unequal_module
from test.unit.test_lattice import lattice, equal_lattice, unequal_lattice
from test.unit.test_assembly import assembly, equal_assembly, unequal_assembly


@pytest.fixture
def core(assembly):
    return Core([[None,     assembly, None    ],
                 [assembly, assembly, assembly],
                 [None,     assembly, None    ]])

@pytest.fixture
def equal_core(equal_assembly):
    return Core([[None,           equal_assembly, None          ],
                 [equal_assembly, equal_assembly, equal_assembly],
                 [None,           equal_assembly, None          ]])

@pytest.fixture
def unequal_core(unequal_assembly):
    return Core([[None,             unequal_assembly, None            ],
                 [unequal_assembly, unequal_assembly, unequal_assembly],
                 [None,             unequal_assembly, None            ]])

def test_core_initialization(core, assembly):
    assert isclose(core.height, 12.0)
    assert core.symmetry_opt    == ""
    assert core.quarter_sym_opt == ""
    assert core.nx              == 3
    assert core.ny              == 3
    assert core.nz              == 4

    assert_allclose([core.mod_dim[i] for i in ['X','Y']], [4., 4.])
    assert_allclose(core.mod_dim['Z'], [3.])
    assert all([a == assembly for row in core.assembly_map for a in row if a])
    assert len(core.assemblies) == 1
    assert len(core.lattices)   == 1
    assert len(core.modules)    == 1
    assert len(core.pins)       == 1
    assert len(core.pinmeshes)  == 1
    assert len(core.materials)  == 1

def test_core_equality(core, equal_core, unequal_core):
    assert core == equal_core
    assert core != unequal_core

def test_core_hash(core, equal_core, unequal_core):
    assert hash(core) == hash(equal_core)
    assert hash(core) != hash(unequal_core)

def test_core_write_to_string(core, assembly):
    output = core.write_to_string(prefix="  ",
                                  assembly_mpact_ids={assembly: 3})
    expected_output = "  core\n" + \
                      "      3\n" + \
                      "    3 3 3\n" + \
                      "      3\n"
    assert output == expected_output

def test_core_get_axial_slice(core):
    core_slice = core.get_axial_slice(0.5, 1.5)
    pin_slice  = core_slice.pins[-1]

    assert pin_slice.pinmesh.number_of_material_regions == 8
    assert pin_slice.pinmesh.regions_inside_bounds == [0, 1, 2, 4, 5, 6]
    assert isclose(core_slice.height, 1.0)
    assert_allclose(pin_slice.pinmesh.zvals, [0.5, 1.0])

def test_core_axial_mesh_unionization(material):

    pin_1 = build_rec_pin(thicknesses = {"X": [1.0], "Y": [3.0], "Z": [4.0]},
                          materials   = [material])

    pin_2 = build_rec_pin(thicknesses = {"X": [1.0], "Y": [3.0], "Z": [1.0]},
                          materials   = [material])

    mod_1 = Module(1, [[pin_1]])
    mod_2 = Module(1, [[pin_2]])
    lat_1 = Lattice([[mod_1]])
    lat_2 = Lattice([[mod_2]])
    asy_1 = Assembly([lat_1])
    asy_2 = Assembly([lat_2, lat_2, lat_2, lat_2])

    core = Core([[asy_1, asy_2],
                 [asy_1, asy_1,]])

    assert core.nx == 2
    assert core.ny == 2
    assert core.nz == 4

    assert_allclose(core.mod_dim['Z'], [1.])
    assert all([a == asy_2 for row in core.assembly_map for a in row if a])
    assert len(core.assemblies) == 1
    assert len(core.lattices)   == 1
    assert len(core.modules)    == 1
    assert len(core.pins)       == 1
    assert len(core.pinmeshes)  == 1
    assert len(core.materials)  == 1