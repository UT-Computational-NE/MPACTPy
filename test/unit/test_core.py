import pytest
from math import isclose
from numpy.testing import assert_allclose

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
    assert len(core.assemblies)    == 1
    assert len(assembly.lattices)  == 1
    assert len(assembly.modules)   == 1
    assert len(assembly.pins)      == 1
    assert len(assembly.pinmeshes) == 1
    assert len(assembly.materials) == 1

def test_core_equality(core, equal_core, unequal_core):
    assert core == equal_core
    assert core != unequal_core

def test_core_hash(core, equal_core, unequal_core):
    assert hash(core) == hash(equal_core)
    assert hash(core) != hash(unequal_core)

def test_core_write_to_string(core):
    output = core.write_to_string(prefix="  ")
    expected_output = "  core\n" + \
                      "      1\n" + \
                      "    1 1 1\n" + \
                      "      1\n"
    assert output == expected_output

def test_core_set_unique_elements(assembly):
    core = Core([[None,     assembly, None    ],
                 [assembly, assembly, assembly],
                 [None,     assembly, None    ]])

    other_assembly = assembly
    core.set_unique_elements([other_assembly])
    assert all([a is other_assembly for row in core.assembly_map for a in row if a])

def test_core_get_axial_slice(core):
    core_slice = core.get_axial_slice(0.5, 1.5)
    pin_slice  = core_slice.pins[-1]

    assert pin_slice.pinmesh.number_of_material_regions == 8
    assert pin_slice.pinmesh.regions_inside_bounds == [0, 1, 2, 4, 5, 6]
    assert isclose(core_slice.height, 1.0)
    assert_allclose(pin_slice.pinmesh.zvals, [0.5, 1.0])
