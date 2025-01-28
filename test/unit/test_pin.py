import pytest
from numpy.testing import assert_allclose
from mpactpy.pin import Pin
from test.unit.test_material import material, equal_material, unequal_material
from test.unit.test_pinmesh import general_cylindrical_pinmesh as pinmesh,\
                                   equal_general_cylindrical_pinmesh as equal_pinmesh,\
                                   unequal_general_cylindrical_pinmesh as unequal_pinmesh


@pytest.fixture
def pin(material, pinmesh):
    materials = [material for _ in range(pinmesh.number_of_material_regions)]
    return Pin(pinmesh, materials)

@pytest.fixture
def equal_pin(equal_material, equal_pinmesh):
    materials = [equal_material for _ in range(equal_pinmesh.number_of_material_regions)]
    return Pin(equal_pinmesh, materials)

@pytest.fixture
def unequal_pin(unequal_material, unequal_pinmesh):
    materials = [unequal_material for _ in range(unequal_pinmesh.number_of_material_regions)]
    return Pin(unequal_pinmesh, materials)

def test_pin_initialization(pin, pinmesh, material):
    number_of_material_regions = pinmesh.number_of_material_regions
    assert pin.pinmesh        == pinmesh
    assert len(pin.materials) == number_of_material_regions
    assert pin.materials      == [material for _ in range(number_of_material_regions)]
    assert_allclose([pin.pitch[i] for i in ['X','Y','Z']], [2., 2., 3.])

def test_pin_equality(pin, equal_pin, unequal_pin):
    assert pin == equal_pin
    assert pin != unequal_pin

def test_pin_hash(pin, equal_pin, unequal_pin):
    assert hash(pin) == hash(equal_pin)
    assert hash(pin) != hash(unequal_pin)

def test_pin_write_to_string(pin):
    output = pin.write_to_string(prefix="  ")
    expected_output = "  pin 1 1 / 1 1 1 1 1 1 1 1 1\n"
    assert output == expected_output

def test_pin_set_unique_elements(material, pinmesh):
    materials = [material for _ in range(pinmesh.number_of_material_regions)]
    pin = Pin(pinmesh, materials)

    other_material = material
    other_pinmesh  = pinmesh
    pin.set_unique_elements([other_pinmesh], [other_material])

    assert pin.pinmesh is other_pinmesh
    assert all([mat is other_material for mat in pin.materials])


def test_pin_get_axial_slice(pin):
    pin_slice = pin.get_axial_slice(0.5, 1.5)

    assert pin_slice.pinmesh.number_of_material_regions == 8
    assert pin_slice.pinmesh.regions_inside_bounds == [0, 1, 2, 4, 5, 6]
    assert_allclose([pin_slice.pinmesh.pitch[i] for i in ['X','Y','Z']], [2., 2., 1.])
    assert_allclose(pin_slice.pinmesh.zvals, [0.5, 1.0])

def test_pin_axial_merge(pin):
    merged_pin = pin.axial_merge(pin)

    assert merged_pin.pinmesh.number_of_material_regions == 24
    assert len(merged_pin.materials) == 24
    assert len(merged_pin.pinmesh.ndivz) == 6
    assert_allclose([merged_pin.pinmesh.pitch[i] for i in ['X','Y','Z']], [2., 2., 6.])
    assert_allclose(merged_pin.pinmesh.zvals, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
