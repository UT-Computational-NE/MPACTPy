import pytest
from math import isclose
from numpy.testing import assert_allclose
from mpactpy import PinMesh, RectangularPinMesh, GeneralCylindricalPinMesh, Material, \
                    Pin, build_rec_pin, build_gcyl_pin
from test.unit.test_material import material, equal_material, unequal_material
from test.unit.test_pinmesh import general_cylindrical_pinmesh as pinmesh,\
                                   equal_general_cylindrical_pinmesh as equal_pinmesh,\
                                   unequal_general_cylindrical_pinmesh as unequal_pinmesh, \
                                   rectangular_pinmesh as overlay_mesh, \
                                   pinmesh_2D, openmc_pin, openmc_fuel_material, openmc_moderator_material


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

@pytest.fixture
def template_material():
    return Material(300.0, {"H": 1e0}, Material.MPACTSpecs())

@pytest.fixture
def template_pin(template_material, overlay_mesh):
    T = template_material
    materials = [T, T, T,
                 T, T, T,
                 T, T, T] * 3
    return Pin(overlay_mesh, materials)

@pytest.fixture
def overlay_pin(openmc_fuel_material, openmc_moderator_material, overlay_mesh):
    F = Material.from_openmc_material(openmc_fuel_material)
    M = Material.from_openmc_material(openmc_moderator_material)
    materials = [M, M, M,
                 M, F, M,
                 M, M, M] * 3
    return Pin(overlay_mesh, materials)

@pytest.fixture
def pin_2D(material, pinmesh_2D):
    materials = [material for _ in range(pinmesh_2D.number_of_material_regions)]
    return Pin(pinmesh_2D, materials)

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

def test_pin_write_to_string(pin, pinmesh, material):
    output = pin.write_to_string(prefix="  ",
                                 material_mpact_ids={material: 3},
                                 pinmesh_mpact_ids={pinmesh: 2},
                                 pin_mpact_ids={pin: 1})
    expected_output = "  pin 1 2 / 3 3 3 3 3 3 3 3 3\n"
    assert output == expected_output

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

def test_pin_with_height(pin, pin_2D):
    new_pin = pin_2D.with_height(3.0)
    assert isclose(new_pin.pitch["Z"], 3.0)

    with pytest.raises(AssertionError, match=f"len\(zvals\) = {len(pin.pinmesh.zvals)}, Pin must be strictly 2D"):
        new_pin = pin.with_height(3.0)

def test_pin_subdivide(pinmesh):
    materials = [Material(300.0, {"H1": (material_index + 1) * 1e-6}, Material.MPACTSpecs())
                 for material_index in range(pinmesh.number_of_material_regions)]
    pin = Pin(pinmesh, materials)
    subdivisions = GeneralCylindricalPinMesh.Subdivisions(subd_r=[2, 1, 1, 2], subd_z=[1, 2, 1])

    subdivided_pin = pin.subdivide(subdivisions)

    material_map = [0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 4, 4, 5, 6, 7, 8, 8, 9, 10, 11]
    assert subdivided_pin.pinmesh.number_of_material_regions == 20
    assert len(subdivided_pin.materials) == 20
    assert all(actual_material is materials[material_index]
               for actual_material, material_index in zip(subdivided_pin.materials, material_map))

def test_pin_subdivide_rectangular(overlay_mesh):
    materials = [Material(300.0, {"H1": (material_index + 1) * 1e-6}, Material.MPACTSpecs())
                 for material_index in range(overlay_mesh.number_of_material_regions)]
    pin = Pin(overlay_mesh, materials)
    subdivisions = RectangularPinMesh.Subdivisions(subd_x=[2, 1, 1], subd_y=[1, 2, 1], subd_z=[1, 1, 2])

    subdivided_pin = pin.subdivide(subdivisions)

    material_map = [z * 9 + y * 3 + x
                    for z in [0, 1, 2, 2]
                    for y in [0, 1, 1, 2]
                    for x in [0, 0, 1, 2]]
    assert subdivided_pin.pinmesh.number_of_material_regions == 64
    assert len(subdivided_pin.materials) == 64
    assert all(actual_material is materials[material_index]
               for actual_material, material_index in zip(subdivided_pin.materials, material_map))

def test_pin_divide_into_quadrants_rectangular(overlay_mesh):
    materials = [Material(300.0, {"H1": (material_index + 1) * 1e-6}, Material.MPACTSpecs())
                 for material_index in range(overlay_mesh.number_of_material_regions)]
    pin = Pin(overlay_mesh, materials)

    quadrants = pin.divide_into_quadrants()

    expected_maps = [
        [[z * 9 + y * 3 + x for z in range(3) for y in [1, 2] for x in [0, 1]],
         [z * 9 + y * 3 + x for z in range(3) for y in [1, 2] for x in [1, 2]]],
        [[z * 9 + y * 3 + x for z in range(3) for y in [0, 1] for x in [0, 1]],
         [z * 9 + y * 3 + x for z in range(3) for y in [0, 1] for x in [1, 2]]],
    ]

    for row, expected_map_row in zip(quadrants, expected_maps):
        for quadrant_pin, expected_map in zip(row, expected_map_row):
            assert quadrant_pin.pinmesh.number_of_material_regions == 12
            assert len(quadrant_pin.materials) == 12
            assert all(actual_material is materials[material_index]
                       for actual_material, material_index in zip(quadrant_pin.materials, expected_map))

def test_pin_overlay(openmc_pin, template_pin, template_material, overlay_pin):
    geometry                      = openmc_pin
    offset                        = (-1.5, -1.5, 0.0)
    overlay_policy                = PinMesh.OverlayPolicy(method="centroid")
    include_only: Pin.OverlayMask = {template_material}

    pin          = template_pin.overlay(geometry, offset, include_only, overlay_policy)
    expected_pin = overlay_pin
    assert pin == expected_pin


def test_build_gcyl_pin(material):
    pin = build_gcyl_pin(bounds                  = (-2.5, 2.5, -2.5, 2.5),
                         thicknesses             = {"R": [2.0], "Z": [1.0]},
                         materials               = [material, material],
                         target_cell_thicknesses = {"R": 1.0, "S": 6.0, "Z": 0.5})

    materials    = [material, material, material]
    pin_mesh     = GeneralCylindricalPinMesh(r     = [1.0, 2.0],
                                             xMin  = -2.5, xMax = 2.5,
                                             yMin  = -2.5, yMax = 2.5,
                                             zvals = [1.0],
                                             ndivr = [1, 1], ndiva = [4, 4, 4], ndivz = [2])
    expected_pin = Pin(pin_mesh, materials)

    assert pin == expected_pin

def test_build_gcyl_pin_default_target_cell_thicknesses(material):
    pin = build_gcyl_pin(bounds      = (-2.5, 2.5, -2.5, 2.5),
                         thicknesses = {"R": [2.0], "Z": [1.0]},
                         materials   = [material, material])

    pin_mesh = GeneralCylindricalPinMesh(r     = [2.0],
                                         xMin  = -2.5, xMax = 2.5,
                                         yMin  = -2.5, yMax = 2.5,
                                         zvals = [1.0],
                                         ndivr = [1], ndiva = [1, 1], ndivz = [1])
    expected_pin = Pin(pin_mesh, [material, material])

    assert pin == expected_pin

def test_build_rec_pin(material):
    pin = build_rec_pin(thicknesses             = {"X": [1.0, 1.0], "Y": [3.0], "Z": [5.0]},
                        materials               = [material, material],
                        target_cell_thicknesses = {"X": 0.5, "Y": 1.5})

    materials    = [material, material]
    pin_mesh     = RectangularPinMesh(xvals = [1.0, 2.0],
                                      yvals = [3.0],
                                      zvals = [5.0],
                                      ndivx = [2, 2],
                                      ndivy = [2],
                                      ndivz = [1])
    expected_pin = Pin(pin_mesh, materials)

    assert pin == expected_pin

def test_build_rec_pin_rounds_up_target_cell_thicknesses(material):
    pin = build_rec_pin(thicknesses             = {"X": [1.1], "Y": [1.0], "Z": [1.0]},
                        materials               = [material],
                        target_cell_thicknesses = {"X": 0.5})

    pin_mesh = RectangularPinMesh(xvals = [1.1],
                                  yvals = [1.0],
                                  zvals = [1.0],
                                  ndivx = [3],
                                  ndivy = [1],
                                  ndivz = [1])
    expected_pin = Pin(pin_mesh, [material])

    assert pin == expected_pin
