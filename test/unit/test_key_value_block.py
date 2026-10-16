import pytest

from mpactpy import KeyValueBlock


@pytest.fixture
def key_value_block():
    block = KeyValueBlock("OPTION")
    block.add("bound_cond", "1 1 1 1 1 1")
    block.add("param",      "MOC_KERNEL SIK 2")
    block.add("param",      "SHIELD_MOC_KERNEL SIK 2")
    block.add_comment("simplifiedth T SS")
    block.add("conv_crit",  "1E-6 1E-4", comment="k and flux")
    return block

@pytest.fixture
def equal_key_value_block():
    block = KeyValueBlock("option")
    block.add("BOUND_COND", "1 1 1 1 1 1", comment="different comment")
    block.add("param",      "MOC_KERNEL SIK 2")
    block.add("param",      "SHIELD_MOC_KERNEL SIK 2")
    block.add("conv_crit",  "1E-6 1E-4")
    return block

@pytest.fixture
def unequal_key_value_block():
    block = KeyValueBlock("OPTION")
    block.add("bound_cond", "1 1 1 1 1 1")
    block.add("param",      "SHIELD_MOC_KERNEL SIK 2")
    block.add("param",      "MOC_KERNEL SIK 2")
    block.add("conv_crit",  "1E-6 1E-4")
    return block

def test_key_value_block_initialization(key_value_block):
    assert key_value_block.name == "OPTION"
    assert len(key_value_block.entries) == 5
    assert key_value_block.entries[3] == KeyValueBlock.Entry(None, "simplifiedth T SS")
    assert key_value_block.entries[4] == KeyValueBlock.Entry("conv_crit", "1E-6 1E-4", "k and flux")

def test_key_value_block_equality(key_value_block, equal_key_value_block, unequal_key_value_block):
    assert key_value_block == equal_key_value_block
    assert key_value_block != unequal_key_value_block
    assert KeyValueBlock("STATE") != KeyValueBlock("OPTION")

def test_key_value_block_hash(key_value_block, equal_key_value_block, unequal_key_value_block):
    assert hash(key_value_block) == hash(equal_key_value_block)
    assert hash(key_value_block) != hash(unequal_key_value_block)

def test_key_value_block_write_to_string(key_value_block):
    output = key_value_block.write_to_string(prefix="  ")
    expected_output = "OPTION\n" + \
                      "  bound_cond 1 1 1 1 1 1\n" + \
                      "  param MOC_KERNEL SIK 2\n" + \
                      "  param SHIELD_MOC_KERNEL SIK 2\n" + \
                      "  !simplifiedth T SS\n" + \
                      "  conv_crit 1E-6 1E-4 !k and flux\n"
    assert output == expected_output

    block = KeyValueBlock("TRANS")
    block.add("enable", "")
    assert block.write_to_string(prefix="  ") == "TRANS\n" + \
                                                 "  enable\n"

    with pytest.raises(AssertionError):
        key_value_block.write_to_string(prefix="")

def test_key_value_block_access(key_value_block):
    assert key_value_block["param"] == "MOC_KERNEL SIK 2"
    assert key_value_block["PARAM"] == "MOC_KERNEL SIK 2"
    assert key_value_block.get_all("param") == ["MOC_KERNEL SIK 2", "SHIELD_MOC_KERNEL SIK 2"]
    assert key_value_block.get_all("ray") == []
    assert "Conv_Crit" in key_value_block
    assert "ray" not in key_value_block
    assert "simplifiedth" not in key_value_block

    with pytest.raises(KeyError):
        key_value_block["ray"]

def test_key_value_block_set(key_value_block):
    key_value_block["solver"] = "1 2"
    assert key_value_block.entries[-1] == KeyValueBlock.Entry("solver", "1 2")

    key_value_block["PARAM"] = "MOC_KERNEL LS 2"
    assert key_value_block.get_all("param") == ["MOC_KERNEL LS 2", "SHIELD_MOC_KERNEL SIK 2"]
    assert key_value_block.entries[1].keyword == "param"

    key_value_block.set("ray", "0.05 CHEBYSHEV-YAMAMOTO 16 3", comment="ray spacing")
    assert key_value_block.entries[-1] == KeyValueBlock.Entry("ray", "0.05 CHEBYSHEV-YAMAMOTO 16 3", "ray spacing")

def test_key_value_block_set_comment_rule(key_value_block):
    key_value_block["conv_crit"] = "1E-6 1E-4"
    assert key_value_block.entries[4].comment == "k and flux"

    key_value_block.set("conv_crit", "1E-7 1E-5", keep_comment=True)
    assert key_value_block.entries[4].comment == "k and flux"

    key_value_block.set("conv_crit", "1E-8 1E-6", comment="tighter")
    assert key_value_block.entries[4].comment == "tighter"

    key_value_block["conv_crit"] = "1E-9 1E-7"
    assert key_value_block.entries[4].comment is None

    with pytest.raises(AssertionError):
        key_value_block.set("conv_crit", "1E-6 1E-4", comment="new", keep_comment=True)

def test_key_value_block_delete(key_value_block):
    del key_value_block["param"]
    assert "param" not in key_value_block
    assert len(key_value_block.entries) == 3

    with pytest.raises(KeyError):
        del key_value_block["param"]

def test_key_value_block_from_dict():
    block = KeyValueBlock.from_dict("STATE", {"power":   "0.0",
                                          "deplete": "DAYS 0.0"})
    expected_output = "STATE\n" + \
                      "  power 0.0\n" + \
                      "  deplete DAYS 0.0\n"
    assert block.write_to_string(prefix="  ") == expected_output

def test_key_value_block_invalid_input(key_value_block):
    with pytest.raises(AssertionError):
        KeyValueBlock("")
    with pytest.raises(AssertionError):
        key_value_block.add("rated power", "1.0")
    with pytest.raises(AssertionError):
        key_value_block.add("!power", "1.0")
    with pytest.raises(AssertionError):
        key_value_block.add("power", "1.0 !MW")
    with pytest.raises(AssertionError):
        key_value_block.add("power", "1.0\n  boron 0.0")
    with pytest.raises(AssertionError):
        key_value_block.add("power", "1.0", comment="MW\n")
    with pytest.raises(AssertionError):
        key_value_block.add_comment("first\nsecond")
