import pytest

from mpactpy import CardBlock


@pytest.fixture
def card_block():
    block = CardBlock("OPTION")
    block.add("bound_cond", "1 1 1 1 1 1")
    block.add("param",      "MOC_KERNEL SIK 2")
    block.add("param",      "SHIELD_MOC_KERNEL SIK 2")
    block.add_comment("simplifiedth T SS")
    block.add("conv_crit",  "1E-6 1E-4", comment="k and flux")
    return block

@pytest.fixture
def equal_card_block():
    block = CardBlock("option")
    block.add("BOUND_COND", "1 1 1 1 1 1", comment="different comment")
    block.add("param",      "MOC_KERNEL SIK 2")
    block.add("param",      "SHIELD_MOC_KERNEL SIK 2")
    block.add("conv_crit",  "1E-6 1E-4")
    return block

@pytest.fixture
def unequal_card_block():
    block = CardBlock("OPTION")
    block.add("bound_cond", "1 1 1 1 1 1")
    block.add("param",      "SHIELD_MOC_KERNEL SIK 2")
    block.add("param",      "MOC_KERNEL SIK 2")
    block.add("conv_crit",  "1E-6 1E-4")
    return block

def test_card_block_initialization(card_block):
    assert card_block.name == "OPTION"
    assert len(card_block.entries) == 5
    assert card_block.entries[3] == CardBlock.Entry(None, "simplifiedth T SS")
    assert card_block.entries[4] == CardBlock.Entry("conv_crit", "1E-6 1E-4", "k and flux")

def test_card_block_equality(card_block, equal_card_block, unequal_card_block):
    assert card_block == equal_card_block
    assert card_block != unequal_card_block
    assert CardBlock("STATE") != CardBlock("OPTION")

def test_card_block_hash(card_block, equal_card_block, unequal_card_block):
    assert hash(card_block) == hash(equal_card_block)
    assert hash(card_block) != hash(unequal_card_block)

def test_card_block_write_to_string(card_block):
    output = card_block.write_to_string(prefix="  ")
    expected_output = "OPTION\n" + \
                      "  bound_cond 1 1 1 1 1 1\n" + \
                      "  param MOC_KERNEL SIK 2\n" + \
                      "  param SHIELD_MOC_KERNEL SIK 2\n" + \
                      "  !simplifiedth T SS\n" + \
                      "  conv_crit 1E-6 1E-4 !k and flux\n"
    assert output == expected_output

    block = CardBlock("TRANS")
    block.add("enable", "")
    assert block.write_to_string(prefix="  ") == "TRANS\n" + \
                                                 "  enable\n"

    with pytest.raises(AssertionError):
        card_block.write_to_string(prefix="")

def test_card_block_access(card_block):
    assert card_block["param"] == "MOC_KERNEL SIK 2"
    assert card_block["PARAM"] == "MOC_KERNEL SIK 2"
    assert card_block.get_all("param") == ["MOC_KERNEL SIK 2", "SHIELD_MOC_KERNEL SIK 2"]
    assert card_block.get_all("ray") == []
    assert "Conv_Crit" in card_block
    assert "ray" not in card_block
    assert "simplifiedth" not in card_block

    with pytest.raises(KeyError):
        card_block["ray"]

def test_card_block_set(card_block):
    card_block["solver"] = "1 2"
    assert card_block.entries[-1] == CardBlock.Entry("solver", "1 2")

    card_block["PARAM"] = "MOC_KERNEL LS 2"
    assert card_block.get_all("param") == ["MOC_KERNEL LS 2", "SHIELD_MOC_KERNEL SIK 2"]
    assert card_block.entries[1].keyword == "param"

    card_block.set("ray", "0.05 CHEBYSHEV-YAMAMOTO 16 3", comment="ray spacing")
    assert card_block.entries[-1] == CardBlock.Entry("ray", "0.05 CHEBYSHEV-YAMAMOTO 16 3", "ray spacing")

def test_card_block_set_comment_rule(card_block):
    card_block["conv_crit"] = "1E-6 1E-4"
    assert card_block.entries[4].comment == "k and flux"

    card_block.set("conv_crit", "1E-7 1E-5", keep_comment=True)
    assert card_block.entries[4].comment == "k and flux"

    card_block.set("conv_crit", "1E-8 1E-6", comment="tighter")
    assert card_block.entries[4].comment == "tighter"

    card_block["conv_crit"] = "1E-9 1E-7"
    assert card_block.entries[4].comment is None

    with pytest.raises(AssertionError):
        card_block.set("conv_crit", "1E-6 1E-4", comment="new", keep_comment=True)

def test_card_block_delete(card_block):
    del card_block["param"]
    assert "param" not in card_block
    assert len(card_block.entries) == 3

    with pytest.raises(KeyError):
        del card_block["param"]

def test_card_block_from_dict():
    block = CardBlock.from_dict("STATE", {"power":   "0.0",
                                          "deplete": "DAYS 0.0"})
    expected_output = "STATE\n" + \
                      "  power 0.0\n" + \
                      "  deplete DAYS 0.0\n"
    assert block.write_to_string(prefix="  ") == expected_output

def test_card_block_invalid_input(card_block):
    with pytest.raises(AssertionError):
        CardBlock("")
    with pytest.raises(AssertionError):
        card_block.add("rated power", "1.0")
    with pytest.raises(AssertionError):
        card_block.add("!power", "1.0")
    with pytest.raises(AssertionError):
        card_block.add("power", "1.0 !MW")
    with pytest.raises(AssertionError):
        card_block.add("power", "1.0\n  boron 0.0")
    with pytest.raises(AssertionError):
        card_block.add("power", "1.0", comment="MW\n")
    with pytest.raises(AssertionError):
        card_block.add_comment("first\nsecond")
