from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


class CardBlock():
    """ An ordered collection of the input cards of an MPACT key/value block

    Blocks such as STATE, OPTION, XSEC, and TRANS are a list of cards, each of which
    is a keyword followed by its arguments.  This class keeps those cards in the order
    they were given, allows the same keyword to appear more than once
    (e.g. two `param` cards in an OPTION block), and keeps the comment attached to
    each card as well as comment-only lines (e.g. commented-out cards).

    Card values are stored as the raw argument text and are not validated, leaving
    validation to MPACT itself.  Keyword look-ups are not case sensitive, but keywords
    are written as they were originally given.

    Comments describe the value on their line, so setting a card to a different value
    drops its comment unless a new comment is given or the comment is explicitly kept.

    Parameters
    ----------
    name : str
        The name of the block (e.g. 'STATE')

    Attributes
    ----------
    name : str
        The name of the block (e.g. 'STATE')
    entries : List[Entry]
        The cards and comment-only lines of the block, in order
    """

    @dataclass
    class Entry():
        """ A dataclass for a single line of a card block

        Attributes
        ----------
        keyword : Optional[str]
            The card keyword (None for a comment-only line)
        value : str
            The card arguments as written after the keyword
            (or the comment text for a comment-only line)
        comment : Optional[str]
            The comment following the card on the same line, without the leading '!'
        """
        keyword: Optional[str]
        value:   str           = ""
        comment: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def entries(self) -> List[Entry]:
        return self._entries

    def __init__(self, name: str):

        assert len(name) > 0 and not any(c.isspace() for c in name), f"name = {name}"

        self._name    = name
        self._entries = []

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        return (isinstance(other, CardBlock)           and
                self.name.lower() == other.name.lower() and
                self._cards()     == other._cards()
               )

    def __hash__(self) -> int:
        return hash((self.name.lower(), tuple(self._cards())))

    def __contains__(self, keyword: str) -> bool:
        return self._find(keyword) is not None

    def __getitem__(self, keyword: str) -> str:
        entry = self._find(keyword)
        if entry is None:
            raise KeyError(keyword)
        return entry.value

    def __setitem__(self, keyword: str, value: str) -> None:
        self.set(keyword, value)

    def __delitem__(self, keyword: str) -> None:
        if keyword not in self:
            raise KeyError(keyword)
        self._entries = [entry for entry in self.entries if not self._matches(entry, keyword)]

    @staticmethod
    def from_dict(name: str, cards: Dict[str, str]) -> CardBlock:
        """ Method for creating a card block from a dictionary of keywords and values

        Parameters
        ----------
        name : str
            The name of the block (e.g. 'STATE')
        cards : Dict[str, str]
            The card values (key: keyword, value: card arguments)

        Returns
        -------
        CardBlock
            The card block with one card per dictionary item, in dictionary order
        """

        block = CardBlock(name)
        for keyword, value in cards.items():
            block.add(keyword, value)
        return block

    def get_all(self, keyword: str) -> List[str]:
        """ Method for getting the values of every card with a given keyword

        Parameters
        ----------
        keyword : str
            The card keyword

        Returns
        -------
        List[str]
            The values of the matching cards, in order
        """

        return [entry.value for entry in self.entries if self._matches(entry, keyword)]

    def set(self,
            keyword:      str,
            value:        str,
            comment:      Optional[str] = None,
            keep_comment: bool = False) -> None:
        """ Method for setting the value of a card

        The first card with the given keyword is updated, or a new card is added to the
        end of the block if there is no card with that keyword.  If an existing card is
        given a different value, its comment is dropped unless a new comment is provided
        or keep_comment is True.

        Parameters
        ----------
        keyword : str
            The card keyword
        value : str
            The card arguments
        comment : Optional[str]
            A new comment for the card, without the leading '!'
        keep_comment : bool
            Whether to keep the card's existing comment when its value changes
        """

        assert not (comment is not None and keep_comment), \
            "Cannot both provide a new comment and keep the existing comment"

        entry = self._find(keyword)
        if entry is None:
            self.add(keyword, value, comment)
            return

        self._check_card(keyword, value, comment)
        if comment is not None:
            entry.comment = comment
        elif value != entry.value and not keep_comment:
            entry.comment = None
        entry.value = value

    def add(self, keyword: str, value: str, comment: Optional[str] = None) -> None:
        """ Method for adding a card to the end of the block

        A card is added even if the block already has a card with the same keyword.

        Parameters
        ----------
        keyword : str
            The card keyword
        value : str
            The card arguments
        comment : Optional[str]
            A comment for the card, without the leading '!'
        """

        self._check_card(keyword, value, comment)
        self._entries.append(CardBlock.Entry(keyword, value, comment))

    def add_comment(self, text: str) -> None:
        """ Method for adding a comment-only line to the end of the block

        Parameters
        ----------
        text : str
            The comment text, without the leading '!'
        """

        assert "\n" not in text, f"text = {text}"
        self._entries.append(CardBlock.Entry(None, text))

    def write_to_string(self, prefix: str = "  ") -> str:
        """ Method for writing the card block to a string

        Parameters
        ----------
        prefix : str
            A prefix with which to start each card line.  MPACT requires card
            keywords to start after the first column, so this must not be empty.

        Returns
        -------
        str
            The string that represents the card block
        """

        assert len(prefix) > 0, "prefix must not be empty"

        string = f"{self.name}\n"
        for entry in self.entries:
            if entry.keyword is None:
                string += prefix + f"!{entry.value}\n"
                continue
            line = entry.keyword if len(entry.value) == 0 else f"{entry.keyword} {entry.value}"
            if entry.comment is not None:
                line += f" !{entry.comment}"
            string += prefix + line + "\n"
        return string

    def _cards(self) -> List[tuple]:
        return [(entry.keyword.lower(), entry.value) for entry in self.entries if entry.keyword is not None]

    def _find(self, keyword: str) -> Optional[Entry]:
        return next((entry for entry in self.entries if self._matches(entry, keyword)), None)

    @staticmethod
    def _matches(entry: Entry, keyword: str) -> bool:
        return entry.keyword is not None and entry.keyword.lower() == keyword.lower()

    @staticmethod
    def _check_card(keyword: str, value: str, comment: Optional[str]) -> None:
        assert len(keyword) > 0 and not any(c.isspace() for c in keyword), f"keyword = {keyword}"
        assert not keyword.startswith("!"), f"keyword = {keyword}"
        assert "\n" not in value and "!" not in value, f"value = {value}"
        assert comment is None or "\n" not in comment, f"comment = {comment}"
