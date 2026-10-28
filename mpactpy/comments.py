from __future__ import annotations
from typing import Any, Collection, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Comments():
    """ A dataclass for the comments of one line of an MPACT input card

    Attributes
    ----------
    leading : List[str]
        The comment-only lines directly above the line, without the leading '!'
    inline : Optional[str]
        The comment at the end of the line, without the leading '!'
    """

    leading: List[str]     = field(default_factory=list)
    inline:  Optional[str] = None

    def __post_init__(self):
        assert all("\n" not in text for text in self.leading), f"leading = {self.leading}"
        assert self.inline is None or "\n" not in self.inline, f"inline = {self.inline}"


def write_line(prefix: str, line: str, comments: Optional[Comments]) -> str:
    """ Function for writing a line of an input card with its comments

    Parameters
    ----------
    prefix : str
        A prefix with which to start the line and each of its leading comments
    line : str
        The line, without its comments or newline
    comments : Optional[Comments]
        The comments of the line

    Returns
    -------
    str
        The line with its comments
    """

    if comments is None:
        return prefix + line + "\n"
    string = "".join(prefix + f"!{text}\n" for text in comments.leading)
    inline = "" if comments.inline is None else f" !{comments.inline}"
    return string + prefix + line + inline + "\n"


def assert_keys_written(comments: Dict[Any, Comments], lines: Collection[Any]) -> None:
    """ Function for asserting that every comment of a card belongs to a line that is written

    Parameters
    ----------
    comments : Dict[Any, Comments]
        The comments of each line of the card (key: None for the card line)
    lines : Collection[Any]
        The keys of the other lines that are written
    """

    assert all(key is None or key in lines for key in comments), \
        f"comments = {comments}, expected keys None or {list(lines)}"
