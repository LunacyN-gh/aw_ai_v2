from __future__ import annotations

from dataclasses import dataclass, field, replace


PROFILE = "deterministic-v2"
PROPERTIES = frozenset(("city", "factory", "airport", "tower", "hq"))
INCOME_TILES = PROPERTIES - {"tower"}
DEFENSE = {"plain": 1, "road": 0, "forest": 2, "mountain": 4,
           "water": 0, "river": 0, "bridge": 0, "shoal": 0,
           "city": 3, "factory": 3, "airport": 3, "tower": 3, "hq": 4}
PROPERTY_VALUE = {"city": 4000, "factory": 8000, "airport": 6500,
                  "tower": 4500, "hq": 12000}


@dataclass(frozen=True, slots=True)
class Spec:
    cost: int
    move: int
    mode: str
    minimum: int = 1
    maximum: int = 1
    capture: bool = False
    indirect: bool = False


SPECS = {
    "infantry": Spec(1000, 3, "foot", capture=True),
    "mech": Spec(3000, 2, "mech", capture=True),
    "recon": Spec(4000, 8, "wheel"),
    "tank": Spec(7000, 6, "tread"),
    "md_tank": Spec(16000, 5, "tread"),
    "artillery": Spec(6000, 5, "tread", 2, 3, indirect=True),
    "rocket": Spec(15000, 5, "wheel", 3, 5, indirect=True),
    "anti_air": Spec(8000, 6, "tread"),
    "b_copter": Spec(9000, 6, "air"),
    "fighter": Spec(20000, 9, "air"),
    "bomber": Spec(22000, 7, "air"),
}
# No-luck base damage for the supported roster, from AWBW's damage chart.
# Combat timing/rounding/economy remain an experimental profile, not AWBW parity.
_ORDER = ("anti_air", "artillery", "b_copter", "bomber", "fighter", "infantry",
          "md_tank", "mech", "recon", "rocket", "tank")
_ROWS = (
    (45, 50, 120, 75, 65, 105, 10, 105, 60, 55, 25),
    (75, 75, 0, 0, 0, 90, 45, 85, 80, 80, 70),
    (25, 65, 65, 0, 0, 75, 25, 75, 55, 65, 55),
    (95, 105, 0, 0, 0, 110, 95, 110, 105, 105, 105),
    (0, 0, 100, 100, 55, 0, 0, 0, 0, 0, 0),
    (5, 15, 7, 0, 0, 55, 1, 45, 12, 25, 5),
    (105, 105, 12, 0, 0, 105, 55, 95, 105, 105, 85),
    (65, 70, 9, 0, 0, 65, 15, 55, 85, 85, 55),
    (4, 45, 10, 0, 0, 70, 1, 65, 35, 55, 6),
    (85, 80, 0, 0, 0, 95, 55, 90, 90, 85, 80),
    (65, 70, 10, 0, 0, 75, 15, 70, 85, 85, 55),
)
BASE_DAMAGE = {(a, d): _ROWS[i][j] for i, a in enumerate(_ORDER)
               for j, d in enumerate(_ORDER)}


@dataclass(frozen=True, slots=True, eq=False)
class Board:
    width: int
    height: int
    tiles: tuple[str, ...]
    neighbors: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    properties: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        if not 1 <= self.width <= 128 or not 1 <= self.height <= 128:
            raise ValueError("dimensions must be between 1 and 128")
        if len(self.tiles) != self.width * self.height or any(t not in DEFENSE for t in self.tiles):
            raise ValueError("invalid terrain grid")
        links = []
        for p in range(len(self.tiles)):
            x, y = self.xy(p)
            links.append(tuple(self.pos(a, b) for a, b in
                               ((x-1, y), (x+1, y), (x, y-1), (x, y+1))
                               if 0 <= a < self.width and 0 <= b < self.height))
        object.__setattr__(self, "neighbors", tuple(links))
        object.__setattr__(self, "properties", tuple(p for p, t in enumerate(self.tiles) if t in PROPERTIES))

    @classmethod
    def plain(cls, width=20, height=20, terrain=None):
        tiles = ["plain"] * (width * height)
        for (x, y), tile in (terrain or {}).items():
            if not 0 <= x < width or not 0 <= y < height:
                raise ValueError("terrain outside board")
            tiles[y * width + x] = tile
        return cls(width, height, tuple(tiles))

    def pos(self, x, y):
        return y * self.width + x

    def xy(self, p):
        return p % self.width, p // self.width

    def distance(self, a, b):
        ax, ay = self.xy(a)
        bx, by = self.xy(b)
        return abs(ax-bx) + abs(ay-by)


@dataclass(frozen=True, slots=True)
class Unit:
    id: str
    owner: int
    kind: str
    pos: int
    hp: int = 100
    acted: bool = False

    @property
    def spec(self):
        return SPECS[self.kind]

    @property
    def displayed_hp(self):
        return (self.hp + 9) // 10


@dataclass(frozen=True, slots=True)
class Action:
    kind: str
    actor: str = ""
    destination: int = -1
    target: str = ""
    build: str = ""


END = Action("end")


@dataclass(slots=True)
class State:
    board: Board
    units: dict[str, Unit]
    player: int = 0
    funds: list[int] = field(default_factory=lambda: [0, 0])
    owners: dict[int, int] = field(default_factory=dict)
    captures: dict[int, tuple[str, int]] = field(default_factory=dict)
    turn: int = 0
    winner: int | None = None
    next_id: int = 0
    profile: str = PROFILE
    income_capture_limit: int | None = None
    allowed_builds: tuple[str, ...] | None = None
    occupancy: dict[int, str] = field(init=False, repr=False)

    def __post_init__(self):
        self.occupancy = {u.pos: u.id for u in self.units.values()}

    def clone(self):
        return State(self.board, self.units.copy(), self.player, self.funds.copy(),
                     self.owners.copy(), self.captures.copy(), self.turn, self.winner,
                     self.next_id, self.profile, self.income_capture_limit, self.allowed_builds)

    def update(self, uid, **changes):
        old = self.units[uid]
        new = replace(old, **changes)
        if old.pos != new.pos:
            del self.occupancy[old.pos]
            self.occupancy[new.pos] = uid
        self.units[uid] = new

    def remove(self, uid):
        unit = self.units.pop(uid)
        del self.occupancy[unit.pos]
        self.clear_capture(uid)

    def clear_capture(self, uid, except_at=-1):
        for p, (capturer, _) in tuple(self.captures.items()):
            if capturer == uid and p != except_at:
                del self.captures[p]

    def army(self, player=None):
        return [u for u in self.units.values() if player is None or u.owner == player]

    def key(self):
        return (self.profile, self.board, self.player, tuple(self.funds), self.winner,
                self.next_id, self.income_capture_limit, self.allowed_builds, tuple(sorted(self.owners.items())), tuple(sorted(self.captures.items())),
                tuple(sorted((u.id, u.owner, u.kind, u.pos, u.hp, u.acted) for u in self.units.values())))

    def validate(self):
        if self.profile != PROFILE:
            raise ValueError("unsupported rules profile")
        if self.player not in (0, 1) or len(self.funds) != 2 or min(self.funds) < 0:
            raise ValueError("invalid player or funds")
        if self.winner not in (None, 0, 1) or self.turn < 0 or self.next_id < 0:
            raise ValueError("invalid turn or outcome")
        if self.allowed_builds is not None:
            if (not isinstance(self.allowed_builds, tuple) or not self.allowed_builds
                    or any(k not in SPECS for k in self.allowed_builds)
                    or len(set(self.allowed_builds)) != len(self.allowed_builds)):
                raise ValueError("invalid map production roster")
            if any(u.kind not in self.allowed_builds for u in self.units.values()):
                raise ValueError("unit excluded by map production roster")
        if self.income_capture_limit is not None:
            total = sum(t in INCOME_TILES for t in self.board.tiles)
            if (type(self.income_capture_limit) is not int
                    or not total//2 < self.income_capture_limit <= total):
                raise ValueError("income capture limit must be an integer above half and at most all income properties")
        if len(self.occupancy) != len(self.units):
            raise ValueError("multiple units on a tile")
        for uid, u in self.units.items():
            if not uid or u.id != uid or u.kind not in SPECS or u.owner not in (0, 1):
                raise ValueError("invalid unit")
            if not 0 <= u.pos < len(self.board.tiles) or not 1 <= u.hp <= 100:
                raise ValueError("invalid unit position or HP")
        for p, owner in self.owners.items():
            if p not in self.board.properties or owner not in (0, 1):
                raise ValueError("invalid property ownership")
        for p, (uid, remaining) in self.captures.items():
            u = self.units.get(uid)
            if (p not in self.board.properties or u is None or u.pos != p or not u.spec.capture
                    or not 1 <= remaining <= 20 or self.owners.get(p) == u.owner):
                raise ValueError("invalid capture")


def material(unit):
    return unit.spec.cost * (unit.hp / 100 + .15)
