#!/usr/bin/env python3
"""
WFN to Gaussian Cube converter (pure Python, no third-party deps)

- Input: Gaussian wavefunction file (classic AIMPAC-style .wfn). Basic support only.
- Output: Gaussian .cube file with total electron density (in bohr units).

Notes and limitations:
- WFN formats vary. This parser targets a common AIMPAC/Gaussian-style layout where
  basis functions are listed explicitly (S, PX, PY, PZ, Dxx, Dxy, ...), 
  followed by MO blocks with occupancies and coefficients.
- If your file is actually WFX (XML-like), this script will currently not parse it.
- Basis ordering matters. If the WFN ordering differs, densities will be wrong.
- No external libraries are used; performance is acceptable for modest grids/basis sizes.

Usage example:
  python wfn2cube.py --input input.wfn --output density.cube \
    --spacing 0.2 --padding 3.0 --max-points 6000000

"""

import argparse
import math
import os
import sys
from typing import List, Tuple, Optional, Dict


# ----------------------------- Math utilities -----------------------------

def double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    result = 1
    k = n
    while k > 1:
        result *= k
        k -= 2
    return result


def primitive_cart_normalization(alpha: float, lx: int, ly: int, lz: int) -> float:
    """Normalization for Cartesian Gaussian N * x^lx y^ly z^lz * exp(-alpha r^2).
    Reference: Szabo & Ostlund, standard Cartesian GTO normalization.
    """
    lsum = lx + ly + lz
    num = (2.0 ** (2 * lsum + 1.5)) * (alpha ** (lsum + 1.5))
    den = (math.pi ** 1.5) * double_factorial(2 * lx - 1) * double_factorial(2 * ly - 1) * double_factorial(2 * lz - 1)
    return math.sqrt(num / den)


# ----------------------------- Data structures -----------------------------

class Atom:
    def __init__(self, atomic_number: int, charge: float, x: float, y: float, z: float):
        self.atomic_number = int(atomic_number)
        self.charge = float(charge)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @property
    def coord(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class AOPrimitive:
    def __init__(self, center_index: int, lx: int, ly: int, lz: int, alpha: float, coeff: float = 1.0, normalized: bool = True):
        self.center_index = int(center_index)  # 0-based index into atoms list
        self.lx = int(lx)
        self.ly = int(ly)
        self.lz = int(lz)
        self.alpha = float(alpha)
        self.coeff = float(coeff)
        self.normalized = bool(normalized)

    def value(self, r: Tuple[float, float, float], center: Tuple[float, float, float]) -> float:
        dx = r[0] - center[0]
        dy = r[1] - center[1]
        dz = r[2] - center[2]
        rr = dx * dx + dy * dy + dz * dz
        poly = (dx ** self.lx) * (dy ** self.ly) * (dz ** self.lz)
        val = poly * math.exp(-self.alpha * rr)
        if self.normalized:
            n = primitive_cart_normalization(self.alpha, self.lx, self.ly, self.lz)
            val *= n
        return self.coeff * val


class AOFunction:
    """Represents an AO basis function. For classic AIMPAC WFN we treat each basis
    function as a single (possibly normalized) primitive (no contraction info is provided
    consistently in all variants). If contraction is needed, model it as multiple
    AOPrimitive entries and sum them in value().
    """
    def __init__(self, primitives: List[AOPrimitive]):
        self.primitives = primitives

    def value(self, r: Tuple[float, float, float], atoms: List[Atom]) -> float:
        total = 0.0
        for prim in self.primitives:
            center = atoms[prim.center_index].coord
            total += prim.value(r, center)
        return total


class MolecularOrbital:
    def __init__(self, energy_au: float, occupancy: float, spin: str, coefficients: List[float]):
        self.energy_au = float(energy_au)
        self.occupancy = float(occupancy)
        self.spin = spin  # 'Alpha' or 'Beta' or 'Restricted'
        self.coefficients = coefficients  # per-AO coefficient list


class Wavefunction:
    def __init__(self, atoms: List[Atom], aos: List[AOFunction], mos: List[MolecularOrbital]):
        self.atoms = atoms
        self.aos = aos
        self.mos = mos
        if len(mos) > 0 and len(aos) != len(mos[0].coefficients):
            raise ValueError(f"Inconsistent AO count: {len(aos)} vs MO coeff length {len(mos[0].coefficients)}")

    def build_density_matrix(self, drop_threshold: float = 1e-14) -> List[Tuple[int, int, float]]:
        """Build upper-triangular density matrix elements D[mu,nu] = sum_i occ_i c_{i,mu} c_{i,nu}.
        Returns a sparse list of (mu, nu, value) with mu <= nu and |value| > drop_threshold.
        """
        nao = len(self.aos)
        # Initialize dense row buffers as Python lists for simplicity
        D_upper: Dict[Tuple[int, int], float] = {}
        for mo in self.mos:
            occ = mo.occupancy
            if abs(occ) < 1e-12:
                continue
            coeffs = mo.coefficients
            # Outer product accumulation (upper triangle only)
            for mu in range(nao):
                c_mu = coeffs[mu]
                if c_mu == 0.0:
                    continue
                for nu in range(mu, nao):
                    c_nu = coeffs[nu]
                    if c_nu == 0.0:
                        continue
                    key = (mu, nu)
                    D_upper[key] = D_upper.get(key, 0.0) + occ * c_mu * c_nu
        # Sparsify
        items: List[Tuple[int, int, float]] = []
        for (mu, nu), val in D_upper.items():
            if abs(val) > drop_threshold:
                items.append((mu, nu, val))
        # Sort for deterministic order
        items.sort(key=lambda t: (t[0], t[1]))
        return items


# ----------------------------- Parsing (WFN) -----------------------------

class WFNParseError(Exception):
    pass


def _guess_is_wfx(buffer: str) -> bool:
    return "<WFX" in buffer or "<WaveFunction>" in buffer or "<MolecularOrbital" in buffer


def parse_wfn(path: str) -> Wavefunction:
    """Parse a Gaussian/AIMPAC-style WFN file.

    This function supports a commonly encountered simple variant where:
      - A block lists atoms (atomic number, effective charge, x y z) in bohr.
      - A block lists basis functions one-per-line with fields:
            center_index  label  exponent
        where label is one of: S, PX, PY, PZ, DXX, DYY, DZZ, DXY, DXZ, DYZ.
      - A sequence of MO blocks each providing occupancy, energy, spin, and
        a flat list of AO coefficients.

    If your file doesn't match, an error is raised describing the first mismatch.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f]
    except FileNotFoundError:
        raise WFNParseError(f"File not found: {path}")

    header = "\n".join(lines[:100]).upper()
    if _guess_is_wfx(header):
        raise WFNParseError("The input appears to be WFX (XML-like). This script currently supports classic WFN only.")

    # Very simple state machine driven by sentinel keywords
    idx = 0

    def skip_blank(i: int) -> int:
        while i < len(lines) and (lines[i].strip() == "" or lines[i].strip().startswith("#")):
            i += 1
        return i

    # Title (optional)
    idx = skip_blank(idx)
    if idx < len(lines):
        title = lines[idx].strip()
        idx += 1
    else:
        title = "WFN"

    # Seek atoms block
    atoms: List[Atom] = []
    # Try to find a line starting with 'ATOMS' or 'CENTERS'
    find_atoms = -1
    for j in range(idx, min(idx + 200, len(lines))):
        u = lines[j].strip().upper()
        if u.startswith("ATOMS") or u.startswith("CENTERS") or u.startswith("NATOMS"):
            find_atoms = j
            break
    if find_atoms == -1:
        # Fallback: assume next non-empty N lines starts with integer count then that many atoms
        # Try to parse pattern: NATOMS <int>
        nat = None
        for j in range(idx, min(idx + 50, len(lines))):
            toks = lines[j].split()
            if len(toks) >= 2 and toks[0].upper() in ("NATOMS", "NAT", "CENTERS", "ATOMS"):
                try:
                    nat = int(toks[1])
                    find_atoms = j
                    break
                except ValueError:
                    pass
        if nat is None:
            raise WFNParseError("Could not locate atoms block. Expected a line starting with ATOMS/CENTERS/NATOMS <N>.")
    idx = find_atoms

    # Read number of atoms and then NAT lines
    tokens = lines[idx].split()
    nat: Optional[int] = None
    for t in tokens:
        if t.isdigit():
            nat = int(t)
            break
    if nat is None:
        # Next line might be the NAT integer
        idx += 1
        try:
            nat = int(lines[idx].split()[0])
        except Exception as e:
            raise WFNParseError("Failed to read atom count in atoms block") from e
    # Move to first atom line
    idx += 1

    for _ in range(nat):
        if idx >= len(lines):
            raise WFNParseError("Unexpected EOF while reading atoms")
        parts = lines[idx].split()
        if len(parts) < 5:
            raise WFNParseError(f"Atom line too short: '{lines[idx]}'")
        try:
            # Accept formats like: Z charge x y z  OR  Z x y z (charge implicit=Z)
            z = int(float(parts[0]))
            if len(parts) == 5:
                charge = float(z)
                x, y, zc = float(parts[1]), float(parts[2]), float(parts[3])
            else:
                charge = float(parts[1])
                x, y, zc = float(parts[2]), float(parts[3]), float(parts[4])
        except Exception as e:
            raise WFNParseError(f"Failed parsing atom line: '{lines[idx]}'") from e
        atoms.append(Atom(z, charge, x, y, zc))
        idx += 1

    # Seek basis block
    idx = skip_blank(idx)
    find_basis = -1
    for j in range(idx, min(idx + 400, len(lines))):
        u = lines[j].strip().upper()
        if u.startswith("BASIS") or u.startswith("BASIS FUNCTIONS") or u.startswith("AOS") or u.startswith("PRIMITIVES"):
            find_basis = j
            break
    if find_basis == -1:
        # Heuristic: search for a line saying "NBF" or "BASIS COUNT"
        for j in range(idx, min(idx + 100, len(lines))):
            if "NBF" in lines[j].upper():
                find_basis = j
                break
    if find_basis == -1:
        raise WFNParseError("Could not locate basis/AO block header.")
    idx = find_basis

    # Extract number of AO lines
    line = lines[idx]
    nbasis: Optional[int] = None
    for t in line.replace(',', ' ').split():
        if t.isdigit():
            try:
                nbasis = int(t)
                break
            except ValueError:
                pass
    if nbasis is None:
        # Try to use next line
        try:
            nbasis = int(lines[idx + 1].split()[0])
            idx += 1
        except Exception as e:
            raise WFNParseError("Failed to determine number of basis functions") from e
    idx += 1

    # Parse nbasis AO lines: center label exponent
    def label_to_lmn(label: str) -> Tuple[int, int, int]:
        u = label.upper()
        if u == 'S':
            return (0, 0, 0)
        if u == 'PX' or u == 'X':
            return (1, 0, 0)
        if u == 'PY' or u == 'Y':
            return (0, 1, 0)
        if u == 'PZ' or u == 'Z':
            return (0, 0, 1)
        # D shell common labels
        if u in ('DXX', 'XX'):
            return (2, 0, 0)
        if u in ('DYY', 'YY'):
            return (0, 2, 0)
        if u in ('DZZ', 'ZZ'):
            return (0, 0, 2)
        if u in ('DXY', 'XY'):
            return (1, 1, 0)
        if u in ('DXZ', 'XZ'):
            return (1, 0, 1)
        if u in ('DYZ', 'YZ'):
            return (0, 1, 1)
        raise WFNParseError(f"Unsupported AO label: {label}")

    aos: List[AOFunction] = []
    for k in range(nbasis):
        if idx >= len(lines):
            raise WFNParseError("Unexpected EOF while reading basis functions")
        parts = lines[idx].split()
        if len(parts) < 3:
            raise WFNParseError(f"AO line too short: '{lines[idx]}'")
        try:
            # Accept forms: center label exponent  OR  index center label exponent ...
            if parts[0].isdigit() and parts[1].isdigit():
                center_idx = int(parts[1]) - 1  # often 1-based in files
                label = parts[2]
                alpha = float(parts[3]) if len(parts) > 3 else None
            else:
                center_idx = int(parts[0]) - 1
                label = parts[1]
                alpha = float(parts[2])
            if alpha is None:
                raise ValueError("missing alpha")
            lx, ly, lz = label_to_lmn(label)
        except Exception as e:
            raise WFNParseError(f"Failed parsing AO line: '{lines[idx]}'") from e
        prim = AOPrimitive(center_idx, lx, ly, lz, alpha, coeff=1.0, normalized=True)
        aos.append(AOFunction([prim]))
        idx += 1

    # Seek MO blocks
    idx = skip_blank(idx)
    mos: List[MolecularOrbital] = []

    def try_parse_float_from_line(line_text: str, keys: Tuple[str, ...]) -> Optional[float]:
        U = line_text.upper()
        for key in keys:
            pos = U.find(key)
            if pos >= 0:
                # Grab the first float after the key
                tail = line_text[pos + len(key):]
                for tok in tail.replace('=', ' ').replace(':', ' ').split():
                    try:
                        return float(tok)
                    except ValueError:
                        continue
        return None

    while idx < len(lines):
        u = lines[idx].strip().upper()
        if not u:
            idx += 1
            continue
        if not (u.startswith('MO') or 'ORBITAL' in u or 'EIGEN' in u or 'OCC' in u):
            # Skip unrelated sections
            idx += 1
            continue

        # We found an MO-like header; parse a small header window
        occ = None
        energy = None
        spin = 'Restricted'
        header_window = "\n".join(lines[idx: idx + 5])
        occ = try_parse_float_from_line(header_window, ("OCC", "OCCUP", "OCCUPANCY"))
        energy = try_parse_float_from_line(header_window, ("ENER", "EIG", "E=", "EV"))
        UHW = header_window.upper()
        if "ALPHA" in UHW:
            spin = 'Alpha'
        elif "BETA" in UHW:
            spin = 'Beta'

        # Move idx to the first coefficient line: seek a line that looks numeric and has many floats
        coeffs: List[float] = []
        m = idx
        while m < len(lines):
            text = lines[m].strip()
            if text == "":
                m += 1
                continue
            # Heuristic: a line with at least 3 floats is the start
            toks = text.replace('D', 'E').split()
            numfloats = 0
            for tok in toks:
                try:
                    float(tok)
                    numfloats += 1
                except ValueError:
                    pass
            if numfloats >= 3:
                break
            m += 1
        if m >= len(lines):
            break
        # Read subsequent lines until we have nbasis coefficients
        while m < len(lines) and len(coeffs) < nbasis:
            toks = lines[m].replace('D', 'E').split()
            for tok in toks:
                try:
                    coeffs.append(float(tok))
                except ValueError:
                    continue
            m += 1
        if len(coeffs) != nbasis:
            # Not a real MO block; skip ahead one line and continue
            idx += 1
            continue
        if occ is None:
            # Default: closed-shell guess
            occ = 2.0
        if energy is None:
            energy = 0.0
        mos.append(MolecularOrbital(energy, occ, spin, coeffs))
        idx = m

    if not mos:
        raise WFNParseError("No molecular orbitals with coefficients were found.")

    # If occupancies look all 0 or all 1, try a simple closed-shell guess using electron count
    occ_sum = sum(mo.occupancy for mo in mos)
    if occ_sum < 1e-6:
        # Guess electrons from atom charges
        nelec_guess = sum(int(round(a.charge)) for a in atoms)
        norb_occ = min(len(mos), nelec_guess // 2)
        for i, mo in enumerate(mos):
            mo.occupancy = 2.0 if i < norb_occ else 0.0

    return Wavefunction(atoms, aos, mos)


# ----------------------------- Grid and cube writer -----------------------------

def compute_bounding_box(atoms: List[Atom]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    xs = [a.x for a in atoms]
    ys = [a.y for a in atoms]
    zs = [a.z for a in atoms]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def compute_grid(atoms: List[Atom], spacing: float, padding: float) -> Tuple[Tuple[float, float, float], Tuple[int, int, int], Tuple[float, float, float]]:
    (xmin, ymin, zmin), (xmax, ymax, zmax) = compute_bounding_box(atoms)
    xmin -= padding
    ymin -= padding
    zmin -= padding
    xmax += padding
    ymax += padding
    zmax += padding
    nx = int(math.floor((xmax - xmin) / spacing)) + 1
    ny = int(math.floor((ymax - ymin) / spacing)) + 1
    nz = int(math.floor((zmax - zmin) / spacing)) + 1
    return (xmin, ymin, zmin), (nx, ny, nz), (spacing, spacing, spacing)


def write_cube_density(output_path: str,
                       wf: Wavefunction,
                       origin: Tuple[float, float, float],
                       grid_shape: Tuple[int, int, int],
                       spacing: Tuple[float, float, float],
                       density_matrix_ut: List[Tuple[int, int, float]],
                       max_points: Optional[int] = None,
                       progress: bool = True) -> None:
    """Stream density to a Gaussian cube file.

    - origin: (x0,y0,z0) in bohr
    - grid_shape: (nx,ny,nz)
    - spacing: (dx,dy,dz) in bohr
    - density_matrix_ut: list of (mu,nu,val) with mu<=nu (upper-triangular)
    - max_points: optional cap to avoid extremely large outputs (safety)
    """
    atoms = wf.atoms
    aos = wf.aos
    nx, ny, nz = grid_shape

    total_points = nx * ny * nz
    if max_points is not None and total_points > max_points:
        raise RuntimeError(f"Grid too large: {total_points} points exceeds --max-points {max_points}")

    # Open file for writing
    with open(output_path, 'w', encoding='utf-8') as fo:
        # Header (2 comment lines)
        fo.write("WFN to cube (density)\n")
        fo.write("Generated by wfn2cube.py (units: bohr)\n")
        # Atom count and origin
        fo.write(f"{len(atoms):5d} {origin[0]:14.6f} {origin[1]:14.6f} {origin[2]:14.6f}\n")
        # Grid vectors (aligned with axes)
        fo.write(f"{nx:5d} {spacing[0]:14.6f} {0.0:14.6f} {0.0:14.6f}\n")
        fo.write(f"{ny:5d} {0.0:14.6f} {spacing[1]:14.6f} {0.0:14.6f}\n")
        fo.write(f"{nz:5d} {0.0:14.6f} {0.0:14.6f} {spacing[2]:14.6f}\n")
        # Atom list: atomic number, charge, coordinates (bohr)
        for a in atoms:
            fo.write(f"{a.atomic_number:5d} {a.charge:14.6f} {a.x:14.6f} {a.y:14.6f} {a.z:14.6f}\n")

        # Pre-allocate AO buffer to reuse per grid point
        nao = len(aos)
        # Iterate grid in i-fastest (z) order as per cube
        vals_on_line = 0
        line_buf: List[str] = []

        x0, y0, z0 = origin
        dx, dy, dz = spacing

        # Progress setup
        printed = False
        total_written = 0

        for ix in range(nx):
            x = x0 + ix * dx
            if progress and nx > 1 and ix % max(1, nx // 20) == 0:
                pct = 100.0 * ix / max(1, nx - 1)
                print(f"Progress: {pct:5.1f}% (ix={ix}/{nx})", file=sys.stderr)
                printed = True
            for iy in range(ny):
                y = y0 + iy * dy
                for iz in range(nz):
                    z = z0 + iz * dz
                    r = (x, y, z)
                    # Evaluate all AOs at r
                    ao_values = [0.0] * nao
                    for mu in range(nao):
                        ao_values[mu] = aos[mu].value(r, atoms)
                    # Accumulate density using upper-triangular density matrix
                    rho = 0.0
                    for (mu, nu, dval) in density_matrix_ut:
                        if mu == nu:
                            rho += dval * ao_values[mu] * ao_values[nu]
                        else:
                            rho += 2.0 * dval * ao_values[mu] * ao_values[nu]
                    # Append to buffer (6 values per line)
                    line_buf.append(f"{rho:13.5e}")
                    vals_on_line += 1
                    total_written += 1
                    if vals_on_line == 6:
                        fo.write(" ".join(line_buf) + "\n")
                        line_buf.clear()
                        vals_on_line = 0
        if vals_on_line:
            fo.write(" ".join(line_buf) + "\n")
        if progress and printed:
            print("Progress: 100.0%", file=sys.stderr)


# ----------------------------- CLI -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Gaussian WFN (AIMPAC-style) to Gaussian cube (electron density). Units: bohr")
    parser.add_argument('-i', '--input', required=True, help='Path to input .wfn file (text)')
    parser.add_argument('-o', '--output', required=True, help='Path to output .cube file')
    parser.add_argument('--spacing', type=float, default=0.2, help='Grid spacing in bohr (default: 0.2)')
    parser.add_argument('--padding', type=float, default=3.0, help='Padding around molecule in bohr (default: 3.0)')
    parser.add_argument('--d-thresh', type=float, default=1e-12, help='Drop |D(mu,nu)| below this when building density matrix')
    parser.add_argument('--max-points', type=int, default=6_000_000, help='Abort if grid points exceed this (safety)')
    parser.add_argument('--quiet', action='store_true', help='Reduce progress output')

    args = parser.parse_args(argv)

    # Read and parse WFN
    try:
        wf = parse_wfn(args.input)
    except WFNParseError as e:
        print(f"WFN parse error: {e}", file=sys.stderr)
        return 2

    # Grid definition
    origin, (nx, ny, nz), spacing_vec = compute_grid(wf.atoms, args.spacing, args.padding)
    if not args.quiet:
        print(f"Atoms: {len(wf.atoms)}  AOs: {len(wf.aos)}  MOs: {len(wf.mos)}", file=sys.stderr)
        print(f"Grid: {nx} x {ny} x {nz}  (~{nx*ny*nz:,} points)", file=sys.stderr)
        print(f"Origin: {origin}", file=sys.stderr)
        print(f"Spacing: {spacing_vec} (bohr)", file=sys.stderr)

    # Density matrix
    D_ut = wf.build_density_matrix(drop_threshold=args.d_thresh)
    if not args.quiet:
        print(f"Density matrix nonzeros (upper-tri): {len(D_ut)}", file=sys.stderr)

    # Write cube
    try:
        write_cube_density(
            output_path=args.output,
            wf=wf,
            origin=origin,
            grid_shape=(nx, ny, nz),
            spacing=spacing_vec,
            density_matrix_ut=D_ut,
            max_points=args.max_points,
            progress=(not args.quiet),
        )
    except Exception as e:
        print(f"Failed writing cube: {e}", file=sys.stderr)
        return 3

    if not args.quiet:
        print(f"Wrote cube to: {args.output}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
